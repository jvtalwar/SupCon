import argparse
import logging
import os
import datetime
from sc_loss import SupCon
from utils import Prepare_Distributed_Loader, Prepare_Datasets, Initialize_Optimizer, Get_Device, Initialize_Model_Checkpoint_Dir
from resnet_models import ResNet

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


'''
About: Method to initialize distributed training with DDP.
'''
def DDP_Setup():
    #Increase default timeout to allow for time for full dataset (CIFAR10) downloading/caching
    init_process_group(backend = "nccl", timeout = datetime.timedelta(minutes = 45))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def Train(model, loader, optimizer, loss_fx, gpu_id, epoch):
    model.train()
    device = Get_Device(gpu_id)

    train_loss = 0
    num_points_seen = 0
    average_gradient = 0
    for batch_idx, (data, target) in enumerate(loader):
        #track the batch size seen for correct epoch normalization; len(loader.dataset) can be incorrect with DDP due to padding 
        num_points_seen += target.shape[0]
        
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        output = model(data)
        
        loss = loss_fx(output, target.unsqueeze(1))
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * data.shape[0]
        
        gradient_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2).to('cpu') for p in model.parameters() if p.grad is not None]), 2)

        if gpu_id != -1:
            # Track gradients - need to sum across GPUs;
            
            gradient_norm = gradient_norm.to(device)
            torch.distributed.allreduce(gradient_norm, op = torch.distributed.ReduceOp.SUM)

        if gpu_id in {0, -1}:
            # Note: average_gradient is a sum of norms across GPUs (in distributed case) as opposed to the true average 
            # gradient for batch (as for the single GPU procedure), but still gives sense of training progress.
            average_gradient += gradient_norm.item()/len(loader)

    if gpu_id != -1:
        train_loss = torch.tensor(train_loss).to(device)
        torch.distributed.reduce(train_loss, dst = 0, op = torch.distributed.ReduceOp.SUM) #Get total loss and send to rank 0

        if gpu_id == 0:
            train_loss = train_loss/(num_points_seen * torch.distributed.get_world_size())
            train_loss = train_loss.item()
    
    else:
        train_loss = train_loss/num_points_seen

    logging.info(f"EPOCH {epoch}:\n-TRAIN loss: {train_loss:.6f}\n-Average gradient: {average_gradient:.6f}\n")

    return train_loss


@torch.no_grad()
def Validate(model, loader, loss_fx, gpu_id):
    model.eval()
    device = Get_Device(gpu_id)

    val_loss = 0
    num_points_seen = 0
    for batch_idx, (data, target) in enumerate(loader):
        #track the batch size seen for correct epoch normalization; len(loader.dataset) can be incorrect with DDP due to padding 
        num_points_seen += target.shape[0] 
        
        data, target = data.to(device), target.to(device)
        output = model(data)
        
        loss = loss_fx(output, target.unsqueeze(1))
        val_loss += loss.item() * data.shape[0]

    if gpu_id != -1:
        val_loss = torch.tensor(val_loss).to(device)
        torch.distributed.reduce(val_loss, dst = 0, op = torch.distributed.ReduceOp.SUM) #Get total loss and send to rank 0

        if gpu_id == 0:
            val_loss = val_loss/(num_points_seen * torch.distributed.get_world_size())
            val_loss = val_loss.item()
    
    else:
        val_loss = val_loss/num_points_seen
    
    return val_loss


def main(args):
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    torch.manual_seed(3)
    train_dataset, val_dataset = Prepare_Datasets()

    loss_tracker = pd.DataFrame(columns = ["Train_Loss", "Val_Loss"])
    model_name = f"resnet-{args.resnet}"
    pretrain_save_path, _ = Initialize_Model_Checkpoint_Dir(is_distributed = args.distributed, model_name = model_name)
    model = ResNet(which = args.resnet)

    if args.distributed:
        logger.info("Distributed training enabled. Setting up DDP...")

        DDP_Setup()
        gpu_id = int(os.environ["LOCAL_RANK"])

        #Correct batch size to world size
        batch_size = args.batch_size // torch.distributed.get_world_size()

        train_loader = Prepare_Distributed_Loader(dataset = train_dataset, shuffle = True, batch_size = batch_size, num_workers = args.num_workers)
        val_loader = Prepare_Distributed_Loader(dataset = val_dataset, shuffle = False, batch_size = batch_size, num_workers = args.num_workers)
        
        model = model.to(gpu_id)
        model = DDP(model, device_ids = [gpu_id])
        optimizer = Initialize_Optimizer(optimizer_name = args.optimizer, model_params = model.parameters(), learning_rate = args.learning_rate)

    else:
        logger.info("Distributed training disabled. Training on single GPU (or CPU if GPU unavailable)...")

        gpu_id = -1
        train_loader = DataLoader(train_dataset, shuffle = True, batch_size = args.batch_size, num_workers = args.num_workers)
        val_loader = DataLoader(val_dataset, shuffle = False, batch_size = args.batch_size, num_workers = args.num_workers)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        optimizer = Initialize_Optimizer(optimizer_name = args.optimizer, model_params = model.parameters(), learning_rate = args.learning_rate)
        
    loss_fx = SupCon(temperature = 0.1, distributed = args.distributed, device_rank = gpu_id)

    #Get baseline performance
    starting_train_loss = Validate(model = model, loader = train_loader, loss_fx = loss_fx, gpu_id = gpu_id)
    starting_val_loss = Validate(model = model, loader = val_loader, loss_fx = loss_fx, gpu_id = gpu_id)
    
    if gpu_id in {0, -1}:
        loss_tracker.loc[0, "Train_Loss"] = starting_train_loss
        loss_tracker.loc[0, "Val_Loss"] = starting_val_loss

        logger.info(f"\nEPOCH 0 (untrained model):\n-TRAIN loss: {starting_train_loss:.6f}\n-VAL loss: {starting_val_loss:.6f}\n")
        if args.distributed:
            torch.save(model.module.state_dict(), os.path.join(pretrain_save_path ,"Epoch_0.pt"))
        else:
            torch.save(model.state_dict(), os.path.join(pretrain_save_path ,"Epoch_0.pt"))

    #Commence training:
    for epoch in range(1, args.epochs + 1):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        train_loss = Train(model = model, loader = train_loader, optimizer = optimizer, loss_fx = loss_fx, gpu_id = gpu_id, epoch = epoch)
        val_loss = Validate(model = model, loader = val_loader, loss_fx = loss_fx, gpu_id = gpu_id) 
        
        logger.info(f"EPOCH {epoch} VAL loss: {val_loss:.6f}")

        if gpu_id in {0, -1}:
            #Save models and checkpoint losses
            loss_tracker.loc[epoch, "Train_Loss"] = train_loss
            loss_tracker.loc[epoch, "Val_Loss"] = val_loss
            if args.distributed:
                torch.save(model.module.state_dict(), os.path.join(pretrain_save_path , f"Epoch_{epoch}.pt"))
            else:
                torch.save(model.state_dict(), os.path.join(pretrain_save_path , f"Epoch_{epoch}.pt"))

    #Write losses:
    os.makedirs(args.training_summary_path, exist_ok = True)
    ddp_extension = "_DDP" if args.distributed else ""
    loss_tracker.to_csv(os.path.join(os.path.abspath(args.training_summary_path), f"{model_name + ddp_extension}_pretraining_summary.csv"), sep = "\t")

    if args.distributed:
        destroy_process_group()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    path_args = parser.add_argument_group("Pretraining arguments:")

    path_args.add_argument('--epochs', type=int, help='Number of epochs for which to pretrain. (Default: 125)', default = 125)
    path_args.add_argument('--resnet', type=int, help='''Resnet model for which to conduct pretraining on. Valid options include: {18, 50, 101}.
                           (Default: 18 -> resnet-18).''', default = 18)
    path_args.add_argument('--batch_size', type=int, help='Batch size for training and validation. (Default: 256)', default = 256)
    path_args.add_argument('--learning_rate', type = float, help = "Learning rate for model training. (Default: 1e-3)", default = 1e-3)
    path_args.add_argument("--num_workers", type = int, help = "Number of workers for train/val dataloaders. (Default: 2)", default = 2)
    path_args.add_argument("--distributed", action = "store_true", default = False, help = "Flag for whether to implement distributed training with torch DDP.")
    path_args.add_argument("--model_save_path", type = str, help = "Path to save model checkpoints. (Default: './trained_models')", default = "./trained_models")
    path_args.add_argument("--training_summary_path", type = str, help = "Path to save training summary metrics (e.g., train/val losses).", default = "./training_summary")
    path_args.add_argument("--optimizer", type = str, help = """Optimizer to use for training. 
                           Supported options: {'RMSprop', 'AdamW', 'Adam', 'SGD'}. (Default: 'RMSprop')""", 
                           default = "RMSprop") 

    args = parser.parse_args()
    
    main(args)


    
