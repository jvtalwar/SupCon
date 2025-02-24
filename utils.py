'''
@author: James V. Talwar
Created on 2/24/2025

About: Helper functions and classes for SupCon pretraining and linear classifier cross-entropy training.
'''
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
import torch.optim as optim

'''
About: Prepare DDP compatible loader, employing DistributedSampler. Note that with DistributedSampler, shuffle is set to False, and for 
shuffling the loader, sampler is instantiated at each epoch (with set_epoch) to ensure shuffling between epochs.

Input(s): 1) dataset: SNP_Dataset object corresponding to underlying data for loader
          2) shuffle: Boolean corresponding to whether to shuffle the data. In DDP, this is passed to sampler.
Output(s): Dataloader with distributed sampler for training and validating with DDP
'''
def Prepare_Distributed_Loader(dataset, shuffle, **kwargs):
    loader = DataLoader(dataset, pin_memory = True, shuffle = False, sampler = DistributedSampler(dataset, shuffle = shuffle), **kwargs)
    
    return loader 

'''
About: Class for creating two crops of the same image as specified by paper. Class taken directly here from HobbitLong SupCon repository.
'''
class TwoCropTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

'''
About: Method to (download if needed) and initialize training and validation sets from CIFAR10.
'''
def Prepare_Datasets(download_path = "./CIFAR10_data", crop_size = 32):
    # CIFAR10 specific training statistics (corrected train standard deviation from original repo).
    train_mean = (0.4914, 0.4822, 0.4465)
    train_std = (0.2470, 0.2435, 0.2616)

    normalize = transforms.Normalize(mean = train_mean, std = train_std)

    #Initialize train transforms according to augmentations defined in paper/repo
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(size = crop_size, 
                                           scale = (0.2, 1.)),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p = 0.8),
                                           transforms.RandomGrayscale(p = 0.2),
                                           transforms.ToTensor(),
                                           normalize])
    
    val_transforms = transforms.Compose([transforms.ToTensor(), normalize])

    train_dataset = datasets.CIFAR10(root = download_path,
                                     train = True, 
                                     transform = TwoCropTransform(train_transforms),
                                     download = True)

    val_dataset = datasets.CIFAR10(root = download_path, train = False, transform = val_transforms)

    return train_dataset, val_dataset


'''
About: Method to initialize a specified optimizer.

Inputs: 1) optimizer_name: String corresponding to desired optimizer
        2) model_params: Model parameters to be passed to the optimizer
        3) learning_rate: Float corresponding to the desired learning rate.
Output: A torch optimizer corresponding to specified optimizer_name.
'''
def Initialize_Optimizer(optimizer_name, model_params, learning_rate, **kwargs):
    optimizerOptions = {"AdamW": optim.AdamW,
                        "Adam": optim.Adam, 
                        "SGD": optim.SGD,
                        "RMSprop": optim.RMSprop}
    
    if optimizer_name in optimizerOptions:
        return optimizerOptions[optimizer_name](model_params, lr = learning_rate, **kwargs)
    
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' invalid. Valid optimizer_name options are: 'AdamW', 'Adam', 'SGD', and 'RMSprop'.")

'''
About: Method to initialize model checkpoint directories for pretraining and linear classifier training. 
       Returns paths for pretraining and linear classifier checkpoints.

Inputs: 1) is_distributed: Boolean corresponding to whether training is conducted with DDP or not.
        2) model_name: String corresponding to the name of the model being trained.
        3) model_save_path: String corresponding to the root directory of where trained model checkpoints will be saved.

Output: Paths to save pretraining and linear classifier checkpoints.
'''
def Initialize_Model_Checkpoint_Dir(is_distributed, model_name, model_save_path = "./trained_models"):
    distributed_extension = "_DDP" if is_distributed else ""

    pretrain_save_path = os.path.join(model_save_path, f"pretrain/{model_name + distributed_extension}")
    linear_save_path = os.path.join(model_save_path, f"linear/{model_name + distributed_extension}")

    os.makedirs(pretrain_save_path, exist_ok = True)
    os.makedirs(linear_save_path, exist_ok = True)

    return os.path.abspath(pretrain_save_path), os.path.abspath(linear_save_path)
'''
About: Method to get device for training. If gpu_id is -1, then training is conducted without DDP and device is set to cuda (or cpu if cuda unavailable). 
Otherwise, device is set to gpu_id of gpu rank of process.
'''
def Get_Device(gpu_id):
    if gpu_id == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = gpu_id

    return device