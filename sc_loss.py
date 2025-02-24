'''
@author: James V. Talwar
Created on 2/24/2025

About: sc_loss.py contains an implementation for supervised contrastive loss (for both distributed and non-distributed training),  
along with any accessory functions invoked in the implementation.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
About: Method to gather projection network features across GPUs to a single tensor. 
'''
def GatherFeatures(featureTensor):
    #Output feature tensor should be (B * WorldSize) x d
    allFeatures = torch.zeros(featureTensor.shape[0] * torch.distributed.get_world_size(), 
                              featureTensor.shape[1], dtype = featureTensor.dtype, 
                              device = featureTensor.device)
    
    torch.distributed.all_gather_into_tensor(output_tensor= allFeatures, input_tensor = featureTensor)
    
    return allFeatures 

@torch.no_grad()
def GatherLabels(labelTensor):
    gatherLabelList = [torch.ones_like(labelTensor) for i in range(torch.distributed.get_world_size())]

    torch.distributed.all_gather(tensor_list = gatherLabelList,
                                 tensor = labelTensor)
    
    allLabels = torch.cat(gatherLabelList, dim = 0).to(labelTensor.device)

    return allLabels

'''
About: Method to subtract the maximum value along the last dimension of a tensor. Called for numerical stability
before calling log_softmax()
'''
def StabilizeLogits(logits):
    maxLogits, _ = torch.max(logits, dim = -1, keepdim = True)
    logits = logits - maxLogits.detach()
    
    return logits

'''
About: Helper method for computing supervised contrastive loss. Computes the log softmax of the self-ablated logits
       (which handles the interior term of loss), then sums the relevant positives (i.e. instances of the same label)
       and normalizes by the number of positives. Finally loss is multiplied by -1 and normalization by the number of 
       samples (in local batch).

INPUTS: logits: Tensor (dtype = float) of logits (dot product of features and allFeatures/temperature). Diagonal values (i.e. self) are filled
                with min value/large negative number to mask contribution. Dimensions: B x (B*WorldSize)
        positives: Tensor (dtype = float) of all labels that match a given point's label. Normalized by the number of positive instances. 
                   Dimensions: B x (B*WorldSize)

OUTPUT: loss: Tensor (dtype = Float) corresponding to supervised contrastive loss value. Dimensions: 1
'''
def CalculateCrossEntropy(logits, positives):
    interior = F.log_softmax(logits, dim = -1)
    summedAndNormalized = torch.sum(positives * interior, dim = -1) # Dim: [B]; Element-wise multiplication
    loss = - summedAndNormalized.mean() 

    return loss

class SupCon(nn.Module):
    '''
    Supervised contrastive loss as reported by: https://arxiv.org/pdf/2004.11362.pdf 

    Helpful repositories: 1) https://github.com/google-research/syn-rep-learn/blob/main/StableRep/models/losses.py#L49
                          2) https://github.com/HobbitLong/SupContrast/blob/master/losses.py 

    INPUTS:
    temperature: Float corresponding to the temperature to use in the supervised contrastive loss function
    distributed: Boolean corresponding as to whether training is conducted with DDP or not.
    device_rank: Integer corresponding to GPU (global) rank. Default: -1 (for non-distributed case)
     
    '''
    def __init__(self, temperature = 0.07, distributed = False, device_rank = -1): 
        super(SupCon, self).__init__() 
        self.distributed = distributed
        self.temperature = temperature
        self.device_rank = device_rank if distributed else -1
    
    def forward(self, features, labels):
        # features: [B x d] 
        # labels: [B x 1]
        
        batchSize = features.shape[0]

        # 1) Normalized output of projection network to lie on the unit hypersphere (as specified in paper)
        features = F.normalize(features, dim = -1, p = 2) 

        # 2) Gather all features and labels onto current device (if distributed)
        if self.distributed:
            allFeatures = GatherFeatures(features) #(B * WorldSize) x d
            allLabels = GatherLabels(labels) # (B * WorldSize) x 1
        else: 
            allFeatures = features
            allLabels = labels

        #print(f"Device {self.device_rank} labels shape {labels.shape}")
        #print(f"Device {self.device_rank} allLabels shape {allLabels.shape}")

        # 3) Define a mask for labels of the same class: B x (B * WorldSize)
        #  - for every row in labels (on current device), check against all labels to see if same
        mask = torch.eq(labels.view(-1, 1), allLabels.contiguous().view(1, -1)).float().to(features.device) 

        #print(f"Device {self.device_rank} mask: {mask}")
        #print(f"Device {self.device_rank} Mask shape {mask.shape}")

        # 4) Define a mask for anchors: for all N points, only N-1 points should be included (i.e., omit self in computation)
        #   - anchorMask shape: [B x B * WorldSize]
        if self.distributed:
            anchorMask = torch.scatter(torch.ones_like(mask),  1, torch.arange(mask.shape[0]).view(-1, 1).to(features.device) + batchSize * self.device_rank,  0)
        else:
            anchorMask = torch.ones_like(mask).to(features.device) - torch.eye(mask.shape[0]).to(features.device)

        #print(f"Device {self.device_rank} anchor mask: {anchorMask}")

        # 5) "Combine" masks to capture those values either not of the same label or self
        mask = mask * anchorMask

        #print(f"Device {self.device_rank} combined mask: {mask}")
        #print(f"Device {self.device_rank} anchor mask shape {anchorMask.shape}")

        # 6) Compute logits
        logits = torch.matmul(features, allFeatures.T) / self.temperature
        
        #print(f"Device {self.device_rank} logits shape {logits.shape}")

        # 7) Remove/Exclude self logits contributions
        logits = logits - (1 - anchorMask) * torch.finfo(logits.dtype).max 

        #print(f"Device {self.device_rank} logits pre-stabilized: {logits}")

        # 8) Stabilize logits for numerical stability (i.e., subtract max along dim = -1; LSE trick); Potentially optional... (see helpful repo 1 above)
        #  - Note: Max possible value given feature normalization is 1/self.temperature
        logits = StabilizeLogits(logits)

        #print(f"Device {self.device_rank} logits post-stabilized: {logits}")

        # 9) Normalize mask by the number of labels matching self (and clamping denom min to 1 to prevent div by 0)
        mask = mask/mask.sum(dim = -1, keepdim = True).clamp(min = 1)

        #print(f"Device {self.device_rank} normalized mask: {mask}")

        # 10) Calculate SupCon loss 
        loss = CalculateCrossEntropy(logits = logits, positives = mask)

        return loss  