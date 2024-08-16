import numpy as np
import torch
import torch.nn.functional as F

from medpy.metric.binary import jc, dc, hd, hd95, recall, specificity, precision



def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5 # probability > 0.5 as true, its size is (batch_size, num_class, 512, 512)
    target_ = target > 0.5 # 0 or 1 for binary mask
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2* iou) / (iou+1)
    
    try:
        hd95_ = hd95(output_, target_)
    except:
        hd95_ = 0
    
    return iou, dice, hd95_

def calculate_metrics(output, target, num_classes=4):
    
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5 # probability > 0.5 as true, its size is (batch_size, num_class, 512, 512)
    target_ = target > 0.5 # 0 or 1 for binary mask

    # Initialize lists to store IoU and Dice scores for each class
    iou_scores = []
    dice_scores = []
    
    # Loop through each class (axis=1 in the shape (2, 4, 512, 512))
    for class_idx in range(num_classes):
        # Extract the binary mask for the current class
        output_class = output_[:, class_idx, :, :]
        target_class = target_[:, class_idx, :, :]
        
        # Calculate intersection and union
        intersection = np.logical_and(output_class, target_class).sum()
        union = np.logical_or(output_class, target_class).sum()
        
        # Calculate IoU
        iou = (intersection + smooth) / (union + smooth)
        iou_scores.append(iou)
        
        # Calculate Dice
        dice = (2 * iou) / (iou + 1)
        dice_scores.append(dice)
        #keep 4 decimal places
    iou_scores = [round(score, 4) for score in iou_scores]
    dice_scores = [round(score, 4) for score in dice_scores]
    return iou_scores + dice_scores

def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

def indicators(output, target):
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5

    iou_ = jc(output_, target_)
    dice_ = dc(output_, target_)
    hd_ = hd(output_, target_)
    hd95_ = hd95(output_, target_)
    recall_ = recall(output_, target_)
    specificity_ = specificity(output_, target_)
    precision_ = precision(output_, target_)

    return iou_, dice_, hd_, hd95_, recall_, specificity_, precision_
