#! /data/cxli/miniconda3/envs/th200/bin/python
import argparse
import os
from glob import glob
import random
import numpy as np
import pandas as pd

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import OrderedDict

import archs

from dataset import Dataset
from metrics import iou_score
from metrics import calculate_metrics
from utils import AverageMeter
from albumentations import RandomRotate90,Resize
import time

from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None, help='model name')
    parser.add_argument('--output_dir', default='outputs', help='ouput dir')
    parser.add_argument('--test_size', default=0.2,type=float, help='test dataset size')     
    args = parser.parse_args()

    return args

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    seed_torch()
    args = parse_args()
    class_to_gray = {0: 0, 1: 50, 2: 100, 3: 150} # class to gray value and get it back
    with open(f'{args.output_dir}/{args.name}/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)
    if args.name is not None:
        config['name'] = args.name
    config['batch_size'] = 1 # to get the output of each image
    cudnn.benchmark = True
     
    model = archs.__dict__[config['arch']](config['num_classes'], config['input_channels'], config['deep_supervision'], embed_dims=config['input_list'])

    model = model.cuda()

    dataset_name = config['dataset']
    img_ext = '.png'
    print("dataset_name:", dataset_name)
    if dataset_name == 'TYPE1':
        mask_ext = '.png'
    elif dataset_name == 'TYPE2':
        mask_ext = '.png'
    elif dataset_name == 'TYPE3':
        mask_ext = '.png'

    # Data loading code
    img_ids = sorted(glob(os.path.join(config['data_dir'], config['dataset'], 'images', '*' + img_ext)))
    # img_ids.sort()
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    _, val_img_ids = train_test_split(img_ids, test_size=args.test_size, random_state=config['dataseed'])

    ckpt = torch.load(f'{args.output_dir}/{args.name}/model.pth')
    # ckpt = torch.load(f'{output_dir}/{name}/model.pth',map_location=torch.device('cpu')) 
    try:        
        model.load_state_dict(ckpt)
    except:
        print("Pretrained model keys:", ckpt.keys())
        print("Current model keys:", model.state_dict().keys())

        pretrained_dict = {k: v for k, v in ckpt.items() if k in model.state_dict()}
        current_dict = model.state_dict()
        diff_keys = set(current_dict.keys()) - set(pretrained_dict.keys())

        print("Difference in model keys:")
        for key in diff_keys:
            print(f"Key: {key}")

        model.load_state_dict(ckpt, strict=False)
        
    model.eval()

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(config['data_dir'], config['dataset'], 'images'),
        mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False,
        pin_memory=True)

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    hd95_avg_meter = AverageMeter()

    col_names = ['img_id'] + ['iou_{}'.format(i) for i in range(config['num_classes'])] \
                    + ['dice_{}'.format(i) for i in range(config['num_classes'])]
    df_iou_dice = pd.DataFrame(columns=col_names)
    count = 0
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()
            model = model.cuda()
            # compute output
            output = model(input)

            iou, dice, hd95_ = iou_score(output, target)
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))
            hd95_avg_meter.update(hd95_, input.size(0))

            output = torch.sigmoid(output).cpu().numpy() # (batch_size, num_class, 512, 512)
            # output[output>=0.5]=1
            # output[output<0.5]=0
            df_iou_dice.loc[count] = meta['img_id'] + calculate_metrics(output,target)
            count += 1
            
            os.makedirs(os.path.join(args.output_dir, config['name'], 'out_val'), exist_ok=True)
            for pred, img_id in zip(output, meta['img_id']):
                # Get the class with the highest probability for each pixel
                class_map = np.argmax(pred, axis=0)
                # Create a grayscale image based on the class map
                grayscale_image = np.zeros((class_map.shape[0],class_map.shape[1]), dtype=np.uint8)
                for class_idx, gray_value in class_to_gray.items():
                     grayscale_image[class_map == class_idx] = gray_value
                # pred_np = pred[0].astype(np.uint8)
                # pred_np = pred_np * 255
                # img = Image.fromarray(pred_np, 'L')
                img = Image.fromarray(grayscale_image, 'L')
                img.save(os.path.join(args.output_dir, config['name'], 'out_val/{}.jpg'.format(img_id)))

    df_iou_dice.to_csv(os.path.join(args.output_dir, config['name'], 'iou_dice_val.csv'), index=False)
    print(config['name'])
    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)
    print('HD95: %.4f' % hd95_avg_meter.avg)



if __name__ == '__main__':
    main()
