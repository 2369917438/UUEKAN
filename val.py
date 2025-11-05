#! /data/cxli/miniconda3/envs/th200/bin/python
import argparse
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
from glob import glob
import random
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
import albumentations as A
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import OrderedDict

import archs

from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter
from albumentations import RandomRotate90,Resize
import time

from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None, help='model name')
    parser.add_argument('--model', default=None, help='model architecture (e.g., UNet, UUEKAN, UUNET)')
    parser.add_argument('--output_dir', default='outputs', help='ouput dir')
            
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

    # Build new output directory structure: outputs/{model_arch}/{dataset}/
    # If --model parameter is specified, use that architecture directly
    if args.model:
        config_path = f'{args.output_dir}/{args.model}/{args.name}/config.yml'
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
    else:
        # First, try to read the model architecture from config.yml
        config_path = f'{args.output_dir}/{args.name}/config.yml'
        
        # If direct path does not exist, try to find it in model architecture subdirectories
        if not os.path.exists(config_path):
            # Search all possible model architecture subdirectories
            possible_archs = ['UNet', 'UUEKAN', 'UUNET']
            for arch in possible_archs:
                test_path = f'{args.output_dir}/{arch}/{args.name}/config.yml'
                if os.path.exists(test_path):
                    config_path = test_path
                    break
    
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Determine the actual output path based on the configuration
    model_arch = config.get('arch', 'Unknown')
    output_path = f"{args.output_dir}/{model_arch}/{args.name}"

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # First, load the checkpoint to determine the correct number of channels
    try:
        ckpt = torch.load(f'{output_path}/model.pth', weights_only=False)
    except Exception as e:
        print(f"⚠️  Failed to load model with weights_only=False: {e}")
        print("🔄 Trying with map_location parameter...")
        ckpt = torch.load(f'{output_path}/model.pth', map_location='cpu', weights_only=False)

    # Infer base_channels from the checkpoint
    base_channels = 64  # default value
    for key in ckpt.keys():
        if 'double_conv.0.weight' in key:
            # The number of output channels of the first convolutional layer is base_channels
            base_channels = ckpt[key].shape[0]
            print(f"🔍 Inferred base_channels = {base_channels} from checkpoint")
            break

    # Create the model based on the architecture, ensuring consistency with checkpoint parameters
    if config['arch'] in ['UUNET', 'UUEKAN']:
        # UUNET and UUEKAN use a heavy architecture (base_channels)
        model = archs.__dict__[config['arch']](
            config['num_classes'], 
            config['input_channels'], 
            config['deep_supervision'], 
            img_size=max(config['input_h'], config['input_w']),
            base_channels=base_channels,  # Use inferred number of channels
            use_uncertainty=True,
            no_kan=config.get('no_kan', False),  # UUEKAN supports no_kan parameter
            **{k: v for k, v in config.items() if k not in ['num_classes', 'input_channels', 'deep_supervision', 'no_kan']}
        )
    else:
        # Other models like UNet
        model = archs.__dict__[config['arch']](
            config['num_classes'], 
            config['input_channels'], 
            config['deep_supervision'],
            img_size=max(config['input_h'], config['input_w']),
            base_channels=base_channels
        )

    model = model.cuda()

    dataset_name = config['dataset']
    img_ext = '.png'

    if dataset_name == 'busi':
        mask_ext = '_mask.png'
    elif dataset_name == 'glas':
        mask_ext = '.png'
    elif dataset_name == 'cvc':
        mask_ext = '.png'
    elif dataset_name == 'ddti':
        mask_ext = '_mask.png'
    elif dataset_name == 'heus':
        mask_ext = '.png'

    # Data loading code
    img_ids = sorted(glob(os.path.join(config['data_dir'], config['dataset'], 'images', '*' + img_ext)))
    # img_ids.sort()
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=config['dataseed'])

    # Checkpoint already loaded above

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
        A.Normalize(),
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
        drop_last=False)

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    hd95_avg_meter = AverageMeter()
    f1_avg_meter = AverageMeter()

    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()
            model = model.cuda()
            # compute output
            output = model(input)

            iou, dice, hd95_, f1 = iou_score(output, target)
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))
            hd95_avg_meter.update(hd95_, input.size(0))
            f1_avg_meter.update(f1, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()
            output[output>=0.5]=1
            output[output<0.5]=0

            os.makedirs(os.path.join(output_path, 'out_val'), exist_ok=True)
            for pred, img_id in zip(output, meta['img_id']):
                pred_np = pred[0].astype(np.uint8)
                pred_np = pred_np * 255
                img = Image.fromarray(pred_np, 'L')
                img.save(os.path.join(output_path, 'out_val/{}.jpg'.format(img_id)))

    
    print(config['name'])
    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)
    print('HD95: %.4f' % hd95_avg_meter.avg)
    print('F1: %.4f' % f1_avg_meter.avg)



if __name__ == '__main__':
    main()
