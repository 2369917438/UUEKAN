import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import argparse
from collections import OrderedDict
from glob import glob
import random
import numpy as np

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml

import albumentations as A
from albumentations.augmentations import geometric

from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
from albumentations import RandomRotate90, Resize

import archs

import losses
from dataset import Dataset

from metrics import iou_score, indicators

from utils import AverageMeter, str2bool

from tensorboardX import SummaryWriter

import shutil
import os
import subprocess

from pdb import set_trace as st


ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')


def list_type(s):
    str_list = s.split(',')
    int_list = [int(a) for a in str_list]
    return int_list


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--resume', default=None, type=str,
                        help='path to checkpoint to resume from (default: None)')
    parser.add_argument('--epochs', default=400, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    parser.add_argument('--dataseed', default=2981, type=int,
                        help='')
    
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UUEKAN')
    
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=256, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=256, type=int,
                        help='image height')
    parser.add_argument('--input_list', type=list_type, default=[128, 160, 256])

    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')
    
    # dataset
    parser.add_argument('--dataset', default='busi', help='dataset name')      
    parser.add_argument('--data_dir', default='inputs', help='dataset dir')

    parser.add_argument('--output_dir', default='outputs', help='ouput dir')


    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')

    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    parser.add_argument('--kan_lr', default=1e-2, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--kan_weight_decay', default=1e-4, type=float,
                        help='weight decay')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )
    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument('--no_kan', action='store_true')



    config = parser.parse_args()

    return config


def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'f1': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)

            iou, dice, _, f1 = iou_score(outputs[-1], target)
            iou_, dice_, hd_, hd95_, recall_, specificity_, precision_ = indicators(outputs[-1], target)
            
        else:
            output = model(input)
            loss = criterion(output, target)
            if isinstance(loss, tuple):
                loss = loss[0]
            iou, dice, _, f1 = iou_score(output, target)
            iou_, dice_, hd_, hd95_, recall_, specificity_, precision_ = indicators(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['f1'].update(f1, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('f1', avg_meters['f1'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('f1', avg_meters['f1'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                   'dice': AverageMeter(),
                   'f1': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou, dice, _, f1 = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                if isinstance(loss, tuple):
                    loss = loss[0]
                iou, dice, _, f1 = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['f1'].update(f1, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
                ('f1', avg_meters['f1'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()


    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg),
                        ('f1', avg_meters['f1'].avg)])

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
    config = vars(parse_args())
    
    # UUEKAN and UUNET models default to CombinedLoss (if the user does not manually specify another loss function)
    if config.get('arch') in ['UUEKAN', 'UUNET'] and config.get('loss') == 'BCEDiceLoss':
        config['loss'] = 'CombinedLoss'
        print(f"💡 Detected {config.get('arch')} model (with uncertainty module), automatically switching to CombinedLoss")

    exp_name = config.get('name')
    output_dir = config.get('output_dir')
    model_arch = config.get('arch')
    
    # Build a new output directory structure: outputs/{model_arch}/{dataset}/
    # This way, different models training on the same dataset will not interfere with each other
    output_path = f'{output_dir}/{model_arch}/{exp_name}'
    
    # Clean up old TensorBoard log files
    tensorboard_dir = output_path
    if os.path.exists(tensorboard_dir):
        # Delete all events files
        for file in glob(os.path.join(tensorboard_dir, 'events.out.tfevents.*')):
            try:
                os.remove(file)
                print(f"Deleted old TensorBoard log file: {file}")
            except:
                pass

    my_writer = SummaryWriter(tensorboard_dir)

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    
    os.makedirs(output_path, exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open(f'{output_path}/config.yml', 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True

    # If a checkpoint exists, first infer base_channels from it
    base_channels = None
    if config['resume'] and os.path.isfile(config['resume']):
        try:
            print(f"🔍 Inferring model parameters from checkpoint...")
            checkpoint = torch.load(config['resume'], map_location='cpu', weights_only=False)
            model_state = checkpoint['model_state_dict']
            
            # Infer base_channels from the checkpoint
            for key in model_state.keys():
                if 'double_conv.0.weight' in key:
                    base_channels = model_state[key].shape[0]
                    print(f"🔍 Inferred base_channels = {base_channels} from checkpoint")
                    break
        except Exception as e:
            print(f"⚠️  Could not infer parameters from checkpoint: {e}")
    
    # Set default base_channels
    if base_channels is None:
        if config['arch'] == 'UUNET':
            base_channels = 64
        else:
            base_channels = 64
        print(f"🔧 Using default base_channels = {base_channels}")

    # create model
    if config['arch'] in ['UUNET', 'UUEKAN']:
        # UUNET and UUEKAN use the base_channels parameter to control the number of channels (heavy architecture)
        model = archs.__dict__[config['arch']](
            config['num_classes'], 
            config['input_channels'], 
            config['deep_supervision'], 
            img_size=max(config['input_h'], config['input_w']),
            base_channels=base_channels,  # Use inferred or default number of channels
            use_uncertainty=True,  # Enable uncertainty module
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


    param_groups = []

    kan_fc_params = []
    other_params = []

    for name, param in model.named_parameters():
        # print(name, "=>", param.shape)
        if 'layer' in name.lower() and 'fc' in name.lower(): # higher lr for kan layers
            # kan_fc_params.append(name)
            param_groups.append({'params': param, 'lr': config['kan_lr'], 'weight_decay': config['kan_weight_decay']}) 
        else:
            # other_params.append(name)
            param_groups.append({'params': param, 'lr': config['lr'], 'weight_decay': config['weight_decay']})  
    

    
    # st()
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(param_groups)


    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'], nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'], verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # Disabled: no longer automatically copy code files to the output directory
    # shutil.copy2('train.py', f'{output_dir}/{exp_name}/')
    # # Copy the entire archs folder
    # if os.path.exists(f'{output_dir}/{exp_name}/archs'):
    #     shutil.rmtree(f'{output_dir}/{exp_name}/archs')
    # shutil.copytree('archs', f'{output_dir}/{exp_name}/archs')

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
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=config['dataseed'])

    train_transform = Compose([
        RandomRotate90(),
        A.HorizontalFlip(),
        Resize(config['input_h'], config['input_w']),
        A.Normalize(),
    ])

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        A.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(config['data_dir'], config['dataset'], 'images'),
        mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(config['data_dir'] ,config['dataset'], 'images'),
        mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('f1', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
        ('val_f1', []),
    ])


    best_iou = 0
    best_dice= 0
    trigger = 0
    start_epoch = 0
    
    # Resume from checkpoint
    if config['resume']:
        if os.path.isfile(config['resume']):
            print(f"=> Loading checkpoint '{config['resume']}'")
            # The checkpoint has been loaded above, use it directly
            if 'checkpoint' not in locals():
                # If not loaded above, load it now
                try:
                    checkpoint = torch.load(config['resume'], weights_only=False)
                except Exception as e:
                    print(f"⚠️  Failed to load checkpoint with weights_only=False: {e}")
                    print("🔄 Trying with map_location parameter...")
                    checkpoint = torch.load(config['resume'], map_location='cpu', weights_only=False)
            
            # Restore model state
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Restore optimizer state
            if checkpoint['optimizer_state_dict'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("=> Restored optimizer state")
            else:
                print("=> Optimizer state is empty, using new optimizer state")
            
            # Restore learning rate scheduler state
            if scheduler and checkpoint['scheduler_state_dict'] is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("=> Restored learning rate scheduler state")
            else:
                print("=> Scheduler state is empty, using new scheduler state")
            
            # Restore training state
            start_epoch = checkpoint['epoch']
            best_iou = checkpoint['best_iou']
            best_dice = checkpoint['best_dice']
            
            # Restore log
            if 'log' in checkpoint:
                log = checkpoint['log']
            
            print(f"=> Checkpoint loaded successfully (resuming training from epoch {start_epoch + 1})")
            print(f"=> Best IoU: {best_iou:.4f}, Best Dice: {best_dice:.4f}")
        else:
            print(f"=> Checkpoint file not found: '{config['resume']}'")
    
    for epoch in range(start_epoch, config['epochs']):
        print('Epoch [%d/%d]' % (epoch + 1, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - f1 %.4f - val_loss %.4f - val_iou %.4f - val_dice %.4f - val_f1 %.4f'
              % (train_log['loss'], train_log['iou'], train_log['f1'], val_log['loss'], val_log['iou'], val_log['dice'], val_log['f1']))

        log['epoch'].append(epoch + 1)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['f1'].append(train_log['f1'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])
        log['val_f1'].append(val_log['f1'])

        # Save the complete log (including history)
        pd.DataFrame(log).to_csv(f'{output_path}/log.csv', index=False)

        my_writer.add_scalar('train/loss', train_log['loss'], global_step=epoch + 1)
        my_writer.add_scalar('train/iou', train_log['iou'], global_step=epoch + 1)
        my_writer.add_scalar('train/f1', train_log['f1'], global_step=epoch + 1)
        my_writer.add_scalar('val/loss', val_log['loss'], global_step=epoch + 1)
        my_writer.add_scalar('val/iou', val_log['iou'], global_step=epoch + 1)
        my_writer.add_scalar('val/dice', val_log['dice'], global_step=epoch + 1)
        my_writer.add_scalar('val/f1', val_log['f1'], global_step=epoch + 1)

        my_writer.add_scalar('val/best_iou_value', best_iou, global_step=epoch + 1)
        my_writer.add_scalar('val/best_dice_value', best_dice, global_step=epoch + 1)

        trigger += 1

        if val_log['iou'] > best_iou:
            # Save complete checkpoint information
            checkpoint = {
                'epoch': epoch + 1,  # Save the next epoch to be trained
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_iou': val_log['iou'],
                'best_dice': val_log['dice'],
                'config': config,
                'log': log
            }
            torch.save(checkpoint, f'{output_path}/checkpoint.pth')
            # Also save the separate model file (for backward compatibility)
            torch.save(model.state_dict(), f'{output_path}/model.pth')
            
            best_iou = val_log['iou']
            best_dice = val_log['dice']
            print("=> saved best model")
            print('IoU: %.4f' % best_iou)
            print('Dice: %.4f' % best_dice)
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()
    
if __name__ == '__main__':
    main()
