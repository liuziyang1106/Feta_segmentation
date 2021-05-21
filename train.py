import logging
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import datetime
from unet import UNet
import tensorboardX
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from config import args


torch.backends.cudnn.enabled = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():
    best_metric = 100

    model = UNet(n_channels=1, n_classes=8, trilinear=True).to(device)
    if args.load:
        model.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    dataset = BasicDataset(args.train_img_folder, args.train_mask_folder,crop_size=args.crop_size)
    n_val = int(len(dataset) * args.val / 100)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True
                             ,num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False
                           ,num_workers=args.num_workers, pin_memory=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if model.n_classes > 1 else 'max', patience=2)

    # Setting the loss function
    if model.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    saved_metrics, saved_epos = [], []
    writer = tensorboardX.SummaryWriter(args.output_dir)

    for epoch in range(args.epochs):
        train_loss = train(train_loader, model=model, criterion=criterion, aux_criterion = None
                          ,optimizer = optimizer, epoch = epoch, device = device)
        val_loss = valiation(val_loader=valid_loader, model=model, criterion=criterion, aux_criterion=None
                            ,epoch=epoch, device=device)
        scheduler.step(val_loss)


        # ===========  write in tensorboard scaler =========== #
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), epoch)
        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), epoch)

        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Train/loss', train_loss, epoch)
        writer.add_scalar('Val/loss', val_loss, epoch)

        valid_metric = val_loss
        is_best = False
        if valid_metric < best_metric:
            is_best = True
            best_metric = min(valid_metric, best_metric)
                
            saved_metrics.append(valid_metric)
            saved_epos.append(epoch)
            print('=======>   Best at epoch %d, valid MAE %f\n' % (epoch, best_metric))

        save_checkpoint({'epoch': epoch
                        ,'state_dict': model.state_dict()}
                        , is_best
                        , args.output_dir
                        , model_name=args.model
                        )
    return 0

def train(train_loader, model, criterion, aux_criterion, optimizer, epoch, device):

    Epoch_loss = AverageMeter()

    for i, batch in enumerate(train_loader):
        imgs = batch['image']
        true_masks = batch['mask']
        assert imgs.shape[1] == model.n_channels, \
            f'Network has been defined with {model.n_channels} input channels, ' \
            f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
            'the images are loaded correctly.'
        optimizer.zero_grad()

        imgs = imgs.to(device=device, dtype=torch.float32)
        mask_type = torch.float32 if model.n_classes == 1 else torch.long
        true_masks = true_masks.to(device=device, dtype=mask_type)
        masks_pred = model(imgs)

        # # Remove the axis
        true_masks = torch.squeeze(true_masks, dim=1)
        loss = criterion(masks_pred, true_masks)
        
        Epoch_loss.update(loss, imgs.size(0))
        if i % args.print_freq == 0:
            print('Epoch: [{0} / {1}]   [step {2}/{3}] \t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})  \t'.format
                (epoch, args.epochs, i, len(train_loader),loss=Epoch_loss))
    
        loss.backward()
        optimizer.step()

    return Epoch_loss.avg

def valiation(val_loader, model, criterion, aux_criterion, epoch, device):
    Epoch_loss = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            imgs = batch['image']
            true_masks = batch['mask']
            assert imgs.shape[1] == model.n_channels, \
                f'Network has been defined with {model.n_channels} input channels, ' \
                f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                'the images are loaded correctly.'
            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if model.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)
            masks_pred = model(imgs)

            # Remove the axis
            true_masks = torch.squeeze(true_masks, dim=1)
            loss = criterion(masks_pred, true_masks)
            
            Epoch_loss.update(loss, imgs.size(0))
        print('Valid: [steps {0}], Loss {loss.avg:.4f}'.format(len(val_loader), loss=Epoch_loss))
    return Epoch_loss.avg

def save_checkpoint(state, is_best, out_dir, model_name):
    checkpoint_path = out_dir+model_name+'_checkpoint.pth.tar'
    best_model_path = out_dir+model_name+'_best_model.pth.tar'
    torch.save(state, checkpoint_path)
    if is_best:
        torch.save(state, best_model_path)
        print("=======>   This is the best model !!! It has been saved!!!!!!\n\n")

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    print(args)
    res = os.path.join(args.output_dir, 'result.txt')
    if os.path.isdir(args.output_dir): 
        if input("### output_dir exists, rm? ###") == 'y':
            os.system('rm -rf {}'.format(args.output_dir))

    # =========== set train folder =========== #
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    print('=> training from scratch.\n')
    os.system('echo "train {}" >> {}'.format(datetime.datetime.now(), res))
    main()
