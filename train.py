import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch import optim
from torch import nn

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision import transforms

import random
from glob import glob
import pandas as pd
import numpy as np
from PIL import Image
from utils import *
from data_utils import *
from model import *
from args import get_args
import os


def test_iter(model, config, epoch=None, device=torch.device('cpu')):
    prefix = f'{config.size}x{config.size}_' if config.size is not None else 'joblib_'
    test_set = pd.read_csv(os.path.join(config.datapath, f'{prefix}test_dataset/test_data.csv'))
    test_set['l_root'] = test_set['before_file_path'].map(lambda x: os.path.join(config.datapath, f'{prefix}test_dataset/') + x.split('_')[1] + '/' + x.split('_')[2])
    test_set['r_root'] = test_set['after_file_path'].map(lambda x: os.path.join(config.datapath, f'{prefix}test_dataset/') + x.split('_')[1] + '/' + x.split('_')[2])
    test_set['l_path'] = test_set['l_root'] + '/' + test_set['before_file_path'] + '.joblib'
    test_set['r_path'] = test_set['r_root'] + '/' + test_set['after_file_path'] + '.joblib'
    test_set['before_file_path'] = test_set['l_path']
    test_set['after_file_path'] = test_set['r_path']
    test_dataset = KistDataset(test_set, config, 'test')
    test_data_loader = DataLoader(test_dataset,
                                batch_size=config.batch * 2, num_workers=10)
    test_value = []
    with torch.no_grad():
        for test_before, test_after in tqdm(test_data_loader):
            test_before = test_before.to(device)
            test_after = test_after.to(device)
            logit = model(test_before, test_after)
            value = logit.squeeze(1).detach().cpu().float()
            
            test_value.extend(value)

    submission = pd.read_csv(os.path.join(config.datapath, 'sample_submission.csv'))
    _sub = torch.FloatTensor(test_value)

    __sub = _sub.numpy()
    __sub[np.where(__sub<1)] = 1

    submission['time_delta'] = __sub
    res_name = f'{config.name}'
    res_name += f'_{epoch}' if epoch is not None else ''
    submission.to_csv(f'{res_name}.csv', index=False)


def iteration(config, epoch, model, pbar, criterion, mode='train', optimizer: torch.optim.Optimizer=None, lr_schedule=None, decay=None, device=torch.device('cpu')):
    losses = []
    iters = len(pbar)

    for step, (before_image, after_image, time_delta) in enumerate(pbar):
        if mode == 'train':
            optimizer.zero_grad()
        before_image = before_image.to(device)
        after_image = after_image.to(device)
        time_delta = time_delta.to(device)

        if not config.patch:
            logit = model(before_image, after_image)
        else:
            logit = 0
            batch = before_image.shape[0]
            shape = [*before_image.shape[2:]]
            logit = model(before_image.reshape([-1] + shape), after_image.reshape([-1] + shape))
            logit = logit.reshape([batch, -1] + [1]).mean(1)
        loss = torch.sqrt(criterion(logit.squeeze(1).float(), time_delta.float()))

        if mode == 'train':
            loss.backward()
            optimizer.step()
            lr_schedule.step(epoch + step / iters)
        losses.append(loss.detach().cpu())
        pbar.set_postfix({'epoch': epoch, 'mode': mode, 'loss': torch.stack(losses).mean().numpy()})
    return torch.stack(losses).mean()


def RMSE(a, b):
    return torch.sqrt(torch.nn.MSELoss()(a, b))


def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    # 학습 데이터가 있는 폴더 위치
    size = config.size
    # if config.patch:
    #     size = None
    if size is not None:
        size_ = f'{size}x{size}_'
        if config.patch:
            size_ += 'patch_'
    else:
        size_ = 'joblib_'
    if config.mask:
        train_size_ = size_ + 'mask_'
    else:
        train_size_ = size_
    
    name = f'{config.name}_{config.batch}_{config.lr}'
    if config.mask:
        name += '_mask'
    if config.norm:
        name += '_norm'
    if config.patch:
        name += '_patch'
    config.name = name
    seed_everything(2048)

    config.root_path = os.path.join(config.datapath, f'{train_size_}train_dataset')
    valid_batch_size = config.batch * 2
    max_patience = 10
    best_score = np.inf
    criterion = RMSE

    train_dataset, valid_dataset = data_preprocess(config)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = CompareNet(config).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=0, last_epoch=- 1, verbose=False)
    decay = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=1/2**(0.5), patience=2, threshold=config.lr / 1000, verbose=True)


    train_data_loader = DataLoader(train_dataset,
                                batch_size=config.batch,
                                shuffle=True, num_workers=10 * torch.cuda.device_count())

    valid_data_loader = DataLoader(valid_dataset,
                                batch_size=valid_batch_size, num_workers=10 * torch.cuda.device_count())


    initial_epoch = 0
    patience = 0
    if config.resume:
        resume = torch.load(f'{name}.pt')
        model.load_state_dict(resume['model'])
        try:
            initial_epoch = resume['epoch']
            optimizer.load_state_dict(resume['opt'])
            patience = resume['patience']
            decay.load_state_dict(resume['decay'])
            best_score = resume['best_score']
        except:
            print('Warning: have some problems to resume')

    for epoch in range(initial_epoch, config.epoch):
        if patience == max_patience:
            print('Early stopping')
            break
        print('---------------', epoch, '---------------')

        model.train()
        with tqdm(train_data_loader) as pbar:
            train_losses = iteration(config, epoch, model, pbar, criterion, optimizer=optimizer, lr_schedule=lr_schedule, decay=decay, device=device)

        model.eval()
        with torch.no_grad():
            with tqdm(valid_data_loader) as pbar:
                val_losses = iteration(config, epoch, model, pbar, criterion, mode='val', device=device)
        decay.step(val_losses)
        
        if val_losses < best_score:
            best_score = val_losses
            checkpoiont = {
                'model': model.state_dict(),
                'epoch': epoch,
                'opt': optimizer.state_dict(),
                'patience': patience,
                'decay': decay.state_dict(),
                'best_score': best_score
            }
            torch.save(checkpoiont, f'{name}.pt')
            patience = 0
            if epoch >= 3:
                with torch.no_grad():
                    test_iter(model, config, epoch)
        else:
            patience += 1
    
    if config.epoch == 0:
        model.load_state_dict(torch.load(f'{name}.pt')['model'])
        test_iter(model, config, device=device)


if __name__ == '__main__':
    main(get_args())

