import torch
from torch.nn.functional import threshold
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch import optim
from torch import nn
from torchvision.transforms import ToTensor
from torchsummary import summary
from torch.nn.functional import cosine_similarity as coss
import os

import random
from glob import glob
import pandas as pd
import numpy as np

from model import *
from data_utils import *
import argparse


args = argparse.ArgumentParser()
args.add_argument('--name', type=str, required=True)
args.add_argument('--batch', type=int, default=16)
args.add_argument('--lr', type=float, default=1e-4)
args.add_argument('--epoch', type=int, default=200)
args.add_argument('--size', type=int, default=600)
args.add_argument('--gpus', type=str, default='-1')
args.add_argument('--norm', type=bool, default=True)
args.add_argument('--mask', action='store_true')
args.add_argument('--patch', action='store_true')
args.add_argument('--self', action='store_true')


def seed_everything(seed): # seed 고정
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    datapath = '/root/datasets/open/old'
    size = config.size
    config.self = True
    if size is not None:
        size_ = f'{size}x{size}_'
    else:
        size_ = ''
    if config.mask:
        train_size_ = size_ + 'mask_'
    else:
        train_size_ = size_
    root_path = os.path.join(datapath, f'{train_size_}train_dataset')
    config.root_path = root_path

    name = f'{config.name}_{config.batch}_{config.lr}'
    if config.mask:
        name += '_mask'
    if config.norm:
        name += '_norm'
    if config.patch:
        name += '_patch'
    seed_everything(2048)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    lr = config.lr
    epochs = config.epoch
    batch_size = config.batch
    valid_batch_size = batch_size * 2
    max_patience = 10
    best_score = np.inf
    criterion = torch.nn.MSELoss()


    # BC 폴더와 LT 폴더에 있는 하위 폴더를 저장한다.
    bc_direct = glob(root_path + '/BC/*')
    bc_direct_name = [x[-5:] for x in bc_direct]
    lt_direct = glob(root_path + '/LT/*')
    lt_direct_name = [x[-5:] for x in lt_direct]

    # 하위 폴더에 있는 이미지들을 하위 폴더 이름과 매칭시켜서 저장한다.
    bc_images = {key : glob(name + '/*.joblib') for key,name in zip(bc_direct_name, bc_direct)}
    lt_images = {key : glob(name + '/*.joblib') for key,name in zip(lt_direct_name, lt_direct)}

    # 하위 폴더에 있는 이미지들에서 날짜 정보만 따로 저장한다.
    bc_dayes = {key : make_day_array(bc_images[key]) for key in bc_direct_name}
    lt_dayes = {key : make_day_array(lt_images[key]) for key in lt_direct_name}

    bc_dfs = []
    for i in bc_direct_name:
        bc_df = pd.DataFrame({
            'file_name':bc_images[i],
            'day':bc_dayes[i],
            'species':'bc',
            'version':i
        })
        bc_dfs.append(bc_df)
        
    lt_dfs = []
    for i in lt_direct_name:
        lt_df = pd.DataFrame({
            'file_name':lt_images[i],
            'day':lt_dayes[i],
            'species':'lt',
            'version':i
        })
        lt_dfs.append(lt_df)

    bc_dataframe = pd.concat(bc_dfs).reset_index(drop=True)
    lt_dataframe = pd.concat(lt_dfs).reset_index(drop=True)
    total_dataframe = pd.concat([bc_dataframe, lt_dataframe]).reset_index(drop=True)

    bc_combination = make_combination(5000, 'bc', total_dataframe, bc_direct_name)
    lt_combination = make_combination(5000, 'lt', total_dataframe, lt_direct_name)

    bc_train = bc_combination.iloc[:4500]
    bc_valid = bc_combination.iloc[4500:]

    lt_train = lt_combination.iloc[:4500]
    lt_valid = lt_combination.iloc[4500:]

    train_set = pd.concat([bc_train, lt_train])
    valid_set = pd.concat([bc_valid, lt_valid])

    train_dataset = KistDataset(config, train_set, size=size)
    valid_dataset = KistDataset(config, valid_set, size=size)

    
    test_set = pd.read_csv(os.path.join(datapath, f'{size_}test_dataset/test_data.csv'))
    test_set['l_root'] = test_set['before_file_path'].map(lambda x: os.path.join(datapath, f'{size_}test_dataset/') + x.split('_')[1] + '/' + x.split('_')[2])
    test_set['r_root'] = test_set['after_file_path'].map(lambda x: os.path.join(datapath, f'{size_}test_dataset/') + x.split('_')[1] + '/' + x.split('_')[2])
    test_set['l_path'] = test_set['l_root'] + '/' + test_set['before_file_path'] + '.joblib'
    test_set['r_path'] = test_set['r_root'] + '/' + test_set['after_file_path'] + '.joblib'
    test_set['before_file_path'] = test_set['l_path']
    test_set['after_file_path'] = test_set['r_path']
    test_dataset = KistDataset(config, test_set, size=size, is_test=True)
    test_data_loader = DataLoader(test_dataset,
                                batch_size=batch_size)

    model = CompareNet(config).to(device)
    summary(model, input_size=[(3, size, size), (3, size, size)])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=0, last_epoch=- 1, verbose=False)
    decay = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=1/2**(0.5), patience=2, threshold=lr / 1000, verbose=True)

    train_data_loader = DataLoader(train_dataset,
                                batch_size=batch_size,
                                shuffle=True, num_workers=10)

    valid_data_loader = DataLoader(valid_dataset,
                                batch_size=valid_batch_size)
    
    patience = 0
    for epoch in range(epochs):
        if patience == max_patience:
            print('Early stopping')
            break
        print('---------------', epoch, '---------------')
        train_losses = []
        model.train()
        iters = len(train_data_loader)
        with tqdm(train_data_loader, total=iters) as pbar:
            for i, (before_image, after_image, time_delta) in enumerate(pbar):
                before_image = before_image.to(device)
                after_image = after_image.to(device)
                time_delta = time_delta.to(device)
                optimizer.zero_grad()
                if not config.patch:
                    before1, before2, after1, after2 = model(before_image[..., 0], after_image[..., 0])
                    before3, before4, after3, after4 = model(before_image[..., 1], after_image[..., 1])
                    postives = coss(before1, before2) + coss(before1, before3) + coss(before1, before4) + coss(before2, before3) + coss(before2, before4) + coss(before3, before4) + coss(after1, after2) + coss(after1, after3) + coss(after1, after4) + coss(after2, after3) + coss(after2, after4) + coss(after3, after4)
                    negatives = coss(before1, after1) + coss(before1, after2) + coss(before1, after3) + coss(before1, after4) + coss(before2, after1) + coss(before2, after2) + coss(before2, after3) + coss(before2, after4) + coss(before3, after1) + coss(before3, after2) + coss(before3, after3) + coss(before3, after4) + coss(before4, after1) + coss(before4, after2) + coss(before4, after3) + coss(before4, after4)
                    train_loss = - torch.log(postives / negatives + 1e-8).mean()
                else:
                    raise ValueError('--patch unsuported')
                    logit = 0
                    for i in range(before_image.shape[1]):
                        logit += model(before_image[:,i], after_image[:,i])
                    logit /= before_image.shape[1]

                train_loss.backward()
                optimizer.step()
                lr_schedule.step(epoch + i / iters)
                train_losses.append(train_loss.detach().cpu())
                pbar.set_postfix({'epoch': epoch, 'mode': 'train', 'loss': torch.stack(train_losses).mean().numpy()})

        valid_losses = []
        model.eval()
        with torch.no_grad():
            with tqdm(valid_data_loader) as pbar:
                for valid_before, valid_after, time_delta in pbar:
                    valid_before = valid_before.to(device)
                    valid_after = valid_after.to(device)
                    valid_time_delta = time_delta.to(device)

                    if not config.patch:
                        before1, before2, after1, after2 = model(valid_before[..., 0], valid_after[..., 0])
                        before3, before4, after3, after4 = model(valid_before[..., 1], valid_after[..., 1])
                        postives = coss(before1, before2) + coss(before1, before3) + coss(before1, before4) + coss(before2, before3) + coss(before2, before4) + coss(before3, before4) + coss(after1, after2) + coss(after1, after3) + coss(after1, after4) + coss(after2, after3) + coss(after2, after4) + coss(after3, after4)
                        negatives = coss(before1, after1) + coss(before1, after2) + coss(before1, after3) + coss(before1, after4) + coss(before2, after1) + coss(before2, after2) + coss(before2, after3) + coss(before2, after4) + coss(before3, after1) + coss(before3, after2) + coss(before3, after3) + coss(before3, after4) + coss(before4, after1) + coss(before4, after2) + coss(before4, after3) + coss(before4, after4)
                        val_loss = - torch.log(postives / negatives + 1e-8).mean()
                    else:
                        logit = 0
                        for i in range(valid_before.shape[1]):
                            logit += model(valid_before[:,i], valid_after[:,i])
                        logit /= valid_before.shape[1]

                    valid_losses.append(valid_loss.detach().cpu())
                    pbar.set_postfix({'epoch': epoch, 'mode': 'val', 'loss': torch.stack(valid_losses).mean().numpy()})

        val_loss = sum(valid_losses)/len(valid_losses)
        decay.step(val_loss)
        checkpoiont = {
            'model': model.state_dict(),
        }
        if val_loss < best_score:
            print(f'BEST VALIDATION_LOSS RMSE : {best_score}')
            best_score = val_loss
            torch.save(checkpoiont, f'{name}.pt')
            patience = 0
        else:
            print(f'VALIDATION_LOSS RMSE : {val_loss}', f'\nBEST : {best_score}')
            patience += 1
    try:
        model.load_state_dict(torch.load(f'{name}.pt')['model'])
    except:
        print('load weight fail')
    # tr = torch.stack([torch.from_numpy(np.stack([joblib.load(train_set.iloc[i]['before_file_path']), joblib.load(train_set.iloc[i]['after_file_path'])], -1)) for i in range(len(train_set))], 0).type(torch.float16)
    # va = torch.stack([torch.from_numpy(np.stack([joblib.load(valid_set.iloc[i]['before_file_path']), joblib.load(valid_set.iloc[i]['after_file_path'])], -1)) for i in range(len(valid_set))], 0).type(torch.float16)
    # te = torch.stack([torch.from_numpy(np.stack([joblib.load(test_set.iloc[i]['before_file_path']), joblib.load(test_set.iloc[i]['after_file_path'])], -1)) for i in range(len(test_set))], 0).type(torch.float16)
    # tr_mean = (tr / 255).mean(0, keepdims=True)
    # tr_std = tr.std(0, keepdims=True)
    # va_mean = (va / 255).mean(0, keepdims=True)
    # va_std = va.std(0, keepdims=True)
    # te_mean = (te / 255).mean(0, keepdims=True)
    # te_std = te.std(0, keepdims=True)
    # tr_te = torch.cat([tr, te], 0)
    # tr_te_mean = (tr_te / 255).mean(0, keepdims=True)
    # tr_te_std = tr_te.std(0, keepdims=True)
    # all = torch.cat([tr_te, va], 0)
    # all_mean = (all / 255).mean(0, keepdims=True)
    # all_std = all.std(0, keepdims=True)


if __name__ == '__main__':
    main(args.parse_args())