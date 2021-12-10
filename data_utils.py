from concurrent.futures.thread import ThreadPoolExecutor
from torch.utils.data import Dataset
from torchvision import transforms
from glob import glob
import pandas as pd
import numpy as np
import joblib
import torch
import os

import random
from PIL import Image
import torchvision


def extract_day(images):
    day = int(images.split('.')[-2][-2:])
    return day


def make_day_array(images):
    day_array = np.array([extract_day(x) for x in images])
    return day_array


def make_combination(length, species, data_frame, direct_name):
    before_file_path = []
    after_file_path = []
    time_delta = []

    for i in range(length):
        
        # 하위 폴더 중에서 랜덤하게 선택을 한다.
        direct = random.randrange(0,len(direct_name))
        # 위에서 결정된 폴더를 선택한다. 
        temp = data_frame[data_frame['version'] == direct_name[direct]]
    
        # 밑은 기존의 코드와 동일합니다.
        sample = temp[temp['species'] == species].sample(2)
        after = sample[sample['day'] == max(sample['day'])].reset_index(drop=True)
        before = sample[sample['day'] == min(sample['day'])].reset_index(drop=True)

        before_file_path.append(before.iloc[0]['file_name'])
        after_file_path.append(after.iloc[0]['file_name'])
        delta = int(after.iloc[0]['day'] - before.iloc[0]['day'])
        time_delta.append(delta)

    combination_df = pd.DataFrame({
        'before_file_path': before_file_path,
        'after_file_path': after_file_path,
        'time_delta': time_delta,
    })

    combination_df['species'] = species

    return combination_df


def data_preprocess(config):
# BC 폴더와 LT 폴더에 있는 하위 폴더를 저장한다.
    bc_direct = glob(config.root_path + '/BC/*')
    bc_direct_name = [x[-5:] for x in bc_direct]
    lt_direct = glob(config.root_path + '/LT/*')
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

    train_dataset = KistDataset(train_set, config, mode='train')
    valid_dataset = KistDataset(valid_set, config, mode='val')
    return train_dataset, valid_dataset


class KistDataset(Dataset):
    def __init__(self, combination_df, config, mode='train'):
        self.combination_df = combination_df

        transform_list = []
        self.mode = mode

        if self.mode == 'train':
            transform_list.append(transforms.autoaugment.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET))
        # transform_list.append(transforms.ToTensor())
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, idx):
        before_image = torch.from_numpy(joblib.load(self.combination_df.iloc[idx]['before_file_path']))
        after_image = torch.from_numpy(joblib.load(self.combination_df.iloc[idx]['after_file_path']))

        before_image = before_image.permute([2,0,1])
        after_image = after_image.permute([2,0,1])
        before_image = self.transform(before_image) / 255.
        after_image = self.transform(after_image) / 255.
        if self.mode == 'test':
            return before_image, after_image
        time_delta = self.combination_df.iloc[idx]['time_delta']
        return before_image, after_image, time_delta

    def __len__(self):
        return len(self.combination_df)
