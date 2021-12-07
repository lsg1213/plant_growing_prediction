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


def extract_day(file_name):
    day = int(file_name.split('.')[-2][-2:])
    return day


def make_day_array(image_pathes):
    day_array = np.array([extract_day(file_name) for file_name in image_pathes])
    return day_array


def make_image_path_array(root_path=None):
    if root_path is None:
        bc_directories = glob('./BC/*')
        lt_directories = glob('./LT/*')

    else:
        bc_directories = glob(root_path + 'BC/*')
        lt_directories = glob(root_path + 'LT/*')

    bc_image_path = []
    for bc_path in bc_directories:
        images = glob(bc_path + '/*.png')
        bc_image_path.extend(images)

    lt_image_path = []
    for lt_path in lt_directories:
        images = glob(lt_path + '/*.png')
        lt_image_path.extend(images)

    return bc_image_path, lt_image_path


def make_dataframe(root_path=None):
    bc_image_path, lt_image_path = make_image_path_array(root_path)
    bc_day_array = make_day_array(bc_image_path)
    lt_day_array = make_day_array(lt_image_path)

    bc_df = pd.DataFrame({'file_name': bc_image_path,
                          'day': bc_day_array})
    bc_df['species'] = 'bc'

    lt_df = pd.DataFrame({'file_name': lt_image_path,
                          'day': lt_day_array})
    lt_df['species'] = 'lt'

    total_data_frame = pd.concat([bc_df, lt_df]).reset_index(drop=True)

    return total_data_frame


def make_combination(length, species, data_frame):
    before_file_path = []
    after_file_path = []
    time_delta = []

    for i in range(length):
        sample = data_frame[data_frame['species'] == species].sample(2)
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


def slice_img(img):
    # img (w, h, c)
    # return (5, w, h, c)
    w, h, _ = img.shape
    wsize = w * 1 // 5
    hsize = h * 1 // 5

    return torch.stack(
                [img[hsize:-hsize, wsize:-wsize],
                img[:3 * hsize, :3 * wsize],
                img[-3 * hsize:, :3 * wsize],
                img[-3 * hsize:, -3 * wsize:],
                img[:3 * hsize, -3 * wsize:]], 0)


class KistDataset(Dataset):
    def __init__(self, config, combination_df, size, is_test=None):
        self.combination_df = combination_df
        self.config = config
        try:
            getattr(self.config, 'self')
            self.condition = False
        except:
            self.condition = True
        if self.config.norm:
            ext = joblib.load(os.path.join(self.config.root_path, 'extra.joblib'))
            self.mean = torch.from_numpy(ext['mean'][0]).type(torch.float32).permute([2,0,1])
            self.std = torch.from_numpy(ext['std'][0]).type(torch.float32).permute([2,0,1])
            if self.config.patch:
                self.mean = self.mean.unsqueeze(0)
                self.std = self.std.unsqueeze(0)
            if not self.condition:
                self.mean = self.mean.unsqueeze(-1)
                self.std = self.std.unsqueeze(-1)
            
        transform_list = [
            transforms.autoaugment.AutoAugment(transforms.autoaugment.AutoAugmentPolicy.IMAGENET)
            # transforms.ToTensor(),
            # transforms.Resize(size)
        ]
        if self.config.patch:
            transform_list.insert(0, transforms.Resize(self.config.size, transforms.InterpolationMode.BICUBIC))
        self.transform = transforms.Compose(transform_list)
        self.is_test = is_test

    def __getitem__(self, idx):
        before_image = joblib.load(self.combination_df.iloc[idx]['before_file_path'])
        after_image = joblib.load(self.combination_df.iloc[idx]['after_file_path'])
        raw_before_image = torch.from_numpy(before_image)
        raw_after_image = torch.from_numpy(after_image)

        if self.config.patch:
            before_image = slice_img(raw_before_image)
            after_image = slice_img(raw_after_image)

            before_image = torch.stack(list(map(self.transform, before_image.permute((0, 3, 1, 2)))), 0)
            after_image = torch.stack(list(map(self.transform, after_image.permute((0, 3, 1, 2)))), 0)
        else:
            before_image = self.transform(raw_before_image.permute([2,0,1]))
            after_image = self.transform(raw_after_image.permute([2,0,1]))
            if not self.condition:
                before_image = torch.stack([before_image, self.transform(raw_before_image.permute([2,0,1]))], -1)
                after_image = torch.stack([after_image, self.transform(raw_after_image.permute([2,0,1]))], -1)

        if self.config.norm:
            before_image = (before_image - self.mean) / self.std
            after_image = (after_image - self.mean) / self.std
        else:
            before_image /= 255.
            after_image /= 255.

        
        if self.is_test:
            return before_image, after_image
        time_delta = self.combination_df.iloc[idx]['time_delta']
        return before_image, after_image, time_delta

    def __len__(self):
        return len(self.combination_df)