import os
import joblib
import numpy as np
from glob import glob
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import cv2 as cv

from tqdm.std import tqdm
import shutil


def main():
    datapath = '/root/datasets/open'
    resize_shape = (600, 600)
    save_name = f'{resize_shape[0]}x{resize_shape[1]}'
    for dataset in ['mask_train_dataset', 'train_dataset', 'test_dataset']:
        img = np.zeros([0] + [*resize_shape] + [3], dtype=np.uint8)
        mean = None
        std = None
        print(dataset)
        save_dataset_path = os.path.join(datapath, save_name + '_' + dataset)
        if not os.path.exists(save_dataset_path):
            os.makedirs(save_dataset_path)
        csvdata_path = sorted(glob(os.path.join(datapath, dataset, '*.csv')))
        for csv in csvdata_path:
            shutil.copy2(csv, os.path.join(save_dataset_path, os.path.join(save_dataset_path, os.path.basename(csv))))

        for datatype in ['LT', 'BC']:
            print(dataset, datatype)
            save_datatype_path = os.path.join(save_dataset_path, datatype)
            if not os.path.exists(save_datatype_path):
                os.makedirs(save_datatype_path)
            for folder in tqdm(sorted(glob(os.path.join(datapath, dataset, datatype, '*')))):
                save_folder_path = os.path.join(save_datatype_path, os.path.basename(folder))
                if not os.path.exists(save_folder_path):
                    os.makedirs(save_folder_path)
                with ThreadPoolExecutor() as pool:
                    images = list(pool.map(lambda x: cv.resize(np.array(Image.open(x))[:,400:-400], dsize=resize_shape, interpolation=cv.INTER_LANCZOS4), sorted(glob(os.path.join(folder, '*')))))
                for image in images:
                    if image.shape[-1] == 4:
                        if (image[..., 3] != 255).sum() != 0:
                            raise ValueError('channel error')
                        image = image[...,:3]
                    img = np.concatenate([img, np.array(image)[np.newaxis,...]])
                with ThreadPoolExecutor() as pool:
                    list(pool.map(lambda image, path: joblib.dump(np.array(image), os.path.join(save_folder_path, os.path.splitext(os.path.basename(path))[0] + '.joblib')), img, sorted(glob(os.path.join(folder, '*')))))
        mean = img.astype(np.float64).mean(0, keepdims=True)
        std = img.std(0, keepdims=True)
        extpath = os.path.join(save_dataset_path, 'extra.joblib')
        joblib.dump({'mean': mean, 'std': std}, extpath)
        save_normdataset_path = os.path.join(datapath, save_name + '_' + dataset + '_norm')
        for datatype in ['BC', 'LT']:
            print(dataset, datatype)
            save_datatype_path = os.path.join(save_dataset_path, datatype)
            save_normdatatype_path = os.path.join(save_normdataset_path, datatype)
            if not os.path.exists(save_datatype_path):
                os.makedirs(save_datatype_path)
            
            for folder in tqdm(sorted(glob(os.path.join(save_datatype_path, '*')))):
                save_folder_path = os.path.join(save_datatype_path, os.path.basename(folder))
                save_normfolder_path = os.path.join(save_normdatatype_path, os.path.basename(folder))
                if not os.path.exists(save_normfolder_path):
                    os.makedirs(save_normfolder_path)

                with ThreadPoolExecutor() as pool:
                    images = list(pool.map(lambda x: joblib.load(x), sorted(glob(os.path.join(folder, '*.joblib')))))
                
                if image.shape[-1] == 4:
                    if (image[..., 3] != 255).sum() != 0:
                        raise ValueError('channel error')
                    image = image[...,:3]
                images = np.stack(images)
                images = (images - mean) / std

                with ThreadPoolExecutor() as pool:
                    list(pool.map(lambda image, path: joblib.dump(image, os.path.join(save_normfolder_path, os.path.splitext(os.path.basename(path))[0] + '_norm.joblib')), images, sorted(glob(os.path.join(folder, '*')))))


if __name__ == '__main__':
    main()