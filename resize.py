import os
import joblib
import numpy as np
from glob import glob
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from tqdm.std import tqdm
import shutil


def main():
    datapath = '/root/datasets/open'
    resize_shape = (600, 600)
    save_name = f'{resize_shape[0]}x{resize_shape[1]}'
    for dataset in ['train_dataset', 'test_dataset']:
        print(dataset)
        save_dataset_path = os.path.join(datapath, save_name + '_' + dataset)
        if not os.path.exists(save_dataset_path):
            os.makedirs(save_dataset_path)
        csvdata_path = sorted(glob(os.path.join(datapath, dataset, '*.csv')))
        for csv in csvdata_path:
            shutil.copy2(csv, os.path.join(save_dataset_path, os.path.join(save_dataset_path, os.path.basename(csv))))

        for datatype in ['BC', 'LT']:
            print(dataset, datatype)
            save_datatype_path = os.path.join(save_dataset_path, datatype)
            if not os.path.exists(save_datatype_path):
                os.makedirs(save_datatype_path)
            for folder in tqdm(sorted(glob(os.path.join(datapath, dataset, datatype, '*')))):
                save_folder_path = os.path.join(save_datatype_path, os.path.basename(folder))
                if not os.path.exists(save_folder_path):
                    os.makedirs(save_folder_path)
                with ThreadPoolExecutor() as pool:
                    images = list(pool.map(lambda x: Image.open(x).resize(resize_shape, Image.LANCZOS), sorted(glob(os.path.join(folder, '*')))))
                with ThreadPoolExecutor() as pool:
                    list(pool.map(lambda image, path: joblib.dump(np.array(image), os.path.join(save_folder_path, os.path.splitext(os.path.basename(path))[0] + '.joblib')), images, sorted(glob(os.path.join(folder, '*')))))


if __name__ == '__main__':
    main()