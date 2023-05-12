# @Time    : 2023/5/11 15:18
# @Author  : emo


import os.path

from sklearn.model_selection import train_test_split
from data.utils import write_dataset_to_txt


class SkinDisease(object):

    def __init__(self):
        self.base_path = 'dataset/skin_disease'
        self.dataset_path = os.path.join(self.base_path, 'image')
        self.dataset_description = os.path.join(self.base_path, 'description/small_dataset.txt')
        # 生成 train.txt, val.txt
        self.train_description = os.path.join(self.base_path, 'generate_description/train.txt')
        self.val_description = os.path.join(self.base_path, 'generate_description/val.txt')
        self.generate_description()
        pass

    def generate_description(self):
        img_paths, labels = [], []
        dict_label = {
            'benign': 0,
            'malignant': 1
        }
        with open(self.dataset_description, 'r') as f:
            lines = f.read().split('\n')
            for line in lines:
                img_paths.append(line.split(',')[0])
                labels.append(dict_label[line.split(',')[1]])

        train_x, val_x, train_y, val_y = train_test_split(img_paths, labels, stratify=labels, test_size=0.2,
                                                          random_state=2023)
        train_set = (train_x, train_y)
        val_set = (val_x, val_y)
        write_dataset_to_txt(train_set, self.train_description)
        write_dataset_to_txt(val_set, self.val_description)
        pass
