
import os
import cv2
import pandas as pd
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import Dataset, DataLoader
from preprocess import *


class CustomDataset(Dataset):
    def __init__(self, medical_df, labels, transform=None):
        self.medical_df = medical_df
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.medical_df)

    def __getitem__(self, index):
        data_path = "/home/ljj0512/shared/data/refine224_train_img/"
        img_path = os.path.join(data_path,
                                self.medical_df["img_path"].iloc[index][-14:])
        images = np.load(img_path)
        if self.transform:
            x = [   train_transforms(image=images[0])["image"],
                    train_transforms(image=images[1])["image"],
                    train_transforms(image=images[2])["image"],
                    train_transforms(image=images[3])["image"] ]
            x = torch.stack(x)

        if self.labels is not None:
            tabular = torch.Tensor(self.medical_df.drop(columns=['ID', 'img_path', 'mask_path', '수술연월일']).iloc[index, :].to_numpy())
            label = self.labels.iloc[index].to_numpy()
            return images, tabular, label 
        else:
            tabular = torch.Tensor(self.medical_df.drop(columns=['ID', 'img_path', '수술연월일']).iloc[index, :])
            return images, tabular            


train_transforms = A.Compose([
                            A.HorizontalFlip(),
                            A.VerticalFlip(),
                            A.Rotate(limit=90, border_mode=cv2.BORDER_CONSTANT,p=0.3),
                            # A.Resize(224,512),
                            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])

test_transforms = A.Compose([
                            # A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
                            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])


def get_dataloader(train_data, valid_data, train_label, valid_label, args):
    train_dataset = CustomDataset(train_data,
                                    train_label,
                                    transform=train_transforms)
    valid_dataset = CustomDataset(valid_data,
                                    valid_label, 
                                    transform=test_transforms)
    dl_train = DataLoader(dataset=train_dataset, 
                            batch_size = args.batch_size, 
                            pin_memory = True, 
                            shuffle = True)
    dl_valid = DataLoader(dataset=valid_dataset,
                            batch_size = args.batch_size, 
                            pin_memory = True)
    return dl_train, dl_valid



numeric_cols = ['나이', '암의 장경', 'ER_Allred_score', 'PR_Allred_score', 'KI-67_LI_percent', 'HER2_SISH_ratio']
ignore_cols = ['ID', 'img_path', 'mask_path', '수술연월일', 'N_category']

def get_values(value):
    return value.values.reshape(-1, 1)

def preprocess_dataset(train_df, test_df):
    train_df["img_path"] = "data" + train_df["img_path"].str.replace("./", "/", regex = True)
    test_df["img_path"] = "data" + test_df["img_path"].str.replace("./", "/", regex = True)

    train_df["img_path"] = train_df["img_path"].str.replace("train_img", "refine_train_img")
    test_df["img_path"] = test_df["img_path"].str.replace("test_img", "refine_test_img")

    train_df["img_path"] = train_df["img_path"].str.replace("png", "npy")
    test_df["img_path"] = test_df["img_path"].str.replace("png", "npy")
    
    train_df['암의 장경'] = train_df['암의 장경'].fillna(train_df['암의 장경'].mean())
    train_df = train_df.fillna(0)

    test_df['암의 장경'] = test_df['암의 장경'].fillna(train_df['암의 장경'].mean())
    test_df = test_df.fillna(0)


    for col in train_df.columns:
        if col in ignore_cols:
            continue
        if col in numeric_cols:
            scaler = StandardScaler()
            train_df[col] = scaler.fit_transform(get_values(train_df[col]))
            test_df[col] = scaler.transform(get_values(test_df[col]))
        else:
            le = LabelEncoder()
            train_df[col] = le.fit_transform(get_values(train_df[col]))
            test_df[col] = le.transform(get_values(test_df[col]))

    return train_df, test_df


def load_dataset():
    train_df, test_df = pd.read_csv(os.path.join("data", "train.csv")), pd.read_csv(os.path.join("data", "test.csv"))
    train_df, test_df = preprocess_dataset(train_df, test_df)
    return train_df, test_df
     
     
