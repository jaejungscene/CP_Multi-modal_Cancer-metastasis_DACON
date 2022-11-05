import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"

import wandb
from datetime import datetime
temp = datetime.now().strftime("%d-%m-%y,%H:%M:%S")
wandb.init(project="Cencer-Metastasis", entity="jaejungscene", name=temp)
del temp

CFG = {
    'IMG_SIZE':600,
    'EPOCHS':20,
    'LEARNING_RATE':1e-4,
    'BATCH_SIZE':16,
    'SEED':41,
    "MODEL":"efficientnet_b7"

}
print(CFG)


import random
import pandas as pd
import numpy as np
import os
import cv2
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torchvision.models as models
from timm import create_model

from sklearn import metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings(action='ignore') 

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정




train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')




train_df['암의 장경'] = train_df['암의 장경'].fillna(train_df['암의 장경'].mean())
train_df = train_df.fillna(0)

test_df['암의 장경'] = test_df['암의 장경'].fillna(train_df['암의 장경'].mean())
test_df = test_df.fillna(0)



train_df, val_df, train_labels, val_labels = train_test_split(
                                                    train_df.drop(columns=['N_category']), 
                                                    train_df['N_category'], 
                                                    test_size=0.2, 
                                                    random_state=CFG['SEED']
                                                )




def get_values(value):
    return value.values.reshape(-1, 1)

numeric_cols = ['나이', '암의 장경', 'ER_Allred_score', 'PR_Allred_score', 'KI-67_LI_percent', 'HER2_SISH_ratio']
ignore_cols = ['ID', 'img_path', 'mask_path', '수술연월일', 'N_category']

for col in train_df.columns:
    if col in ignore_cols:
        continue
    if col in numeric_cols:
        scaler = StandardScaler()
        train_df[col] = scaler.fit_transform(get_values(train_df[col]))
        val_df[col] = scaler.transform(get_values(val_df[col]))
        test_df[col] = scaler.transform(get_values(test_df[col]))
    else:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(get_values(train_df[col]))
        val_df[col] = le.transform(get_values(val_df[col]))
        test_df[col] = le.transform(get_values(test_df[col]))




class CustomDataset(Dataset):
    def __init__(self, medical_df, labels, transforms=None, test=False):
        self.medical_df = medical_df
        self.transforms = transforms
        self.labels = labels
        self.test = test
        
    def __getitem__(self, index):
        if self.test:
            data_path = "/home/ljj0512/private/workspace/CP_Multi-modal_Cencer-metastasis_DACON/data/test_imgs"
        else:
            data_path = "/home/ljj0512/private/workspace/CP_Multi-modal_Cencer-metastasis_DACON/data/train_imgs"
        img_path = os.path.join(data_path,self.medical_df["img_path"].iloc[index][-14:])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
                
        if self.labels is not None:
            tabular = torch.Tensor(self.medical_df.drop(columns=['ID', 'img_path', 'mask_path', '수술연월일']).iloc[index])
            label = self.labels[index]
            return image, tabular, label
        else:
            tabular = torch.Tensor(self.medical_df.drop(columns=['ID', 'img_path', '수술연월일']).iloc[index])
            return image, tabular
        
    def __len__(self):
        return len(self.medical_df)


train_transforms = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=(0, 0.2), contrast_limit=(0, 0.5), p=0.8),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.2, 0.5), always_apply=False, p=1.0),
            A.ColorJitter(brightness = 0.2, saturation = 0.5, hue = 0.5),
            A.VerticalFlip(),
            A.HorizontalFlip(),
            ])


train_transforms = A.Compose([
                            A.HorizontalFlip(),
                            A.VerticalFlip(),
                            A.Rotate(limit=90, border_mode=cv2.BORDER_CONSTANT,p=0.3),
                            A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])

test_transforms = A.Compose([
                            A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])




train_dataset = CustomDataset(train_df, train_labels.values, train_transforms)
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

val_dataset = CustomDataset(val_df, val_labels.values, test_transforms)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)




class ImgFeatureExtractor(nn.Module):
    def __init__(self):
        super(ImgFeatureExtractor, self).__init__()
        # self.backbone = models.efficientnet_b0(pretrained=True)
        self.backbone = create_model(CFG["MODEL"],pretrained=True)
        self.embedding = nn.Linear(1000,512)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.embedding(x)
        return x




class TabularFeatureExtractor(nn.Module):
    def __init__(self):
        super(TabularFeatureExtractor, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(in_features=23, out_features=128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=512),
        )
        
        
    def forward(self, x):
        # res = nn.Linear(in_features=23, out_features=512)(x)
        x = self.embedding(x)
        # x = res+x
        return x
    





class ClassificationModel(nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()
        self.img_feature_extractor = ImgFeatureExtractor()
        self.tabular_feature_extractor = TabularFeatureExtractor()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=1024, out_features=1),
            # nn.Sigmoid(),
        )
        
    def forward(self, img, tabular):
        img_feature = self.img_feature_extractor(img)
        tabular_feature = self.tabular_feature_extractor(tabular)
        feature = torch.cat([img_feature, tabular_feature], dim=-1)
        output = self.classifier(feature)
        return output





def validation(model, criterion, val_loader, device):
    model.eval()
    pred_labels = []
    true_labels = []
    val_loss = []
    threshold = 0.5
    with torch.no_grad():
        for img, tabular, label in tqdm(iter(val_loader)):
            true_labels += label.tolist()
            
            img = img.float().to(device)
            tabular = tabular.float().to(device)
            label = label.float().to(device)
            
            model_pred = model(img, tabular)
            
            loss = criterion(model_pred, label.reshape(-1,1))
            
            val_loss.append(loss.item())
            
            model_pred = model_pred.squeeze(1).to('cpu')  
            pred_labels += model_pred.tolist()
    
    pred_labels = np.where(np.array(pred_labels) > threshold, 1, 0)
    val_score = metrics.f1_score(y_true=true_labels, y_pred=pred_labels, average='macro')
    return np.mean(val_loss), val_score




def train(model, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    
    best_score = 0
    best_model = None
    
    for epoch in range(1, CFG['EPOCHS']+1):
        model.train()
        train_loss = []
        for img, tabular, label in tqdm(iter(train_loader)):
            img = img.float().to(device)
            tabular = tabular.float().to(device)
            label = label.float().to(device)
            
            optimizer.zero_grad()
            
            model_pred = model(img, tabular)
            
            loss = criterion(model_pred, label.reshape(-1,1))
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
        val_loss, val_score = validation(model, criterion, val_loader, device)
        print(f'Epoch [{epoch}], Train Loss : [{np.mean(train_loss):.5f}] Val Loss : [{val_loss:.5f}] Val Score : [{val_score:.5f}]')
        wandb.log({
                    "train_loss" : np.mean(train_loss),
                    "valid_loss" : val_loss,
                    "valid_score" : val_score,
        })
        if scheduler is not None:
            scheduler.step(val_score)
        
        if best_score < val_score:
            best_score = val_score
            best_model = model
            save_checkpoint({
                'epoch': epoch,
                'best_score': best_score,
                'state_dict': best_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            })
    
    return best_model


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    log_dir = "/home/ljj0512/private/workspace/CP_Multi-modal_Cencer-metastasis_DACON/log"
    directory = f"{log_dir}/base/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory+filename
    torch.save(state, filename)


model = ClassificationModel()
print("params: {:,}".format(sum([p.data.nelement() for p in model.parameters()])))
print()
model = nn.DataParallel(model)
model.eval()
optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, threshold_mode='abs',min_lr=1e-8, verbose=True)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["EPOCHS"]/5, eta_min=1e-8)
infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device)

# checkpoint = torch.load("/home/ljj0512/private/workspace/CP_Multi-modal_Cencer-metastasis_DACON/log/base/checkpoint.pth.tar")
# infer_model = nn.DataParallel(ClassificationModel()).cuda()
# infer_model.load_state_dict(checkpoint["state_dict"])

test_dataset = CustomDataset(test_df, None, test_transforms, test=True)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=1)

def inference(model, test_loader, device):
    # model.to(device)
    model.eval()
    preds = []
    threshold = 0.5
    
    with torch.no_grad():
        for img, tabular in tqdm(iter(test_loader)):
            img = img.float().to(device)
            tabular = tabular.float().to(device)
            
            model_pred = model(img, tabular)
            
            model_pred = model_pred.squeeze(1).to('cpu')
            
            preds += model_pred.tolist()
    
    preds = np.where(np.array(preds) > threshold, 1, 0)
    
    return preds



preds = inference(infer_model, test_loader, device)


sample_dir = "/home/ljj0512/private/workspace/CP_Multi-modal_Cencer-metastasis_DACON/data/sample_submission.csv"
submit = pd.read_csv(sample_dir)
submit['N_category'] = preds
submit.to_csv('./submit01.csv', index=False)