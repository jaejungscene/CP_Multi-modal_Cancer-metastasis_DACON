import os
from args import get_args_parser
args = get_args_parser().parse_args()
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
import numpy as np
import torch
import torch.nn as nn
import tqdm
import pandas as pd
import cv2
import random
from torch.utils.data import Dataset, DataLoader
from model import load_model
from args import get_args_parser
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

CFG = {
    "BATCH_SIZE":32,
    "IMG_SIZE":224
}

device = torch.device('cuda')
test_df = pd.read_csv('./data/test.csv')

test_transforms = A.Compose([
                            A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])

class CustomDataset(Dataset):
    def __init__(self, medical_df, labels, transforms=None):
        self.medical_df = medical_df
        self.transforms = transforms
        self.labels = labels
        
    def __getitem__(self, index):
        datadir = "/home/ljj0512/private/workspace/CP_Multi-modal_Cencer-metastasis_DACON/data"
        img_path = self.medical_df['img_path'].iloc[index]
        img_path = os.path.join(datadir, img_path[2:])
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

test_dataset = CustomDataset(test_df, None, test_transforms)
print(test_dataset[0][0].shape)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=4)

def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    preds = []
    threshold = 0.5
    
    with torch.no_grad():
        for img, tabular in test_loader:
            img = img.float().to(device)
            tabular = tabular.float().to(device)
            
            model_pred = model(img, tabular)

            model_pred = model_pred.squeeze(1).to('cpu')
            
            preds += model_pred.tolist()
    
    preds = np.where(np.array(preds) > threshold, 1, 0)
    
    return preds

infer_model = load_model(args, infer=True)
infer_model = nn.DataParallel(infer_model).cuda()
checkpoint = torch.load('/home/ljj0512/private/workspace/CP_Multi-modal_Cencer-metastasis_DACON/log/2022-11-04 09:49:23-1/checkpoint.pth.tar')
infer_model.load_state_dict(checkpoint["state_dict"])
print("start")
preds = inference(infer_model, test_loader, device)

datadir = "/home/ljj0512/private/workspace/CP_Multi-modal_Cencer-metastasis_DACON/data"
submit = pd.read_csv(datadir+'/sample_submission.csv')
submit['N_category'] = preds
submit.to_csv(datadir+'/submit.csv', index=False)

