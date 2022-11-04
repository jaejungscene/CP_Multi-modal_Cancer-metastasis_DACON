import numpy as np
import torch
from dataset import CustomDataset, DataLoader
import tqdm
import pandas as pd

device = torch.device('cuda')
test_df = pd.read_csv('./data/test.csv')

test_dataset = CustomDataset(test_df, None, test_transforms)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

def inference(model, test_loader, device):
    model.to(device)
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

submit = pd.read_csv('./sample_submission.csv')
submit['N_category'] = preds
submit.to_csv('./submit.csv', index=False)