import argparse
import os
import torch
import numpy as np
import rasterio as rio
import pandas as pd
from matplotlib import pyplot as plt
import hydra
from tqdm import tqdm
import json
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from src.model import RiverModel
from src.dataset import RiverSegmentationDataset
from src.utils import preprocess_mask

@torch.no_grad()
def evaluate(cfg):
    model = RiverModel.load_from_checkpoint(cfg.evaluate.model).to(cfg.evaluate.device)
    with open(cfg.data.dataset.test) as f:
        dataset = json.load(f)

    y_true = []
    y_pred = []
    for elem in tqdm(dataset):
        mask = rio.open(os.path.join(cfg.data.masks, elem["mask"]["filename"]))
        raster = rio.open(os.path.join(cfg.data.rasters, elem["raster"]["filename"]))
        
        with open(os.path.join(cfg.evaluate.predictions, elem["raster"]["filename"], "prediction.np"), "rb") as f:
            prediction = torch.tensor(np.load(f))

        mask_raster = preprocess_mask(torch.tensor(mask.read()).to(cfg.evaluate.device), cfg)
        
        window = cfg.train.window_size
        prediction_raster = torch.zeros_like(mask_raster)
        for i in range(0, int(mask_raster.shape[1]/window)-1):
            for j in range(0, int(mask_raster.shape[2]/window)-1):
                mask_window = mask_raster[:, i*window:(i+1)*window, j*window:(j+1)*window]
                
                mask_x, mask_y  = rio.transform.xy(mask.transform, i*window, j*window)
                prediction_i, prediction_j = raster.index(mask_x, mask_y)

                if prediction_i < 0: prediction_i = 0
                if prediction_j < 0: prediction_j = 0

                prediction_window = prediction[:, prediction_i:prediction_i+window, prediction_j:prediction_j+window]
                prediction_raster[:, i*window:(i+1)*window, j*window:(j+1)*window] = prediction_window

        y_true.extend(mask_raster.view(mask_raster.numel()).cpu().numpy().astype(int).tolist())
        y_pred.extend(prediction_raster.view(prediction_raster.numel()).cpu().numpy().astype(int).tolist())
    
    #output_folder = os.path.join(cfg.evaluate.predictions, elem["raster"]["filename"])
    
    #df = pd.DataFrame(dict(true=y_true, pred=y_pred))
    #df.to_csv(os.path.join(output_folder, "classification.csv"))
    
    cm = confusion_matrix(y_true, y_pred)
    am = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cr = classification_report(y_true, y_pred, target_names=["no water", "water"])
    
    print(f'''
=== CONFUSION MATRIX ===
{str(cm)}
========================

=== ACCURACY MATRIX ===
{str(am)}
========================

=== CLASSIFICATION REPORT ===
{str(cr)}
=============================
''')

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    if cfg.evaluate.model == "" or cfg.evaluate.predictions == "":
        raise Exception("You have to specify evaluate.model and evaluate.predictions")
    
    evaluate(cfg)

if __name__ == "__main__":
    main()

