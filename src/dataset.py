import os
import random
import torch
import rasterio as rio
import json
import numpy as np
from matplotlib import pyplot as plt

from src.utils import preprocess

class RiverSegmentationDataset(torch.utils.data.IterableDataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.river = rio.open(cfg.data.river)
        self.river_raster = self.river.read()
        self.river_points = np.argwhere(self.river_raster > 0)[:, 1:]

        if split == "train":
            dataset_file = self.cfg.data.dataset.train
        elif split == "test":
            dataset_file = self.cfg.data.dataset.test
        else:
            raise ValueError(f"Unknown split {split}")
        
        with open(dataset_file, "r") as f:
            dataset = json.load(f)
        
        self.dataset = []
        for elem in dataset:
            mask = rio.open(os.path.join(cfg.data.masks, elem["mask"]["filename"]))
            raster = rio.open(os.path.join(cfg.data.rasters, elem["raster"]["filename"]))
            
            self.dataset.append(dict(
                mask=mask,
                mask_raster=mask.read(),
                raster=raster,
                raster_raster=raster.read(),
            ))
    
    def __iter__(self):
        return self

    def sample(self):
        W = int(self.cfg.train.window_size/2)

        elem = random.choice(self.dataset)
        river_i, river_j = random.choice(self.river_points)
        river_x, river_y = rio.transform.xy(self.river.transform, river_i, river_j)
        
        raster_i, raster_j = elem["raster"].index(river_x, river_y)
        mask_i, mask_j = elem["mask"].index(river_x, river_y)

        return dict(
            elem=elem,
            raster=[raster_i, raster_j],
            raster_bounds = ((W, elem["raster"].shape[0] - W - 1), (W, elem["raster"].shape[1] - W - 1)),
            mask=[mask_i, mask_j],
            mask_bounds = ((W, elem["mask"].shape[0] - W - 1), (W, elem["mask"].shape[1] - W - 1)),
            river=[river_i, river_j]
        )

    def check_bounds(self, sample):
        if sample["raster"][0] < sample["raster_bounds"][0][0] or \
           sample["raster"][0] > sample["raster_bounds"][0][1] or \
           sample["raster"][1] < sample["raster_bounds"][1][0] or \
           sample["raster"][1] > sample["raster_bounds"][1][1]:
           return False
        
        if sample["mask"][0] < sample["mask_bounds"][0][0] or \
           sample["mask"][0] > sample["mask_bounds"][0][1] or \
           sample["mask"][1] < sample["mask_bounds"][1][0] or \
           sample["mask"][1] > sample["mask_bounds"][1][1]:
           return False
        
        return True

    def __next__(self):
        W = int(self.cfg.train.window_size/2)
        
        while True:
            sample = self.sample()
            if not self.check_bounds(sample):
                continue
            
            raster = sample["elem"]["raster_raster"]
            mask = sample["elem"]["mask_raster"]
            river = self.river_raster

            raster_i = sample["raster"][0]
            raster_j = sample["raster"][1]
            mask_i = sample["mask"][0]
            mask_j = sample["mask"][1]
            river_i = sample["river"][0]
            river_j = sample["river"][1]

            raster = raster[:, raster_i-W:raster_i+W, raster_j-W:raster_j+W]
            mask = mask[:, mask_i-W:mask_i+W, mask_j-W:mask_j+W]
            river = river[:, river_i-W:river_i+W, river_j-W:river_j+W]
            
            raster = torch.tensor(raster)
            mask = torch.tensor(mask)
            river = torch.tensor(river)

            if raster.shape != mask.shape:
                continue
            if raster.shape != river.shape:
                continue
            
            raster, mask, river = preprocess(
                cfg=self.cfg,
                raster=raster,
                mask=mask,
                river=river
            )

            return raster, mask, river


import hydra
@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg):
    ds = RiverSegmentationDataset(cfg, "train")
    it = iter(ds)
    while True:
        r, m, ri = next(it)
        print(r.shape, m.shape, ri.shape)
        fig, axes = plt.subplots(3)
        axes[0].matshow(r.numpy()[0])
        axes[1].matshow(m.numpy()[0])
        axes[2].matshow(ri.numpy()[0])

        plt.show()

if __name__ == "__main__":
    main()