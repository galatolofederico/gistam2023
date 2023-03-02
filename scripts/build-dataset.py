import os
import random
import torch
import rasterio as rio
import json
import hydra
from datetime import datetime

from src.utils import get_date_from_filename

def serialize_datetime(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError ("Type %s not serializable" % type(obj))

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg):
    raster_files = []
    for filename in os.listdir(cfg.data.rasters):
        raster_files.append(dict(
            filename=filename,
            date=get_date_from_filename(filename)
        ))

    mask_files = []
    for filename in os.listdir(cfg.data.masks):
        mask_files.append(dict(
            filename=filename,
            date=get_date_from_filename(filename)
        ))

    dataset = []
    for mask_file in mask_files:
        closer_raster = raster_files[0]
        closer_delta = abs(mask_file["date"] - closer_raster["date"])
        for raster_file in raster_files:
            delta = abs(mask_file["date"] - raster_file["date"])
            if delta < closer_delta:
                closer_raster = raster_file
                closer_delta = delta
        
        dataset.append(dict(
            mask=mask_file,
            raster=closer_raster
        ))

    random.shuffle(dataset)
    train_pivot = int(len(dataset)*cfg.data.dataset.training_perc)
    train = dataset[:train_pivot]
    test = dataset[train_pivot:]

    with open(cfg.data.dataset.full, "w") as f:
        json.dump(dataset, f, indent=4, default=serialize_datetime)
    
    with open(cfg.data.dataset.train, "w") as f:
        json.dump(train, f, indent=4, default=serialize_datetime)
    
    with open(cfg.data.dataset.test, "w") as f:
        json.dump(test, f, indent=4, default=serialize_datetime)
        

if __name__ == "__main__":
    main()