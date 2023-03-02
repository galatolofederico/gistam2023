import os
import argparse
import hydra
import rasterio as rio

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg):
    river = rio.open(cfg.data.river).read()
    print(f"RIVER = {river.shape}")
    
    for filename in os.listdir(cfg.data.rasters):
        raster = rio.open(os.path.join(cfg.data.rasters, filename)).read()
        print(f"RASTER[{filename}] = {raster.shape}")
    
    for filename in os.listdir(cfg.data.masks):
        mask = rio.open(os.path.join(cfg.data.masks, filename)).read()
        print(f"MASK[{filename}] = {mask.shape}")

if __name__ == "__main__":
    main()