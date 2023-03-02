import os
import argparse
import rasterio as rio
from tqdm import tqdm
import numpy as np
import hydra
from matplotlib import pyplot as plt

from src.model import RiverModel
from predict import predict

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    if cfg.predict_folder.folder == "" or cfg.predict_folder.river == "" or cfg.predict_folder.model == "" or cfg.predict_folder.output == "":
        raise Exception("Required predict_folder.folder predict_folder.river predict_folder.model and predict_folder.output")

    model = RiverModel.load_from_checkpoint(cfg.predict_folder.model).to(cfg.predict_folder.device)
    river = rio.open(cfg.predict_folder.river)
    os.makedirs(cfg.predict_folder.output, exist_ok=True)

    for filename in tqdm(os.listdir(cfg.predict_folder.folder)):
        file = os.path.join(cfg.predict_folder.folder, filename)
        raster = rio.open(file)
        prediction = output = predict(
            raster=raster,
            river=river,
            model=model,
            device=cfg.predict_folder.device,
            window=cfg.train.window_size,
            preprocess_std=cfg.data.clip.raster_std
        )
        
        output_folder = os.path.join(cfg.predict_folder.output, filename)
        os.makedirs(output_folder, exist_ok=True)
        with open(os.path.join(output_folder, "prediction.np"), "wb") as f:
            np.save(f, prediction.cpu().numpy())

        plt.matshow(prediction.cpu().numpy()[0])
        plt.savefig(os.path.join(output_folder, "prediction.png"))
        plt.close()


if __name__ == "__main__":
    main()
