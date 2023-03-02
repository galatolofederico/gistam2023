import argparse
import torch
import rasterio as rio
from matplotlib import pyplot as plt
import hydra

from src.model import RiverModel
from src.utils import preprocess_raster

def to_batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

@torch.no_grad()
def predict(*, raster, river, model, device, window, preprocess_std):
    raster_raster = torch.tensor(raster.read()).to(device)
    river_raster = torch.tensor(river.read()).to(device)

    inputs = []
    output = torch.zeros_like(raster_raster)
    for i in range(0, int(raster_raster.shape[1]/window)-1):
        for j in range(0, int(raster_raster.shape[2]/window)-1):
            raster_window = raster_raster[:, i*window:(i+1)*window, j*window:(j+1)*window]
            
            raster_x, raster_y  = rio.transform.xy(raster.transform, i*window, j*window)
            river_i, river_j = river.index(raster_x, raster_y)

            if river_i < 0: river_i = 0
            if river_j < 0: river_j = 0

            river_window = river_raster[:, river_i:river_i+window, river_j:river_j+window]

            raster_window = preprocess_raster(raster_window, preprocess_std)
            river_window = river_window.clip(0, 1)

            inputs.append(dict(
                raster=raster_window,
                river=river_window,
                slice=(i*window, (i+1)*window, j*window, (j+1)*window)
            ))

            #window_output = model(raster_window.unsqueeze(0)).squeeze(0) * river_window
            #output[:, i*window:(i+1)*window, j*window:(j+1)*window] = window_output 
    
    rasters = to_batch([input["raster"] for input in inputs], 32)
    rivers = to_batch([input["river"] for input in inputs], 32)
    slices = [input["slice"] for input in inputs]
    
    out_batches = []
    for batch_rasters, batch_rivers in zip(rasters, rivers):
        batch_rasters = torch.stack(batch_rasters)
        batch_rivers = torch.stack(batch_rivers)
        out_batches.append(torch.sigmoid(model(batch_rasters)) * batch_rivers)
    
    out_batches = torch.cat(out_batches)
    for out_batch, out_slice in zip(out_batches, slices):
        output[:, out_slice[0]:out_slice[1], out_slice[2]:out_slice[3]] = out_batch
    
    #output[output < 0.5] = 0
    #output[output >= 0.5] = 1

    return output

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    if cfg.predict.model == "" or cfg.predict.raster == "" or cfg.predict.river == "":
        raise Exception("You have to specify predict.model predict.raster and predict.river")
    model = RiverModel.load_from_checkpoint(cfg.predict.model).to(cfg.predict.device)
    raster = rio.open(cfg.predict.raster)
    river = rio.open(cfg.predict.river)

    output = predict(
        raster=raster,
        river=river,
        model=model,
        device=cfg.predict.device,
        window=cfg.train.window_size,
        preprocess_std=cfg.data.clip.raster_std
    )

    m = output.mean()
    s = output.std()

    output = output.clip(m-2*s, m+2*s)

    if cfg.predict.plot:
        plt.matshow(output[0].cpu().numpy())
        plt.show()

if __name__ == "__main__":
    main()

