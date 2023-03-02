from datetime import datetime
from omegaconf import DictConfig, OmegaConf
import flatdict

def preprocess(*, cfg, raster, mask, river):
    mask = mask.clip(cfg.data.clip.mask.min, cfg.data.clip.mask.max)
    raster = raster.clip(raster.mean() - cfg.data.clip.raster_std*raster.std(), raster.mean() + cfg.data.clip.raster_std*raster.std())
    river = river.clip(0, 1)

    return raster, mask, river

def preprocess_raster(raster, std):
    return raster.clip(raster.mean() - std*raster.std(), raster.mean() + std*raster.std())

def preprocess_mask(mask, cfg):
    return mask.clip(cfg.data.clip.mask.min, cfg.data.clip.mask.max)

def get_date_from_filename(filename):
    return datetime.strptime(filename.split(".")[0], "%Y%m%d")

def hp_from_cfg(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    return dict(flatdict.FlatDict(cfg, delimiter="/"))