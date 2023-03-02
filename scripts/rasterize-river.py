import argparse
import rasterio as rio
from rasterio import features
import geopandas as gpd

parser = argparse.ArgumentParser()

parser.add_argument("--shp", default="./data/data/river/river.shp")
parser.add_argument("--raster-template", default="./data/data/IDAN/20190407.tif")
parser.add_argument("--output", default="./data/data/river/river.tif")

args = parser.parse_args()


river = gpd.read_file(args.shp)
rst = rio.open(args.raster_template)

meta = rst.meta.copy()
meta.update(compress='lzw')

with rio.open(args.output, 'w+', **meta) as out:
    out_arr = out.read(1)
    shapes = [[shape, 1] for shape in river.geometry]
    burned = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
    out.write_band(1, burned)
