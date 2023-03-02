import argparse
import os
import json
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import datetime
import rasterio as rio

parser = argparse.ArgumentParser()

parser.add_argument("--folder", required=True)
parser.add_argument("--save", default="")
parser.add_argument("--save-plot", default="")
parser.add_argument("--plot", action="store_true")
parser.add_argument("--ma", action="store_true")
parser.add_argument("--ma-window", type=int, default=5)
parser.add_argument("--dataset", action="store_true")

args = parser.parse_args()

dates = []
predictions = []
for filename in os.listdir(args.folder):    
    date = datetime.datetime.strptime(filename.split(".")[0], "%Y%m%d")
    if args.dataset:
        prediction = rio.open(os.path.join(args.folder, filename)).read()
        prediction = prediction.clip(0, 1)
    else:
        with open(os.path.join(args.folder, filename, "prediction.np"), "rb") as f:
            prediction = np.load(f)

    dates.append(date)
    predictions.append(prediction.sum())

df = pd.DataFrame(dict(date=dates, prediction=predictions))
df = df.sort_values(by="date")
if args.ma:
    df["prediction"] = df["prediction"].rolling(args.ma_window).mean()

if args.plot:
    df.plot(x="date", y="prediction")
    if args.save_plot:
        plt.savefig(args.save_plot)
    else:
        plt.show()

if args.save:
    df.to_excel(args.save)
