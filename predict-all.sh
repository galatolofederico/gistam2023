#!/bin/sh

MODEL_TAG=30k

python predict-folder.py \
    predict_folder.device=cuda:1 \
    predict_folder.folder=data/data/IDAN \
    predict_folder.river=data/data/river/river.tif \
    predict_folder.model=model-IDAN-$MODEL_TAG.pth \
    predict_folder.output=results/$MODEL_TAG/IDAN

python predict-folder.py \
    predict_folder.device=cuda:1 \
    predict_folder.folder=data/data/leesigma \
    predict_folder.river=data/data/river/river.tif \
    predict_folder.model=model-leesigma-$MODEL_TAG.pth \
    predict_folder.output=results/$MODEL_TAG/leesigma

python predict-folder.py \
    predict_folder.device=cuda:1 \
    predict_folder.folder=data/data/intensity \
    predict_folder.river=data/data/river/river.tif \
    predict_folder.model=model-intensity-$MODEL_TAG.pth \
    predict_folder.output=results/$MODEL_TAG/intensity

python predict-folder.py \
    predict_folder.device=cuda:1 \
    predict_folder.folder=data/data/IDAN \
    predict_folder.river=data/data/river/river.tif \
    predict_folder.model=model-IDAN-$MODEL_TAG-weights.pth \
    predict_folder.output=results/$MODEL_TAG-weights/IDAN

python predict-folder.py \
    predict_folder.device=cuda:1 \
    predict_folder.folder=data/data/leesigma \
    predict_folder.river=data/data/river/river.tif \
    predict_folder.model=model-leesigma-$MODEL_TAG-weights.pth \
    predict_folder.output=results/$MODEL_TAG-weights/leesigma

python predict-folder.py \
    predict_folder.device=cuda:1 \
    predict_folder.folder=data/data/intensity \
    predict_folder.river=data/data/river/river.tif \
    predict_folder.model=model-intensity-$MODEL_TAG-weights.pth \
    predict_folder.output=results/$MODEL_TAG-weights/intensity
