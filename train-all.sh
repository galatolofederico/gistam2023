#!/bin/sh

WANDB_TAG="second-30k-long"
STEPS="30000"


python train.py \
    -cn IDAN \
    wandb.log=True \
    wandb.tag=$WANDB_TAG \
    train.steps=$STEPS \
    train.save_file="$(pwd)/model-IDAN-30k.pth"

python train.py \
    -cn leesigma \
    wandb.log=True \
    wandb.tag=$WANDB_TAG \
    train.steps=$STEPS \
    train.save_file="$(pwd)/model-leesigma-30k.pth"

python train.py \
    -cn intensity \
    wandb.log=True \
    wandb.tag=$WANDB_TAG \
    train.steps=$STEPS \
    train.save_file="$(pwd)/model-intensity-30k.pth"

python train.py \
    -cn IDAN \
    wandb.log=True \
    wandb.tag=$WANDB_TAG \
    train.steps=$STEPS \
    model.bce_weights=True \
    train.save_file="$(pwd)/model-IDAN-30k-weights.pth"

python train.py \
    -cn leesigma \
    wandb.log=True \
    wandb.tag=$WANDB_TAG \
    train.steps=$STEPS \
    model.bce_weights=True \
    train.save_file="$(pwd)/model-leesigma-30k-weights.pth"

python train.py \
    -cn intensity \
    wandb.log=True \
    wandb.tag=$WANDB_TAG \
    train.steps=$STEPS \
    model.bce_weights=True \
    train.save_file="$(pwd)/model-intensity-30k-weights.pth"
