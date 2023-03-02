#!/bin/sh

MODEL_TAG=30k

python evaluate.py -cn IDAN \
    evaluate.model=model-IDAN-$MODEL_TAG.pth \
    evaluate.predictions=./results/$MODEL_TAG/IDAN \
    evaluate.device=cuda:1

python evaluate.py -cn leesigma \
    evaluate.model=model-leesigma-$MODEL_TAG.pth \
    evaluate.predictions=./results/$MODEL_TAG/leesigma \
    evaluate.device=cuda:1

python evaluate.py -cn intensity \
    evaluate.model=model-intensity-$MODEL_TAG.pth \
    evaluate.predictions=./results/$MODEL_TAG/intensity \
    evaluate.device=cuda:1

python evaluate.py -cn IDAN \
    evaluate.model=model-IDAN-$MODEL_TAG-weights.pth \
    evaluate.predictions=./results/$MODEL_TAG-weights/IDAN \
    evaluate.device=cuda:1

python evaluate.py -cn leesigma \
    evaluate.model=model-leesigma-$MODEL_TAG-weights.pth \
    evaluate.predictions=./results/$MODEL_TAG-weights/leesigma \
    evaluate.device=cuda:1

python evaluate.py -cn intensity \
    evaluate.model=model-intensity-$MODEL_TAG-weights.pth \
    evaluate.predictions=./results/$MODEL_TAG-weights/intensity \
    evaluate.device=cuda:1

