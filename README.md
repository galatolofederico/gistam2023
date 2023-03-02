# gistam2023

Repository for the paper [Using deep learning and radar backscatter for mapping river water surface]()

## Installation

Clone this repository

```
git clone https://github.com/galatolofederico/gistam2023.git
cd gistam2023
```

Create a virtualenv and install the requirements

```
virtualenv --python=python3.8 env
. ./env/bin/activate
pip install -r requirements.txt
```

If you are interested in the dataset please [contact us](mailto:federico.galatolo@ing.unipi.it)

## Usage

### To build the datasets run

```
python -m scripts.build-dataset -cn IDAN
python -m scripts.build-dataset -cn leesigma
python -m scripts.build-dataset -cn intensity
```

### To train a model run

```
python train.py \ 
    -cn <CONFIG> \
    wandb.log=<True/False> \
    wandb.tag=<WANDB_TAG> \
    train.steps=<STEPS> \
    train.save_file="$(pwd)/model-<CONFIG>-<STEPS>.pth"
```
Where
 - `<CONFIG>` can be: `IDAN` `leesigma` or `itensity`
 - `<STEPS>` has been set to `30000` in the paper

This script will save the trained model in `$(pwd)/model-<CONFIG>-<STEPS>.pth`

### To run the inference with a model run

```
python predict-folder.py \
    predict_folder.device=cuda:0 \
    predict_folder.folder=<DATA_FOLDER> \
    predict_folder.river=<RIVER_MASK_TIF> \
    predict_folder.model=<MODEL> \
    predict_folder.output=<OUTPUT_FOLDER>
```

## Scripts

To train all the models from the paper run

```
./train_all.sh
```

To run the inference with the trained models run

```
./predict-all.sh
```

To compute the performance metrics for all the models run

```
./evaluate-all.sh
```

## Contributions and license

The code is released as Free Software under the [GNU/GPLv3](https://choosealicense.com/licenses/gpl-3.0/) license. Copying, adapting and republishing it is not only allowed but also encouraged. 

For any further question feel free to reach me at  [federico.galatolo@ing.unipi.it](mailto:federico.galatolo@ing.unipi.it) or on Telegram  [@galatolo](https://t.me/galatolo)