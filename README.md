# A Joint Sequence Fusion Model for Video Question Answering and Retrieval

This project hosts the tensorflow implementation for our **ECCV 2018** paper, **A Joint Sequence Fusion Model for Video Question Answering and Retrieval}**.


## Reference

If you use this code or dataset as part of any published research, please refer the following paper.

```
@inproceedings{
  author    = {Youngjae Yu and Jongseok Kim and Gunhee Kim},
  title     = "{A Joint Sequence Fusion Model for Video Question Answering and Retrieval}"
  booktitle = {ECCV},
  year      = 2018
}
```


## Setup


### Install dependencies
```
pip install -r requirements.txt
```

### Setup python paths
```
git submodule update --init --recursive
add2virtualenv .
```



### Prepare Data

- Video Feature

  1. Download [LSMDC data](https://sites.google.com/site/describingmovies/lsmdc-2016/download).

  2. Extract rgb features using pool5 layer of the pretrained ResNet-152 model.

  3. Extract audio features using [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset).

  4. Concat rgb and video features and save it into hdf5 file, and save it in 'dataset/LSMDC/LSMDC16_features/RESNET_pool5wav.hdf5'.

- Dataset
  - We processed raw data frames file in LSMDC17 and MSR-VTT dataset
  - [Download dataframe files](https://drive.google.com/drive/folders/1_Wyr2VEWU4N-OgLBaQDGWXqD2TXXUBaF?usp=sharing)
  - Save these files in "dataset/LSMDC/DataFrame"

- Vocabulary

  - We make word embedding matrix using GloVe Vector.
  - [Download vocabulary files](https://drive.google.com/drive/folders/1GsArc0BuxzMAYobzbhWMj7MEUPDuneeC?usp=sharing)
  - Save these files in "dataset/LSMDC/Vocabulary"


### Training

modify `configuartion.py` to suit your environment.

  - train_tag can be 'MC', 'FIB'

Run `train.py`.

```
python train.py --tag="tag"
```

