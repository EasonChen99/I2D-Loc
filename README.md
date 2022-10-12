# I2D-Loc
This repository contains the source code for our paper:

[I2D-Loc: Camera localization via depth to image flow](https://levenberg.github.io/I2D-Loc/)<br/>
ISPRS 2022 <br/>
Kuangyi Chen and Huai Yu<br/>

<img src="Network.png">

## Requirements
The code has been tested with PyTorch 1.12 and Cuda 11.6.
```Shell
conda create -n i2d python=3.7 -y
conda activate i2d
pip install -r requirements.txt
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
cd core/correlation_package
python setup.py install
cd ..
cd visibility_package
python setup.py install
cd ../..
```

## Demos
Pretrained models can be downloaded from [google drive](https://drive.google.com/drive/folders/19VWNCPR1me7SnON1NYJRFrdgd1sKj052?usp=sharing)

You can demo a trained model on a sequence of frames
```Shell
python demo.py --load_checkpoints=checkpoints/2_10/checkpoints.pth --render
```

## Required Data
To evaluate/train I2D-Loc, you will need to download the required datasets.
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)



By default `datasets.py` will search for the datasets in these locations. You can create symbolic links to wherever the datasets were downloaded in the `datasets` folder

```Shell
├── datasets
    ├── Sintel
        ├── test
        ├── training
    ├── KITTI
        ├── testing
        ├── training
        ├── devkit
    ├── FlyingChairs_release
        ├── data
    ├── FlyingThings3D
        ├── frames_cleanpass
        ├── frames_finalpass
        ├── optical_flow
```

## Evaluation
You can evaluate a trained model using `evaluate.py`
```Shell
python evaluate.py --model=models/raft-things.pth --dataset=sintel --mixed_precision
```

## Training
We used the following training schedule in our paper (2 GPUs). Training logs will be written to the `runs` which can be visualized using tensorboard
```Shell
./train_standard.sh
```

If you have a RTX GPU, training can be accelerated using mixed precision. You can expect similiar results in this setting (1 GPU)
```Shell
./train_mixed.sh
```

## (Optional) Efficent Implementation
You can optionally use our alternate (efficent) implementation by compiling the provided cuda extension
```Shell
cd alt_cuda_corr && python setup.py install && cd ..
```
and running `demo.py` and `evaluate.py` with the `--alternate_corr` flag Note, this implementation is somewhat slower than all-pairs, but uses significantly less GPU memory during the forward pass.
