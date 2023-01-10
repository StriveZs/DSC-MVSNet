# README for DSC-MVSNet

PyTorch implement of “DSC-MVSNet: Attention Aware Cost Volume Regularization Based On Depthwise Separable Convolution for Multi-View Stereo”.

## Installation

```python
pip install -r requirments.txt
```

## Testing

Download the preprocessed test data [DTU testing data](https://drive.google.com/open?id=135oKPefcPTsdtLRzoDAQtPpHuoIrpRI_) (from [Original MVSNet](https://github.com/YoYo000/MVSNet)) and unzip it as the `DTU_TESTING` folder, which should contain one `cams` folder, one `images` folder and one `pair.txt` file.

Test with the pretrained model

```python
python dscmvsnet/test.py --cfg configs/dtu.yaml TEST.WEIGHT outputs/pretrained.pth
```

### dtu.yaml

Please set the following configuration

```python
OUTPUT_DIR: ""  # logfile and .pth save path 
DATA:
  TRAIN:
    ROOT_DIR: "" # training set path
		NUM_VIRTUAL_PLANE:
	VAL:
	    ROOT_DIR: "" # validation set path
	TEST:
	    ROOT_DIR: "" # testing set path
	    NUM_VIEW:
	    IMG_HEIGHT:
	    IMG_WIDTH:
	    NUM_VIRTUAL_PLANE:
	    INTER_SCALE: 2.13
	    MODE: "dtu" # dtu or tanks
TEST:
  WEIGHT: "" # .pth path
  BATCH_SIZE: 1
```

## Depth Fusion

We need to apply depth fusion `tools/depthfusion.py` to get the complete point cloud. Please refer to [MVSNet](https://github.com/YoYo000/MVSNet) for more details. And use `tools/rename_ply.py` to get the rename results.
```python
python tools/depthfusion.py -f dtu -n flow2

python tools/rename_ply.py
```

To obtain the fusibile:

- Check out the modified version fusibile `git clone https://github.com/YoYo000/fusibile`
- Install fusibile by `cmake .` and `make`, which will generate the executable at `FUSIBILE_EXE_PATH`

## Evaluation

We need to download the official STL point clouds for our evaluation. Please download the [STL Point Clouds](http://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip), which is the STL reference point clouds for all the scenes. And please download the [observability masks and evaluation codes](http://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip) from the SampleSet for evaluation. 