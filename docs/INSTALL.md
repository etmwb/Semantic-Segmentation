### Data preparation 
a. Download NYUv2 dataset from [here](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
b. Generate all images from the downloaded .mat file.

```shell 
python tools/convert_datasets/nyuv2.py PATH/nyu_depth_v2_labeled.mat
```

c. Generate train/test split file.

### Install Detseg 
Following the install.md in mmdetection.
