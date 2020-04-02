## Inference with pretrained models 

- [x] single GPU testing
- [x] multiple GPU testing
- [x] visualize segmentation results 

You can use the following commands to test a dataset.

```shell
# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--scales ${SCALES}] 

# multi-gpu testing
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--scales ${SCALES}] 
```

Examples:

Assume that you have already downloaded the checkpoints to the directory `checkpoints/`.

1. Test DDCN and visualize the results. 

```shell
./tools/dist_test.sh configs/nyuv2/danet_r50_depthdeform.py \
    danet_r50_nyuv2_depthdeform.pth \
    2
```

2. Train DDCN.  

Download [pretrained weights](https://hangzh.s3.amazonaws.com/encoding/models/fcn_resnet50_ade-662e979d.zip) originated from [here](https://github.com/zhanghang1989/PyTorch-Encoding/blob/master/encoding/models/model_store.py).

```shell
./tools/dist_train.sh configs/nyuv2/danet_r50_depthdeform.py \
    2
```
