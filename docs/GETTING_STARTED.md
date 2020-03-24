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

1. Test Faster R-CNN and visualize the results. Press any key for the next image.

```shell
./tools/dist_test.sh configs/nyuv2/danet_r50_depthdeform.py \
    danet_r50_nyuv2_depthdeform.pth \
    2
```
