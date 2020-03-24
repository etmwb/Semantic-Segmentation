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
