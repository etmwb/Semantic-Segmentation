###Data preparation 
1. Download NYUv2 dataset from [here](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
2. Generate all images from the downloaded .mat file.

```shell 
python tools/convert_datasets/nyuv2.py PATH/nyu_depth_v2_labeled.mat
```

3. Generate train/test split file.

```
import scipy.io as scio
data = scio.loadmat('splits.mat')['trainNdxs'] # or ['testNdxs']
with open('train.txt', 'w') as f: 
    for i in range(len(data)): 
        f.write(str(data[i][0]-1)+'.png\n')
```
