# Local Aggregation for Unsupervised Learning of Visual Embeddings

This is a Pytorch re-implementation of the Local Aggregation (LA) algorithm ([Paper](https://arxiv.org/abs/1903.12355)).
The Tensorflow version can be found [here](https://github.com/neuroailab/LocalAggregation), which is implemented by the paper author.

**Note:** ~~This implementation is still under testing, although it's almost validated.~~ This implementation has been validated!

# Usage

### Prerequisites

* Ubuntu 16.04
* Pytorch 1.2.0
* [Faiss==1.6.1](https://github.com/facebookresearch/faiss)
* tqdm
* dotmap
* tensorboardX

### Runtime Setup
```
source init_env.sh
```

### Model training

This implementation currently supports LA trained ResNets. We have tested this implementation for ResNet-18. 
As LA algorithm requires training the model using IR algorithm for 10 epochs as a warm start, we first run the IR training using the following command:
```
CUDA_VISIBLE_DEVICES=0 python scripts/instance.py ./config/imagenet_ir.json
```
Then specify `instance_exp_dir` in `./config/imagenet_la.json` and run the following command to do the LA training:
```
CUDA_VISIBLE_DEVICES=0 python scripts/localagg.py ./config/imagenet_la.json
```
By default, both IR and LA are trained using a single GPU. Multi-gpu training is also supported in this implementation.


### Transfer learning 
After finishing the LA training, run the following command to do the transfer learning to ImageNet:
```
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py ./config/imagenet_ft.json
```
