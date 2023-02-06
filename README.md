## MoCo v3 for Self-supervised ResNet and ViT

### Introduction
This is a PyTorch implementation of [DILEMMA](https//arxiv.org/abs/2204.04788) for self-supervised ViT.

This is the reimplementation of the original work with minimal changes to the official [pytorch implementation of MoCov3](https://github.com/facebookresearch/moco-v3). 

### Usage: Preparation

Install PyTorch and download the ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet). Similar to [MoCo v1/2](https://github.com/facebookresearch/moco), this repo contains minimal modifications on the official PyTorch ImageNet code. We assume the user can successfully run the official PyTorch ImageNet code.
For ViT models, install [timm](https://github.com/rwightman/pytorch-image-models).

### Usage: Self-supervised Pre-Training

#### ViT-Small with 1-node (8-GPU) training, batch 1024

```
python main_moco.py \
  -a vit_small -b 1024 \
  --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
  --epochs=300 --warmup-epochs=40 \
  --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
  --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders]
```

#### Notes:
1. To enable sparsity you should set `--token_drop_rate` to a non-zero value, for example 0.75 keeps 25% of the tokens, general rule of thumb is that bigger models can have larger sparsities.
2. To enable DILEMMA loss, you can set `--dilemma_probability` to a non-zero value, it seems that 0.2 is always good.
3. The batch size specified by `-b` is the total batch size across all GPUs.
4. The learning rate specified by `--lr` is the *base* lr, and is adjusted by the [linear lr scaling rule](https://arxiv.org/abs/1706.02677) in [this line](https://github.com/facebookresearch/moco-v3/blob/main/main_moco.py#L213).
5. Using a smaller batch size has a more stable result (see paper), but has lower speed. Using a large batch size is critical for good speed in TPUs (as we did in the paper).
6. In this repo, only *multi-gpu*, *DistributedDataParallel* training is supported; single-gpu or DataParallel training is not supported. This code is improved to better suit the *multi-node* setting, and by default uses automatic *mixed-precision* for pre-training.

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

### Citation
```
@article{Sameni2022DILEMMASS,
  title={DILEMMA: Self-Supervised Shape and Texture Learning with Transformers},
  author={Sepehr Sameni and Simon Jenni and Paolo Favaro},
  journal={ArXiv},
  year={2022},
  volume={abs/2204.04788}
}
```
