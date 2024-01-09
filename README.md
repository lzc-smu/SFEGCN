# CylinGCN: Cylindrical Structures Segmentation in 3D Biomedical Optical Imaging by a contour-based Graph Convolutional Network

![city](assets/SFEGCN.bmp)

> [**Organ-level instance segmentation enables continuous time-space-spectrum analysis of photoacoustic tomography images**](https://arxiv.org/pdf/)  
> Zhichao Liang, Shuangyang Zhang, Zongxin Mo, Xiaoming Zhang, Anqi Wei, Wufan Chen, Li Qi

Any questions or discussions are welcomed!

## Installation

Please refer to [INSTALL.md](assets/INSTALL.md) for installation instructions.

## Testing
Test:
```
python test.py --demo data../JPEGImages/ --load_model exp/checkpoints.../model_199.pth --arch dlagcnmulti_34 --dataset pat --output_imgs
```
    
## Training

Train:
```
python train.py --task space --input_mode space --dataset pat --arch dlagcnmulti_34
```

## Acknowledgement
Our work benefits a lot from [CenterTrack](https://github.com/xingyizhou/CenterTrack#tracking-objects-as-points) 
and [DeepSnake](https://github.com/zju3dv/snake). Thanks for their great contributions.


## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@article{Liang2024,
  title={Organ-level instance segmentation enables continuous time-space-spectrum analysis of photoacoustic tomography images},
  author = {Zhichao Liang, Shuangyang Zhang, Zongxin Mo, Xiaoming Zhang, Anqi Wei, Wufan Chen, Li Qi},
  journal = {},
  volume = {},
  pages = {},
  year = {},
  issn = {}  
}
```