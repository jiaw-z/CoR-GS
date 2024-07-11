# CoR-GS
This is the official repository for our ECCV2024 paper **CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization**.

[Paper](https://arxiv.org/pdf/2405.12110) | [Project](https://jiaw-z.github.io/CoR-GS/) | [Video](https://youtu.be/O83v9Wrn3c4)

![method](assets/method.png)

## Abstract

3D Gaussian Splatting (3DGS) creates a radiance field consisting of 3D Gaussians to represent a scene. With sparse training views, 3DGS easily suffers from overfitting, negatively impacting rendering. This paper introduces a new co-regularization perspective for improving sparse-view 3DGS. When training two 3D Gaussian radiance fields, we observe that the two radiance fields exhibit point disagreement and rendering disagreement that can unsupervisedly predict reconstruction quality, stemming from the randomness of densification implementation. We further quantify the two disagreements and demonstrate the negative correlation between them and accurate reconstruction, which allows us to identify inaccurate reconstruction without accessing ground-truth information. Based on the study, we propose CoR-GS, which identifies and suppresses inaccurate reconstruction based on the two disagreements: (1) Co-pruning considers Gaussians that exhibit high point disagreement in inaccurate positions and prunes them. (2) Pseudo-view co-regularization considers pixels that exhibit high rendering disagreement are inaccurate and suppress the disagreement. Results on LLFF, Mip-NeRF360, DTU, and Blender demonstrate that CoR-GS effectively regularizes the scene geometry, reconstructs the compact representations, and achieves state-of-the-art novel view synthesis quality under sparse training views. 


## TODO
- [x] scripts for LLFF.

- [ ] scripts for Mipnerf360, DTU, Blender datasets.

## Installation

Tested on Ubuntu 18.04, CUDA 11.3, PyTorch 1.12.1

``````
conda env create --file environment.yml
conda activate corgs
``````



## Evaluation

### LLFF

1. Download LLFF from [the official download link](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).

2. Start training and testing:

   ```bash
   # for example
   bash scripts/run_llff.sh ${gpu_id} data/nerf_llff_data/fern output/llff/fern
   ```



## Customized Dataset
Similar to Gaussian Splatting, our method can read standard COLMAP format datasets. Please customize your sampling rule in `scenes/dataset_readers.py`, and see how to organize a COLMAP-format dataset from raw RGB images referring to our preprocessing of DTU.



## Citation

Consider citing as below if you find this repository helpful to your project:

```
@article{zhang2024cor,
      title={CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization},
      author={Zhang, Jiawei and Li, Jiahe and Yu, Xiaohan and Huang, Lei and Gu, Lin and Zheng, Jin and Bai, Xiao},
      journal={arXiv preprint arXiv:2405.12110},
      year={2024}
    }
```

## Acknowledgement

Special thanks to the following awesome projects!

- [Gaussian-Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [FSGS](https://github.com/VITA-Group/FSGS)
- [DNGaussian](https://github.com/Fictionarry/DNGaussian)