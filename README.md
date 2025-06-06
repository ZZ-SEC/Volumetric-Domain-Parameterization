#  Volumetric Domain Parameterization

## Description

This is the source code for the paper:

"Integral Parameterization of Volumetric Domains via Deep Neural Networks"

by Zheng Zhan, Wenping Wang, and Falai Chen*

Computer Methods in Applied Mechanics and Engineering, 2025

## Note

In the paper, we utilized the algorithm from <a href="https://github.com/kuiwuchn/3x3_SVD_CUDA">"Fast CUDA 3x3 SVD"</a> to accelerate singular value decomposition (SVD). 
However, this algorithm requires additional compilation for different CUDA versions and python versions.
Therefore, in this code, we use "torch.linalg.svdvals" instead, which is slightly slower but easy to use. 

## BibTex 

Please cite the following paper if it helps. 

```
@article{zhanIntegralParameterizationVolumetric2025,
  title = {Integral Parameterization of Volumetric Domains via Deep Neural Networks},
  author = {Zhan, Zheng and Wang, Wenping and Chen, Falai},
  year = {2025},
  month = jun,
  journal = {Computer Methods in Applied Mechanics and Engineering},
  volume = {441},
  pages = {117988},
  issn = {00457825},
  doi = {10.1016/j.cma.2025.117988}
}
```
