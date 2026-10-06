<!-- <div align="center">
DreamVLA: Vision-Language-Action Models Dream Comprehensive World Knowledge -->
<!-- </div> -->




<h3 align="center" style="font-size:48px; font-weight:bold; color:#9C276A; margin: 0;">
  <a href="https://arxiv.org/abs/2507.04447" style="color:#9C276A; text-decoration: none;">
    DreamVLA: A Vision-Language-Action Model <br> Dreamed with Comprehensive World Knowledge
  </a>
</h3>
<p align="center">
 NeurIPS 2025
</p>
<p align="center">
  ⭐ If our project helps you, please give us a star on GitHub to support us!
</p>

<div align="center">

<!-- <p align="center">
  <a href="https://zhangwenyao1.github.io/">Wenyao Zhang</a>*,
  <a href="https://ericliuhhh.github.io/">Hongsi Liu</a>*,
  <a href="https://qizekun.github.io/">Zekun Qi</a>*,
  <a href="https://wangyunnan.github.io/">Yunnan Wang</a>*,
  <a href="#">Xinqiang Yu</a>,
  <a href="https://jzhzhang.github.io/">Jiazhao Zhang</a>,
  <a href="https://runpeidong.web.illinois.edu/">Runpei Dong</a>,
  <a href="https://jiaweihe.com/">Jiawei He</a>,<br>
  <a href="https://scholar.google.com/citations?user=X7M0I8kAAAAJ&hl=en">Zhizheng Zhang</a>,
  <a href="https://hughw19.github.io/">He Wang</a>,
  <a href="https://ericyi.github.io/">Li Yi</a>,
  <a href="https://www.eitech.edu.cn/?p=leader-Wenjun%20Zeng&tid=19&lang=en">Wenjun Zeng</a>,
  <a href="http://home.ustc.edu.cn/~jinxustc/">Xin Jin</a>
</p>
<!-- </div> -->
<p>
  <a href="https://arxiv.org/abs/2507.04447">
    <img src="https://img.shields.io/badge/Paper-PDF-orange.svg" alt="Paper PDF">
  </a>
  <a href="https://zhangwenyao1.github.io/DreamVLA">
    <img src="https://img.shields.io/badge/Project-Page-Green.svg" alt="Project Page">
  </a>
  <a href="https://huggingface.co/WenyaoZhang/DreamVLA">
    <img src="https://img.shields.io/badge/🤗-Hugging_Face-yellow.svg" alt="Hugging Face">
  </a>
  <a href="https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg" alt="Code License">
  </a>
  <a href="https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE">
    <img src="https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg" alt="Data License">
  </a>
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dreamvla-a-vision-language-action-model/robot-manipulation-on-calvin)](https://paperswithcode.com/sota/robot-manipulation-on-calvin?p=dreamvla-a-vision-language-action-model)
</p>

<p align="center">
If you have any questions about the code, feel free to open an issue!
</p>

</div>


## The difference from previous works
<div style="text-align: center;">
    <img src="assets/paradigm_compare.gif" width=100% >
</div>

## Overall framework of DreamVLA
<div style="text-align: center;">
    <img src="assets/pipeline.gif" width=100% >
</div>


## Clone this repo

```
git clone https://github.com/Zhangwenyao1/DreamVLA
```

This repository's code is based on the [Seer](https://github.com/OpenRobotLab/Seer/tree/main).


## Running on the Benchmark

### Decision-Influential World Abstraction

A complete DIWA policy path is available behind `--use_diwa`, including
object-centric future abstraction, counterfactual influence, measured-Q
regret geometry, adaptive sparse imagination and evaluation metrics. See
[docs/DIWA_IMPLEMENTATION.md](docs/DIWA_IMPLEMENTATION.md) for the measured
data contract, training, evaluation and verification commands.
Cross-simulator smoke tests for CALVIN, RoboCasa and RoboTwin, plus per-source
training/inference smokes for DROID and all 12 configured OXE datasets, are documented in
[scripts/MULTI_DATASET/DIWA/README.md](scripts/MULTI_DATASET/DIWA/README.md).

#### CALVIN ABC-D <a name="calvin abc-d"></a>
- [Installation](docs/CALVIN_ABC-D_INSTALL.md)
- [Running Code](docs/CALVIN_ABC-D_RUN.md)
- CALVIN Result

  | Method                  |        1 |        2 |        3 |        4 |        5 | Avg. Len. ↑ |
  | :---------------------- | -------: | -------: | -------: | -------: | -------: | ----------: |
  | Roboflamingo \[30]      |     82.4 |     61.9 |     46.6 |     33.1 |     23.1 |        2.47 |
  | Susie \[118]            |     87.0 |     69.0 |     49.0 |     38.0 |     26.0 |        2.69 |
  | GR-1 \[14]              |     85.4 |     71.2 |     59.6 |     49.7 |     40.1 |        3.06 |
  | 3D Diffusor Actor \[93] |     92.2 |     78.7 |     63.9 |     51.2 |     41.3 |        3.27 |
  | OpenVLA \[1]            |     91.3 |     77.8 |     62.0 |     52.1 |     43.5 |        3.27 |
  | RoboDual \[119]         |     94.4 |     82.7 |     72.1 |     62.4 |     54.4 |        3.66 |
  | UNIVLA \[120]           |     95.5 |     85.8 |     75.4 |     66.9 |     56.5 |        3.80 |
  | Pi0 \[32]               |     93.8 |     85.0 |     76.7 |     68.1 |     59.9 |        3.84 |
  | CLOVER \[121]           |     96.0 |     83.5 |     70.8 |     57.5 |     45.4 |        3.53 |
  | UP-VLA \[57]            |     92.8 |     86.5 |     81.5 |     76.9 |     69.9 |        4.08 |
  | Robovlm \[37]           |     98.0 |     93.6 |     85.4 |     77.8 |     70.4 |        4.25 |
  | Seer \[56]              |     96.3 |     91.6 |     86.1 |     80.3 |     74.0 |        4.28 |
  | VPP \[49]               |     95.7 |     91.2 |     86.3 |     81.0 |     75.0 |        4.29 |
  | **DreamVLA (Ours)**     | **98.2** | **94.6** | **89.5** | **83.4** | **78.1** |    **4.44** |


#### LIBERO <a name="libero"></a>
- [Installation](docs/LIBERO_INSTALL.md)
- [Running Code](docs/LIBERO_RUN.md)
- LIBERO Result
  | Methods                | LIBERO-Spatial | LIBERO-OBJECT | LIBERO-GOAL | LIBERO-LONG |  Average |
  | :--------------------- | -------------: | ------------: | ----------: | ----------: | -------: |
  | Diffusion Policy \[72] |           78.3 |          92.5 |        68.3 |        50.5 |     72.4 |
  | Octo \[9]              |           78.9 |          85.7 |        84.6 |        51.1 |     75.1 |
  | OpenVLA \[1]           |           84.7 |          88.4 |        79.2 |        53.7 |     76.5 |
  | SpatialVLA \[31]       |           88.2 |          89.9 |        78.6 |        55.5 |     78.1 |
  | **DreamVLA (Ours)**    |       **97.5** |      **94.0** |    **89.5** |    **89.5** | **92.6** |



## TODO
- [x] Release the code with LIBERO 



## Acknowledgement

We would like to express our deepest gratitude to [Yang Tian](https://scholar.google.com/citations?user=leXXHKwAAAAJ&hl=zh-CN) for the technique support!!!

## Citation

If you find our ideas / environments helpful, please cite our work at

```
article{dreamvla25,
          author = {Wenyao Zhang and
                    Hongsi Liu and
                    Zekun Qi and
                    Yunan Wang and
                    Xinqiang Yu and
                    Jiazhao Zhang and
                    Runpei Dong and
                    Jiawei He and
                    He Wang and
                    Zhizheng Zhang and
                    Li Yi and 
                    Wenjun Zeng and
                    Xin Jin},
          title        = {DreamVLA: A Vision-Language-Action Model Dreamed with Comprehensive World Knowledge},
          journal      = {CoRR},
          volume       = {abs/2507.04447},
          year         = {2025},
          url          = {https://doi.org/10.48550/arXiv.2507.04447},
          doi          = {10.48550/ARXIV.2507.04447},
          eprinttype    = {arXiv},
          eprint       = {2507.04447}
        }
```
