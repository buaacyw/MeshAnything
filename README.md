<p align="center">
  <h3 align="center"><strong>MeshAnything:<br> Artist-Created Mesh Generation<br> with Autoregressive Transformers</strong></h3>

<p align="center">
    <a href="https://buaacyw.github.io/">Yiwen Chen</a><sup>1,2*</sup>,
    <a href="https://tonghe90.github.io/">Tong He</a><sup>2†</sup>,
    <a href="https://dihuang.me/">Di Huang</a><sup>2</sup>,
    <a href="https://ywcmaike.github.io/">Weicai Ye</a><sup>2</sup>,
    <a href="https://ch3cook-fdu.github.io/">Sijin Chen</a><sup>3</sup>,
    <a href="https://me.kiui.moe/">Jiaxiang Tang</a><sup>4</sup><br>
    <a href="https://chenxin.tech/">Xin Chen</a><sup>5</sup>,
    <a href="https://caizhongang.github.io/">Zhongang Cai</a><sup>6</sup>,
    <a href="https://scholar.google.com.hk/citations?user=jZH2IPYAAAAJ&hl=en">Lei Yang</a><sup>6</sup>,
    <a href="https://www.skicyyu.org/">Gang Yu</a><sup>7</sup>,
    <a href="https://guosheng.github.io/">Guosheng Lin</a><sup>1†</sup>,
    <a href="https://icoz69.github.io/">Chi Zhang</a><sup>8†</sup>
    <br>
    <sup>*</sup>Work done during a research internship at Shanghai AI Lab.
    <br>
    <sup>†</sup>Corresponding authors.
    <br>
    <sup>1</sup>S-Lab, Nanyang Technological University,
    <sup>2</sup>Shanghai AI Lab,
    <br>
    <sup>3</sup>Fudan University,
    <sup>4</sup>Peking University,
    <sup>5</sup>University of Chinese Academy of Sciences,
    <br>
    <sup>6</sup>SenseTime Research,
    <sup>7</sup>Stepfun,
    <sup>8</sup>Westlake University
</p>


<div align="center">

<a href='https://arxiv.org/abs/2406.10163'><img src='https://img.shields.io/badge/arXiv-2406.10163-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://buaacyw.github.io/mesh-anything/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://github.com/buaacyw/MeshAnything/blob/master/LICENSE.txt'><img src='https://img.shields.io/badge/License-SLab-blue'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://huggingface.co/Yiwen-ntu/MeshAnything/tree/main"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Weights-HF-orange"></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://huggingface.co/spaces/Yiwen-ntu/MeshAnything"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Gradio%20Demo-HF-orange"></a>

</div>


<p align="center">
    <img src="demo/demo_video.gif" alt="Demo GIF" width="512px" />
</p>


## Release
- [6/17] 🔥🔥 We released the 350m version of **MeshAnything**.

## Contents
- [Release](#release)
- [Contents](#contents)
- [Installation](#installation)
- [Usage](#usage)
- [Important Notes](#important-notes)
- [TODO](#todo)
- [Acknowledgement](#acknowledgement)
- [Star History](#star-history)
- [BibTeX](#bibtex)

## Installation
Our environment has been tested on Ubuntu 22, CUDA 11.8 with A100, A800 and A6000.
1. Clone our repo and create conda environment
```
git clone https://github.com/buaacyw/MeshAnything.git && cd MeshAnything
conda create -n MeshAnything python==3.10.13
conda activate MeshAnything
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Usage
### Local Gradio Demo <a href='https://github.com/gradio-app/gradio'><img src='https://img.shields.io/github/stars/gradio-app/gradio'></a>
```
python app.py
```

### Mesh Command line inference
```
# folder input
python main.py --input_dir examples --out_dir mesh_output --input_type mesh

# single file input
python main.py --input_path examples/wand.ply --out_dir mesh_output --input_type mesh

# Preprocess with Marching Cubes first
python main.py --input_dir examples --out_dir mesh_output --input_type mesh --mc
```
### Point Cloud Command line inference
```
# Note: if you want to use your own point cloud, please make sure the normal is included.
# The file format should be a .npy file with shape (N, 6), where N is the number of points. The first 3 columns are the coordinates, and the last 3 columns are the normal.

# inference for folder
python main.py --input_dir pc_examples --out_dir pc_output --input_type pc_normal

# inference for single file
python main.py --input_dir pc_examples/mouse.npy --out_dir pc_output --input_type pc_normal
```

## Important Notes
- It takes about 7GB and 30s to generate a mesh on an A6000 GPU.
- The input mesh will be normalized to a unit bounding box. The up vector of the input mesh should be +Y for better results.
- Limited by computational resources, MeshAnything is trained on meshes with fewer than 800 faces and cannot generate meshes with more than 800 faces. The shape of the input mesh should be sharp enough; otherwise, it will be challenging to represent it with only 800 faces. Thus, feed-forward 3D generation methods may often produce bad results due to insufficient shape quality. We suggest using results from 3D reconstruction, scanning and SDS-based method (like [DreamCraft3D](https://github.com/deepseek-ai/DreamCraft3D)) as the input of MeshAnything.
- Please refer to https://huggingface.co/spaces/Yiwen-ntu/MeshAnything/tree/main/examples for more examples.
## TODO

The repo is still being under construction, thanks for your patience. 
- [ ] Release of training code.
- [ ] Release of larger model.

## Acknowledgement

Our code is based on these wonderful repos:

* [MeshGPT](https://nihalsid.github.io/mesh-gpt/)
* [meshgpt-pytorch](https://github.com/lucidrains/meshgpt-pytorch)
* [Michelangelo](https://github.com/NeuralCarver/Michelangelo)
* [transformers](https://github.com/huggingface/transformers)
* [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=buaacyw/MeshAnything&type=Date)](https://star-history.com/#buaacyw/MeshAnything&Date)

## BibTeX
```
@misc{chen2024meshanything,
  title={MeshAnything: Artist-Created Mesh Generation with Autoregressive Transformers},
  author={Yiwen Chen and Tong He and Di Huang and Weicai Ye and Sijin Chen and Jiaxiang Tang and Xin Chen and Zhongang Cai and Lei Yang and Gang Yu and Guosheng Lin and Chi Zhang},
  year={2024},
  eprint={2406.10163},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
