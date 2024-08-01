# Gaussian in the Dark 😈

This repository contains the official authors implementation associated with the paper "Gaussian in the Dark: Real-Time View Synthesis From Inconsistent Dark Images Using Gaussian Splatting", which has been accepted by **Pacific Graphics 2024** (journal track).

<img src="assets/teaser.png">

## Requirements

* Linux OS
* NVIDIA GPUs. We experimented on A6000 GPUs (cuda 11.8).
* Python libraries: see [environment.yml](./environment.yml). You can use the following commands with Anaconda3 to create and activate your virtual environment:
  - `git clone https://github.com/yec22/Gaussian-DK.git`
  - `cd Gaussian-DK`
  - `conda env create -f environment.yml`
  - `conda activate 3dgs_dk`

## Dataset

## Usage

First, please make sure that all requirements are satisfied and all required files are downloaded (see above steps).

```
bash run.sh
```

## Results

**Comparison with 3DGS**
<center class="half">
<img src="assets/light1.gif" style="width:47%"/>
<img src="assets/light2.gif">
</center>

**Light-Up Effect**
<center class="half">
    <img src="assets/light1.gif" width="200"/> <img src="assets/light2.gif" width="200"/>
</center>

## Acknowledgement

Code of this repo is rely on [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) and [HDR-NeRF](https://github.com/xhuangcv/hdr-nerf/). We thank the authors for their great job!