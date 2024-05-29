<div align=center class="logo">
      <img src="Repo_image/DiffTSR_icon.png" style="width:640px">
   </a>
</div>

#
Diffusion-based Blind Text Image Super-Resolution (CVPR2024)
<a href='https://arxiv.org/abs/2312.08886'><img src='https://img.shields.io/badge/arXiv-2312.08886-b31b1b.svg'></a> &nbsp;&nbsp;

[Yuzhe Zhang](https://yuzhezhang-1999.github.io/)<sup>1</sup> | [Jiawei Zhang](https://scholar.google.com/citations?user=0GTpIAIAAAAJ)<sup>2</sup> | Hao Li<sup>2</sup> | [Zhouxia Wang](https://scholar.google.com/citations?user=JWds_bQAAAAJ)<sup>3</sup> | Luwei Hou<sup>2</sup> | [Dongqing Zou](https://scholar.google.com/citations?user=K1-PFhYAAAAJ)<sup>2</sup> | [Liheng Bian](https://scholar.google.com/citations?user=66IFMDEAAAAJ)<sup>1</sup>

<sup>1</sup>Beijing Institute of Technology, <sup>2</sup>SenseTime Research, <sup>3</sup>The University of Hong Kong



## 📢 News
- **2024.05** 🚀Inference code has been released, enjoy.
- **2024.04** 🚀Official repository of DiffTSR.
- **2024.03** 🌟The implementation code will be released shortly.
- **2024.03** ❤️Accepted by CVPR2024.

## 🔥 TODO
- [x] Attach the detailed implementation and supplementary material.
- [x] Add inference code and checkpoints for blind text image SR.
- [ ] Add training code and scripts.

## 👁️ Gallery

[<img src="Repo_image/ImgSli_1.jpg" width="256px"/>](https://imgsli.com/MjY0MTk5) [<img src="Repo_image/ImgSli_2.jpg" width="256px"/>](https://imgsli.com/MjY0MjA0)

## 🛠️ Try
### Dependencies and Installation

- Pytorch >= 1.7.0
- CUDA >= 11.0
```
# git clone this repository
git clone https://github.com/YuzheZhang-1999/DiffTSR
cd DiffTSR

# create new anaconda env
conda env create -f environment.yaml
conda activate DiffTSR
```
### Download the checkpoint
Please download the checkpoint file from the URL below to the ./ckpt/ folder.

- [[GoogleDrive](https://drive.google.com/drive/folders/1K6k5ZcvF3w-1MDN_gXQTdsLgFZ2SM8qy?usp=drive_link)] 

- [[BaiduDisk](https://pan.baidu.com/s/1hfaQzIp_V6H8AhAq5dfr8A)] [Password: vk9n] 




### Inference
```
python inference_DiffTSR.py
# check the code for more detail
```

## 🔎 Overview of DiffTSR
![DiffTSR](Repo_image/paper-DiffTSR-model.jpg)
### Abstract
Recovering degraded low-resolution text images is challenging, especially for Chinese text images with complex strokes and severe degradation in real-world scenarios.
Ensuring both text fidelity and style realness is crucial for high-quality text image super-resolution.
Recently, diffusion models have achieved great success in natural image synthesis and restoration due to their powerful data distribution modeling abilities and data generation capabilities
In this work, we propose an Image Diffusion Model (IDM) to restore text images with realistic styles.
For diffusion models, they are not only suitable for modeling realistic image distribution but also appropriate for learning text distribution.
Since text prior is important to guarantee the correctness of the restored text structure according to existing arts, we also propose a Text Diffusion Model (TDM) for text recognition which can guide IDM to generate text images with correct structures.
We further propose a Mixture of  Multi-modality module (MoM) to make these two diffusion models cooperate with each other in all the diffusion steps.
Extensive experiments on synthetic and real-world datasets demonstrate that our Diffusion-based Blind Text Image Super-Resolution (DiffTSR) can restore text images with more accurate text structures as well as more realistic appearances simultaneously.

### Visual performance comparison overview 
![DiffTSR](Repo_image/paper-fig1.jpg)
Blind text image super-resolution results between different methods on synthetic and real-world text images. Our method can restore text images with high text fidelity and style realness under complex strokes, severe degradation, and various text styles.


<details>
  <summary>📷 More Visual Results</summary>

  ## ![DiffTSR](Repo_image/paper-visual-comp-1.jpg)
  ## ![DiffTSR](Repo_image/paper-visual-comp-2.jpg)
  ## ![DiffTSR](Repo_image/paper-visual-comp-3.jpg)
  ## ![DiffTSR](Repo_image/paper-visual-comp-4.jpg)
  ## ![DiffTSR](Repo_image/paper-visual-comp-5.jpg)
  ## ![DiffTSR](Repo_image/paper-visual-comp-6.jpg)

</details>


## 🎓Citations
```
@article{zhang2023diffusion,
  title={Diffusion-based Blind Text Image Super-Resolution},
  author={Zhang, Yuzhe and Zhang, Jiawei and Li, Hao and Wang, Zhouxia and Hou, Luwei and Zou, Dongqing and Bian, Liheng},
  journal={arXiv preprint arXiv:2312.08886},
  year={2023}
}
```

## 🎫 License
This project is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement
Thanks to these awesome work：
- [Latent Diffusion](https://github.com/CompVis/latent-diffusion)
- [Benchmarking Chinese Text Recognition](https://github.com/FudanVI/benchmarking-chinese-text-recognition)

<details>
<summary>Statistics</summary>

![visitors](https://visitor-badge.laobi.icu/badge?page_id=YuzheZhang-1999.DiffTSR)

</details>