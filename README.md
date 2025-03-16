<div align=center class="logo">
      <img src="Repo_image/DiffTSR_icon.png" style="width:640px">
   </a>
</div>

#
Diffusion-based Blind Text Image Super-Resolution (CVPR2024)
<a href='https://arxiv.org/abs/2312.08886'><img src='https://img.shields.io/badge/arXiv-2312.08886-b31b1b.svg'></a> &nbsp;&nbsp;

[Yuzhe Zhang](https://yuzhezhang-1999.github.io/)<sup>1</sup> | [Jiawei Zhang](https://scholar.google.com/citations?user=0GTpIAIAAAAJ)<sup>2</sup> | Hao Li<sup>2</sup> | [Zhouxia Wang](https://scholar.google.com/citations?user=JWds_bQAAAAJ)<sup>3</sup> | Luwei Hou<sup>2</sup> | [Dongqing Zou](https://scholar.google.com/citations?user=K1-PFhYAAAAJ)<sup>2</sup> | [Liheng Bian](https://scholar.google.com/citations?user=66IFMDEAAAAJ)<sup>1</sup>

<sup>1</sup>Beijing Institute of Technology, <sup>2</sup>SenseTime Research, <sup>3</sup>The University of Hong Kong

## 💬 Q&A
Please Read Before Trying～

### 🇨🇳 中文Q&A：对于大家关心的一些细节问题，这里给出了归纳供大家参考

1. Q: IDM中Unet用的是Stable-Diffusion的权重吗?

   A: 不是。IDM的Unet是从头训练的，没有加载任何预训练权重，IDM的结构也和任何一个Diffusion模型的Unet不一致。但是VAE是加载了ldm的f4的VAE在open-image上预训练的权重，然后在本项目的CTW-HQ-Train数据集上进行了微调，微调了100,000iter，batch_size=16。此外包括TDM和MoM在内的模型均未使用预训练模型，均为从头训练获得。详细训练设置请看 [附加材料](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Zhang_Diffusion-based_Blind_Text_CVPR_2024_supplemental.pdf)Section 1.4。

2. Q: DiffTSR模型的输入尺寸和要求，需要将输入resize吗？

   A: 模型的LR输入需要统一resize到 width=512/height=128；此外因为本项目仅考虑单行文本输入，您所输入的图片需要只包含一行文本，对于IDM和TDM都仅适配了单行文本，在多行本文输入会出现效果扭曲和错误的结果。

3. Q: 图片的推理速度非常慢，有什么解决办法吗？
   
   A: 因为该项目的技术基于Diffusion，所以每处理一张图像都需要进行T次迭代过程（T=200默认）。若想提升推理速度，可以考虑：
   ```
   （1）减小T，因为采样器为DDIM，在T=20时仍有较好表现；
   （2）对DiffTSR模型进行量化，可以参考Diffusion模型量化的相关Repo；
   （3）使用本项目的Baseline model，虽然Baseline会在一定程度上减小性能，但是可以提升约2倍的耗时收益，且对于大多数场景不会有明显效果退化;
   （4）对我的模型进行蒸馏；或者根据该论文训练一个更小的IDM模型，文字场景可能并不需要类似于通用场景图像生成这么重的模型。
   ```

4. Q: 在训练IDM的时候损失是怎么设置的，text_recognization loss是如何实现的？

   A: 训练IDM的时候，使用了两个损失函数（1）L2 loss用于预测噪声，（2）OCR loss用于从预测出的干净的X0上检测文字。
   ```
    (1) 其中 L2 loss 即传统diffsion模型中所用的用于最小化（Unet输出-noise map）,从而使Unet具备噪声估计能力；
    (2) 其中 OCR loss 为从z_t获得z^(t-1)后，再根据这个得到z^0，然后获得x^0=Decoder(z^0)，然后将x^0输入给权重冻结的TransOCR模型从而获得x^0上所包含的文字embedding，然后计算预测的pred-text-embedding和gt-text-embedding之间的cross-entropy loss，即为该OCR loss，且OCR loss添加了一个weight=0.02的约束。
   ```
   内容详见[issue](https://github.com/YuzheZhang-1999/DiffTSR/issues/13)。

5. Q: 训练的损失函数是什么？
  
   A: DiffTSR模型的训练经历了三个步骤，每个步骤用了不同的损失函数的组合。分别为：训练IDM、训练TDM、训练MoM。
   ```
    (1) 训练IDM，IDM从头训练Unet，该损失为L_IDM，包含L2 loss和OCR loss；
    (2) 训练TDM，TDM从头训练Transformer，该损失为L_TDM，参考[Multinomial DIffusion](https://arxiv.org/pdf/2102.05379)Section 4;
    (3) 训练整个DiffTSR，冻结IDM和TDM，仅训练MoM，该损失为L_MoM = L_IDM+L_TDM*weight;
   ```
   其中：
   
   $$
    L_{IDM} = L_2(\epsilon, \epsilon_{pred, t}) + L_{ocr}(Transocr(x^{pred}_0), TextEmbed_{gt})*\lambda, \lambda=0.02
   $$

   $$
    L_{TDM} = KL(\mathcal{C(\pi_{post}(\mathbf{c_t}, \mathbf{c_0}))} || \mathcal{C(\pi_{post}(\mathbf{c_t}, \mathbf{c_{pred, t}}))})
   $$

   $$
    L_{MoM} = L_{IDM} + L_{TDM}*\lambda, \lambda=1
   $$

   具体符号定义和原理推导详见[附加材料](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Zhang_Diffusion-based_Blind_Text_CVPR_2024_supplemental.pdf)Section1和Algorithm 1 DiffTSR Training。

    未完待续...

</details>

<details>
<summary>🇬🇧 English Version(Click to expand)</summary>

    Pending...

</details>


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
@inproceedings{zhang2024diffusion,
  title={Diffusion-based Blind Text Image Super-Resolution},
  author={Zhang, Yuzhe and Zhang, Jiawei and Li, Hao and Wang, Zhouxia and Hou, Luwei and Zou, Dongqing and Bian, Liheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={25827--25836},
  year={2024}
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

![visitors](https://visitor-badge.laobi.icu/badge?page_id=YuzheZhang-1999/DiffTSR)

</details>