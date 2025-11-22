# <p align="center"><font color="orange">VIPerson</font>: Flexibly Generating Virtual Identity for Person Re-Identification</p>

<p align="center">
    <a href="https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_VIPerson_Flexibly_Generating_Virtual_Identity_for_Person_Re-Identification_ICCV_2025_paper.html"><img src="https://img.shields.io/badge/ICCV-2025-blue"></a>
</p>

This is the official PyTorch implementation for our ICCV 2025 paper: **"VIPerson: Flexibly Generating Virtual Identity for Person Re-Identification"**. We propose a novel diffusion-based pipeline to synthesize realistic and diverse pedestrian images for Person Re-identification (ReID).

If you find our work useful, please consider giving a `star` ⭐️!

---

## 📢 News

* **[Nov 2025]** The generated VIPerson and pre-trained ReID models are now available. Enjoy it! 🚀
* **[June 2025]** 🎉 Our paper has been accepted by ICCV 2025!

## 📝 Abstract

Person re-identification (ReID) is to match the person images under different camera views. Training ReID models necessitates a substantial amount of labeled real-world data, leading to high labeling costs and privacy issues. Although several ReID data synthetic methods are proposed to address these issues, they fail to generate images with new identities or real-world camera style.
In this paper, we propose a novel pedestrian generation pipeline, **VIPerson**, to generate camera-realistic pedestrian images with flexible Virtual Identities for the Person ReID task. VIPerson focuses on three key factors in data synthesis:

* **🚶‍♀️ (I) Virtual identity diversity**: Enhancing the latent diffusion model with our proposed **dropout text embedding**, we flexibly generate random and hard identities.
* **📸 (II) Scalable cross-camera variations**: VIPerson introduces scalable variations of scenes and poses within each identity.
* **🎨 (III) Camera-realistic style**: Adopting an identity-agnostic approach to transfer realistic styles, we avoid privacy exposure of real identities.

Extensive experimental results across a broad range of downstream ReID tasks demonstrate the superiority of our generated dataset over existing methods. In addition, VIPerson can be adapted to the privacy-constrained ReID scenario, which widens the application of our pipeline.


## 💾 Dataset

We generated a virtual pedestrian dataset for ReID using the VIPerson pipeline. You can download it from the following links:

* **VIPerson-Dataset**: [Google Drive](https://your_google_drive_link_here) | [Baidu Cloud](https://pan.baidu.com/s/1dZfu7lPT0Iiu0uKrDtgb2w) (Access Code: `aeft`)

The format of VIPerson dataset please refer to [VIPerson.json](https://pan.baidu.com/s/1RrbwimdasYzhQPD_4BWSlw)(Access Code: `6cxf`)


## 📦 Pre-trained Models

We provide the model weights for easy reproduction and future research.

| Model               | Download Link                                                                                           |
| :------------------ | :------------------------------------------------------------------------------------------------------ |
| **VIPerson checkpoint** | [Google Drive](https://your_google_drive_link_here) / [Baidu Cloud](https://pan.baidu.com/s/1w4CXDxwocNGuXRE65NinkQ)(Access Code: `qtxy`) |


## ✅ TODO List

-   [x] Release pre-trained models.
-   [x] Release the **VIPerson** dataset.
-   [ ] Release inference code and identity generator. 
-   [ ] Add training scripts for more downstream ReID models.

## Citing VIPerson

If you use our code or dataset in your research, please consider citing our paper:

```
@inproceedings{zhang2025viperson,
  title={VIPerson: Flexibly Generating Virtual Identity for Person Re-Identification},
  author={Zhang, Xiao-Wen and Zhang, Delong and Peng, Yi-Xing and Ouyang, Zhi and Meng, Jingke and Zheng, Wei-Shi},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={23374--23384},
  year={2025}
}
```

## Acknowledgements

Our implementation references the following outstanding projects. We thank them for their contributions to the open-source community.

* [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
* [ControlNet](https://github.com/lllyasviel/ControlNet)
* [StyleID](https://github.com/jiwoogit/StyleID)
