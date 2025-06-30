# <p align="center"><font color="orange">VIPerson</font>: Flexibly Generating Virtual Identity for Person Re-Identification</p>

<p align="center">
    <a href="https://arxiv.org/abs/YOUR_ARXIV_LINK_HERE"><img src="https://img.shields.io/badge/ICCV-2025-blue"></a>
    <a href="https://github.com/iSEELaboratory/VIPerson"><img src="https://img.shields.io/badge/Code-GitHub-green"></a>
    <a href="https://github.com/iSEELaboratory/VIPerson#dataset"><img src="https://img.shields.io/badge/Dataset-Download-orange"></a>
</p>

This is the official PyTorch implementation for our ICCV 2025 paper: **"VIPerson: Flexibly Generating Virtual Identity for Person Re-Identification"**. We propose a novel diffusion-based pipeline to synthesize realistic and diverse pedestrian images for Person Re-identification (ReID).

If you find our work useful, please consider giving a `star` ⭐️ and `forking` 🍴!

---

## 📢 News

* **[June 2025]** The generated VIPerson and pre-trained ReID models are now available. Enjoy it! 🚀
* **[June 2025]** 🎉 Our paper has been accepted by ICCV 2025!

## 📝 Abstract

Person re-identification (ReID) is to match the person images under different camera views. Training ReID models necessitates a substantial amount of labeled real-world data, leading to high labeling costs and privacy issues. Although several ReID data synthetic methods are proposed to address these issues, they fail to generate images with new identities or real-world camera style.

In this paper, we propose a novel pedestrian generation pipeline, **VIPerson**, to generate camera-realistic pedestrian images with flexible Virtual Identities for the Person ReID task. VIPerson focuses on three key factors in data synthesis:

* **🚶‍♀️ (I) Virtual identity diversity**: Enhancing the latent diffusion model with our proposed **dropout text embedding**, we flexibly generate random and hard identities.
* **📸 (II) Scalable cross-camera variations**: VIPerson introduces scalable variations of scenes and poses within each identity.
* **🎨 (III) Camera-realistic style**: Adopting an identity-agnostic approach to transfer realistic styles, we avoid privacy exposure of real identities.

Extensive experimental results across a broad range of downstream ReID tasks demonstrate the superiority of our generated dataset over existing methods. In addition, VIPerson can be adapted to the privacy-constrained ReID scenario, which widens the application of our pipeline.

## ✨ Visualizations

Samples from our generated **VIPerson** dataset:

* **Diverse Virtual Identities**
    ![Diverse Identities](https://via.placeholder.com/800x200.png?text=Showcase+of+Diverse+Virtual+Identities)

* **Rich Cross-Camera Variations (Pose, Scene)**
    ![Cross-Camera Variations](https://via.placeholder.com/800x200.png?text=Showcase+of+Cross-Camera+Variations)

* **Realistic Camera Styles**
    ![Realistic Styles](https://via.placeholder.com/800x200.png?text=Showcase+of+Realistic+Camera+Styles)

## 💾 Dataset

We generated a large-scale pedestrian dataset for ReID using the VIPerson pipeline. You can download it from the following links:

* **VIPerson-Dataset**: [Google Drive](https://your_google_drive_link_here) | [Baidu Cloud](https://your_baidu_cloud_link_here) (Access Code: `xxxx`)

The dataset follows the standard ReID format:
```text
VIPerson-Dataset/
└── train/
    ├── 0001/
    │   ├── 0001_c1s1_001051_00.jpg
    │   ├── 0001_c2s2_002061_00.jpg
    │   └── ...
    ├── 0002/
    │   ├── 0002_c1s1_001052_00.jpg
    │   └── ...
    └── ...
```

## 🛠️ Setup and Usage

### Installation

1.  Clone this repository:
    ```bash
    git clone [https://github.com/iSEELaboratory/VIPerson.git](https://github.com/iSEELaboratory/VIPerson.git)
    cd VIPerson
    ```

2.  Create and activate a Conda environment:
    ```bash
    conda create -n viperson python=3.9 -y
    conda activate viperson
    ```

3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Inference

To generate new pedestrian images using our pre-trained models:

```bash
# Example inference command
python inference.py --config configs/your_config_file.yaml \
                    --weights /path/to/your/weights.pth \
                    --num_identities 100 \
                    --num_images_per_id 10
```

## 📦 Pre-trained Models

We provide the model weights for easy reproduction and future research.

| Model               | Download Link                                                                                           |
| :------------------ | :------------------------------------------------------------------------------------------------------ |
| **VIPerson Base** | [Google Drive](https://your_google_drive_link_here) / [Baidu Cloud](https://your_baidu_cloud_link_here) |
| *More models here...* | *Links...* |

Please place the downloaded weights into the `checkpoints` directory.

## ✅ TODO List

-   [x] Release pre-trained models.
-   [x] Release the **VIPerson** dataset.
-   [ ] Release the **VIPerson\*** dataset.
-   [ ] Release inference code and identity generator. 
-   [ ] Add training scripts for more downstream ReID models.

## Citing VIPerson

If you use our code or dataset in your research, please consider citing our paper:

```
```

## Acknowledgements

Our implementation references the following outstanding projects. We thank them for their contributions to the open-source community.

* [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
* [ControlNet](https://github.com/lllyasviel/ControlNet)
* ...

## Contact

For any questions, please feel free to open a GitHub Issue or contact us directly at: `your_email@example.com`
