# README
## Overview
This document describes the data preparation and model execution workflow for identity generator.

## Download and Place Required Weights

Before extracting embeddings, training, or generating identities, download the following files from Google Drive. These weights are not included in the Git repository. Keep the filenames exactly as listed and place them at the following paths, relative to the `ID_generator/` directory. If files with these names already exist, replace them with the downloaded weights.

| Weight file | Download | Destination within `ID_generator/` |
| :---------- | :------- | :-------------------------------- |
| `first_stage_decoder_state_dict.pt` | [Google Drive](https://drive.google.com/file/d/16IffdhIxFRDRbeV1HopOJ-hqKuE9rBz5/view?usp=sharing) | `models/autoencoder/first_stage_decoder_state_dict.pt` |
| `first_stage_encoder_state_dict.pt` | [Google Drive](https://drive.google.com/file/d/1_5fpHZxn6UOw9zQODTiacXpL0sChyR5c/view?usp=sharing) | `models/autoencoder/first_stage_encoder_state_dict.pt` |
| `Elastic_R100_295672backbone.pth` | [Google Drive](https://drive.google.com/file/d/119BxcwtECN0FsccJ_mIhAYmNorBKVarW/view?usp=sharing) | `utils/Elastic_R100_295672backbone.pth` |

Run the commands below from the `ID_generator/` directory. The autoencoder files are loaded relative to `paths.root` in `configs/paths/gpu_cluster.yaml`; ensure this setting resolves to your `ID_generator/` directory when the script runs. You can set it to the absolute path of that directory.

The VIPerson checkpoint download is listed separately in the [main README](../README.md#-pre-trained-models). The three files above do not replace the identity generator training checkpoint referenced by `training.checkpoint.path` in the steps below.

## Data Preparation
Prepare two types of resources as follows:
- Image dataset with fixed unified poses
- Corresponding descriptions matching the image dataset

    - The descriptions are stored in a JSON file with the following format:
```json
[
  {
    "image": "1.jpg", 
    "attri": 
        {"basic": "man", 
        "updress": "blue polo shirt", 
        "downdress": "tan shorts", 
        "shoes": "brown sandals", 
        "hat": null, 
        "glasses": "no", 
        "bag": null}, 
    "oricap": "A man wearing blue polo shirt and tan shorts and brown sandals", 
    "caption": "A man wearing blue polo shirt and tan shorts and brown sandals"
  }, 
  ...
]
```



## Step 1: Extract Text Embeddings for Training Images
Fill in the script parameters in `./extract_text_embeddings_from_dir_clip.py`:
- `--data_dir`: File path of descriptions matched with the image dataset
- `--out_dir`: Target path to save output text embeddings

## Step 2: Train Identity Generator
1. Modify `./configs/train_config.yaml`:
    - Set `training.checkpoint.restore` to `false` for training from scratch
    - Set `training.checkpoint.restore` to `true` for resume training, and specify `training.checkpoint.path` for checkpoint weights
2. (Optional) Update `data_root` in `./configs/paths/gpu_cluster.yaml` to the root directory of your image dataset
3. Update dataset paths in `./configs/dataset/person_folder.yaml`:
    - `dataset.samples_root`: Path of training image dataset
    - `dataset.embedding_root`: Path of text embeddings generated in Step 1
4. Launch training script:
```bash
python personmain.py
```

## Step 3: Generate New Identities
1. Edit configurations in `./ID_generator/configs/person_config.yaml`:
    -  `gener_path`: Storage path for images of newly generated identities
    - `gener_context_path`: Storage path for corresponding random vectors
    - `training.checkpoint.path`: Model weight checkpoint trained in Step 2

2. Run the identity generation script:
```bash
python persongener_uncondi_new.py
```

## Step 4: Generate Hard Identities
1. Generate descriptions and matching text embeddings for each new identity obtained in Step 3
2. Configure parameters in `./ID_generator/configs/person_config.yaml`:
    - `gener_path_text_emb`: Path of text embeddings of generated new identity images
    - `gener_hard_path`: Output storage path for hard identity results
3. Execute the generation script:
```bash
python persongener_part_dropout.py
```
