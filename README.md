# CoDLAD
=========
Source code and data for "A Variational Autoencoder Framework with Tissue-Conditioned Latent Diffusion for Cross-Domain Anticancer Drug Response Prediction"

# Requirements
All implementations of CoDLAD are based on PyTorch. CoDLAD requires the following dependencies:
- python==3.7.16
- pytorch==1.13.1
- torch_geometric==2.3.1
- numpy==1.21.5+mkl
- scipy==1.7.3
- pandas==1.3.5
- scikit-learn=1.0.2
- hickle==5.0.2
# Data
- Data defines the data used by the model
    - data/TCGA records training data, test data, and labeling related to the five drugs associated with TCGA.
    - data/PDTC records training data, test data, and labeling related to the fifty drugs associated with PDTC.
    - data/ccle_sample_info.csv records biological information related to CCLE samples.
    - data/pretrain_ccle.csv records gene expression data from unlabeled CCLE samples.  Due to data storage limitations when uploading to Githup, we compress pretrain_ccle.csv into a compressed package. You can use a decompression tool to decompress the complete file when using it.
    - data/pretrain_tcga.csv records gene expression data from unlabeled TCGA samples. Due to data storage limitations when uploading to Githup, we compressed and uploaded pretrain_tcga.csv in separate volumes. When using them, you can use the decompression tool, which will automatically identify all the sub-volumes and merge them into a complete file.
    - data/pdtc_uq1000_feature.csv records gene expression data from unlabeled PDTC samples.
    - data/GDSC1_fitted_dose_response_25Feb20.csv and data/GDSC2_fitted_dose_response_25Feb20.csv records data on drug use and response in GDSC samples. Due to data storage limitations when uploading to Githup, we compress GDSC1_fitted_dose_response_25Feb20.csv into a compressed package. You can use a decompression tool to decompress the complete file when using it.
    - data/DrugResponsesAUCModels.txt records response data for PDTC sample-drug pairs. 
    - data/pdtc_gdsc_drug_mapping.csv records the 50 drug names associated with pdtc and their smiles. 
    - data/uq1000_feature.csv records gene expression data for unlabeled TCGA samples and CCLE samples. Due to data storage limitations when uploading to Githup, we compressed and uploaded pdtc_gdsc_drug_mapping.csv in separate volumes. When using them, you can use the decompression tool, which will automatically identify all the sub-volumes and merge them into a complete file.
    - data/xena_sample_info_df.csv records biological information related to TCGA samples.
- tools/model.py defines the model used in the training process.
- data.py defines the data loading of the model.
- pretrain.py defines the training of the domain invariant feature extraction phase of the model.
- classifier.py defines the classifier training of the model.

## Data Restoration

Since the `pretrain_tcga.csv` file exceeds GitHub's size limit, it has been split into 14 parts. You **must** merge these parts using our script before running any pretraining. The script automatically handles relative paths.

```bash
<!-- Run this from the project root -->
python scripts/merge_data.py

## Data Preparation and Custom Data Usage

To run **CoDLAD**, users need to organize their data into a **source domain** and a **target domain** with **consistent feature dimensions**. In our implementation, the source domain corresponds to **cell line gene expression data**, while the target domain corresponds to **patient (or tumor) gene expression data**.

In this work, we follow the data preprocessing protocol adopted in **CodeAE** [1], and use preprocessed CCLE and TCGA datasets as the default inputs. Each sample is represented by a fixed-dimensional gene expression vector (e.g., 1426 genes) and is associated with a **one-dimensional categorical tissue label**, which is used as conditional information during pretraining.

### Reference

[1] He, D., et al. *A context-aware deconfounding autoencoder for robust prediction of personalized clinical drug response from cell-line compound screening.*  
**Nature Machine Intelligence**, 4(10): 879–892, 2022.


## Usage

Once the environment is properly configured, **CoDLAD** can be executed using the data we provide.
Please note that **drug feature extraction is a required preprocessing step** before model training.

### Step 1: Drug Feature Pretraining

## Usage

Once the environment is properly configured, **CoDLAD** can be executed using the provided data.

> **Note:** Drug feature extraction is a **required preprocessing step** before model training.

### Step 1: Drug Feature Pretraining

First, extract drug molecular representations using a graph-based self-supervised pretraining strategy. This step encodes drug SMILES into fixed-dimensional embeddings and is required for downstream prediction.

```bash
python precontext.py
```

### Step 2: Cross-domain Representation Pretraining

Next, pretrain the cross-domain gene expression encoder using a framework combining VAE, latent diffusion, and adversarial alignment:

```bash
python pretrain.py
```

**Details:**
* **Source Domain:** Cell line gene expression data.
* **Target Domain:** Patient (or tumor) gene expression data.
* **Method:** A shared VAE and domain-specific private VAEs are jointly trained. Latent diffusion regularizes the shared latent space, while adversarial training (WGAN-GP) aligns the source and target distributions.

### Step 3: Downstream Drug Response Prediction

After representation pretraining, train the downstream classifier to predict drug response:

```bash
python classifier.py
```

At this stage, the pretrained gene expression encoders and drug encoders are loaded, and model performance is evaluated on the target-domain test data.

### One-command Reproduction (Optional)

To reproduce all experimental results reported in the paper using the provided data, you can run the following script:

```bash
python train_all.py
```

This script sequentially performs:
1. Cross-domain representation pretraining.
2. Downstream classifier training and evaluation.

Alternatively, you can run our program with your own data and some other settings as follows:
```
1. python pretrain_mask_vae.py \
--outfolder path/to/folder_to_save_pretrain_models \
--source path/to/your_pretrain_source_data.csv \
--target path/to/your_pretrain_target_data.csv

2. python classifier.py \
--dataset other \
--data path/to/your_data_folder \
--drug path/to/your_drug_name.csv \
--pretrain_model path/to/your_pretrain_models_path \
--outfolder path/to/save_result_and_others \
--outname result_file_name.csv 
```
Note: 
>You need to ensure that the data dimensions of your source and target domains are the same.

> The **your_data_folder** is a folder that contains many medication folders while each medication folder contains sourcedata.csv, targetdata.csv, sourcelabel.csv, targetlabel.csv. The format of each file can be referred to. /data/TCGA.


