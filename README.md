# UUEKAN: An Edge-Enhanced Kolmogorov-Arnold Network with Uncertainty-Guided Attention for Medical Image Segmentation

## Abstract

Accurate segmentation of medical images, particularly ultrasound, is challenged by inherent ambiguity, noise, and artifacts like acoustic shadowing, which impair model performance. To address this, we propose UUEKAN, a novel network designed to enhance segmentation by integrating edge-aware feature extraction with uncertainty guidance. UUEKAN introduces two key innovations within its encoder-decoder architecture. First, an Edge-enhanced Kolmogorov-Arnold Network (EKAN) module captures complex non-linear relationships while strengthening boundary features using a Sobel-based learnable edge extractor. Second, it incorporates an Uncertainty-guided Magnitude-Aware Linear Attention (U-MALA) module within its skip connections. The U-MALA module generates uncertainty maps to dynamically re-weight features, focusing the model on artifacts and unsaturated regions while suppressing noise. Extensive experiments on challenging ultrasound datasets demonstrate UUEKAN's superior performance in segmenting ill-defined boundaries and shadowed regions, achieving significant improvements in accuracy and robustness. Our work presents a potent architecture that combines the expressive power of KANs, dedicated edge extraction, and multi-scale uncertainty-guided attention, offering an effective solution for highly challenging medical segmentation tasks.



![image](./pictures/UUEKAN.svg)



![image](./pictures/UMALA.svg)




![image](./pictures/EKAN.svg)



![image](./pictures/comparison_figure.svg)

## Dataset

### Dataset Structure

The project expects the datasets to be organized in the following structure:<br>
Datasets Download：inputs<br>
https://pan.baidu.com/s/15yycTpqSW-IoScNOsalkVw?pwd=jtw3 <br>
```
<data_directory>/
└── <dataset_name>/
    ├── images/
    │   ├── image_001.png
    │   ├── image_002.png
    │   └── ...
    └── masks/
        ├── image_001_mask.png
        ├── image_002_mask.png
        └── ...
```

-   `<data_directory>`: The root directory for all your datasets (e.g., `inputs`).
-   `<dataset_name>`: The name of the specific dataset you are using (e.g., `heus`, `busi`).
-   `images/`: Contains the original medical images.
-   `masks/`: Contains the corresponding segmentation masks.

### Download

*TODO: Add the URL and instructions for downloading the dataset(s) here.*

## Getting Started

Follow these steps to set up the environment and run the training and validation process.

### 1. Setup Environment

First, create a Python environment and install the required dependencies using the `requirements.txt` file.

```bash
# Create a conda environment (optional but recommended)
conda create -n uuekan python=3.10
conda activate uuekan

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Download the dataset and place it in the `inputs` directory following the structure described above. For example, for the `heus` dataset, the path should be `inputs/heus/`.

### 3. Run Training and Validation

The easiest way to start training and validation is to use the provided shell script.

```bash
bash train_val.sh
```

This script will:
1.  Train the `UUEKAN` model on the `HEUS` `BUSI` `CVC`dataset.
2.  Save the best model and logs to the `outputs/UUEKAN/Datasetname/` directory.
3.  Run validation on the test set using the saved model.

You can customize the `train_val.sh` script to change parameters like the dataset name, input size, number of epochs, etc.

Alternatively, you can run the `train.py` and `val.py` scripts directly with your desired arguments.

**Training:**
```bash
python train.py --arch UUEKAN --dataset your_dataset --name your_experiment_name
```

**Validation:**
```bash
python val.py --model UUEKAN --name your_experiment_name
```
