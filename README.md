# Uncertainty-Aware and Explainable Neural Network Classification of Environmental Sounds Using DataSEC

## Project Overview

This project investigates neural network and pretrained audio-model approaches for classifying environmental sound recordings using the DataSEC dataset.

The work focuses on classification performance, uncertainty estimation, explainability and a practical web application for examining individual audio clips.

The intended application is a decision-support tool for environmental noise assessment and acoustic monitoring. It is not designed to replace expert listening or make regulatory compliance decisions.

## Objectives

- Explore and analyse the DataSEC environmental sound dataset.
- Develop an Artificial Neural Network (ANN) classifier using a pretrained PANNs CNN14 architecture.
- Compare the ANN with a classical machine-learning baseline.
- Evaluate pretrained CLAP audio representations using supervised and zero-shot approaches.
- Investigate model calibration and uncertainty-based prediction deferral.
- Examine model explanations using Grad-CAM, LIME and occlusion analysis.
- Develop a Streamlit application for interactive audio classification.

## Dataset

The project uses DataSEC v4:

- Dataset: DataSEC - Dataset for Sound Event Classification of environmental noise
- Record: https://zenodo.org/records/23097290
- Version: v4
- Archive: DATASEC.zip
- Original dataset: 4,292 mono-channel WAV clips
- Sampling rate: 44.1 kHz
- Main classes: 22
- Subclasses: 28

The complete dataset is approximately 4.6 GB and is intentionally not included in this GitHub repository.

See [`data/README.md`](data/README.md) for download and storage instructions.

The dataset contains material from different sources and has mixed licensing conditions. Users must consult the original record and applicable source licences before using or redistributing the audio.

## Repository Structure

```text
.
├── app/             # Streamlit application
├── data/            # Dataset instructions; audio files are excluded
├── src/             # Training, evaluation and explainability scripts
├── notebooks/       # Experimental notebooks
├── logs/            # Required split and label metadata; generated logs excluded
├── models/          # Local model artefacts; excluded from Git
├── report/          # Figures and project visualisations
├── Evidences/       # Development and assessment evidence
├── download_datasec.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_DIRECTORY>
```

### 2. Create a Python environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install project dependencies

The project was developed using Python 3.10.11 and an NVIDIA GPU with CUDA 12.8-enabled PyTorch.

Install the pinned dependencies:

```bash
pip install -r requirements.txt
```

Some dependencies and bundled PANNs source components may require additional setup depending on the Python and operating-system environment.

## Downloading DataSEC

The dataset is stored outside the repository.

Run:

```bash
python download_datasec.py
```

By default, the script saves the archive in a `DATASEC` directory alongside the repository's parent directory.

A custom location can be specified:

```bash
python download_datasec.py --output-dir "D:/Datasets/DATASEC"
```

The downloader retrieves the archive from Zenodo and verifies its published MD5 checksum.

After downloading, extract `DATASEC.zip`. By default, the project expects the extracted dataset at `../DATASEC/DATASEC` relative to the repository. If you store it elsewhere, set the `DATASEC_ROOT` environment variable to the extracted dataset directory before running the scripts. For example, in Windows PowerShell: `$env:DATASEC_ROOT = "D:\Datasets\DATASEC"`.

Do not commit the archive or extracted audio files to GitHub.

The repository includes two small metadata files required for reproducibility:
- `logs/hierarchical_label_mapping.json`: the main-class and subclass mappings.
- `logs/final_split_manifest_with_subclasses.csv`: the fixed dataset split.

Other generated logs and experiment outputs are excluded from Git.

## Pretrained Checkpoints

The project uses two pretrained model checkpoints. These files are not included in this repository because of their large size.

### 1. PANNs CNN14 checkpoint

- Purpose: Initialisation of the partially fine-tuned CNN14 model.
- Official source: https://zenodo.org/records/3987831
- Original filename: `Cnn14_mAP=0.431.pth`
- Required local filename: `Cnn14_mAP_0.431.pth`
- Required location: `src/Cnn14_mAP_0.431.pth`
- Official MD5: `541141fa2ee191a88f24a3219fff024e`

Download the original checkpoint from Zenodo and save it using the required local filename.

### 2. LAION-CLAP checkpoint

- Purpose: CLAP linear probe, zero-shot evaluation and final test evaluation.
- Official project: https://github.com/LAION-AI/CLAP
- Checkpoint: `630k-audioset-best.pt`
- Required location: `src/630k-audioset-best.pt`
- Checkpoint download: https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-best.pt
- Published SHA-256: `8053c9775516af2f4902e1e8281e356cc1bf7a85e8b761908170767b77c3f037`

Download the checkpoint and place it in the specified location.

### Important

Both checkpoint files are excluded from Git by `.gitignore`.

Users must obtain the checkpoints from their respective sources before running the experiments that depend on them.

Respect the original model licences and attribution requirements. This repository does not redistribute pretrained model weights.

## Models Evaluated

### 1. PANNs CNN14

A pretrained CNN14 audio model was partially fine-tuned using DataSEC log-mel spectrogram inputs. The implementation includes main-class and subclass prediction heads.

### 2. MFCC and Spectral SVM

A classical baseline using MFCC and spectral features with a StandardScaler and RBF Support Vector Classifier.

### 3. CLAP Linear Probe

Pretrained CLAP audio embeddings were extracted and used with a supervised logistic regression classifier.

This is a supervised classifier operating on pretrained audio representations; it is not a neural network trained from scratch.

### 4. CLAP Zero-Shot

The pretrained CLAP model was evaluated using text descriptions of the environmental sound classes. Prompt refinement was performed using validation data.

## Final Held-Out Test Results

The final evaluation used 654 eligible test clips. One previously exposed test clip was excluded from quantitative evaluation.

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| CLAP linear probe | 97.09% | 95.34% |
| PANNs CNN14 | 95.72% | 92.03% |
| CLAP zero-shot | 85.32% | 83.21% |
| MFCC and spectral SVM | 85.63% | 75.71% |

A class-stratified paired bootstrap with 10,000 repetitions was used to estimate confidence intervals.

| Model | Macro-F1 95% CI |
|---|---|
| CLAP linear probe | 93.40%–97.22% |
| PANNs CNN14 | 89.29%–94.53% |
| CLAP zero-shot | 80.30%–86.03% |
| MFCC and spectral SVM | 71.43%–79.45% |

The CLAP linear probe achieved the highest macro-F1 in this evaluation. Its macro-F1 improvement over CNN14 was statistically detectable in the paired bootstrap comparison, while the accuracy difference was not statistically clear.

These results should be interpreted with the dataset and pretrained-model limitations described below.

## Uncertainty Estimation

The CNN14 model was evaluated using:

- Temperature scaling for probability calibration.
- Empirical conformal prediction-set analysis.
- Maximum-softmax-probability risk-coverage analysis.

The observed conformal coverage on the eligible test set was approximately 91.59%, with a mean prediction-set size of approximately 0.94.

These are empirical results for this experimental setup, not a claim of an independent distribution-free coverage guarantee.

## Explainability

The project investigates:

- Grad-CAM for visualising influential regions of the CNN14 feature maps.
- LIME for local model explanations.
- Occlusion analysis using time and frequency masking.
- Deletion and insertion tests against random baselines.
- Limited explanation sanity checks.

Explanations are treated as diagnostic evidence about model behaviour. They do not establish the physical cause of a sound or prove that a prediction is correct.

## Web Application

A Streamlit application is provided in the `app/` directory.

From the project root, run:

```bash
streamlit run app/app.py
```

The application supports audio upload, classification, top predictions, calibrated confidence display, uncertainty-related prediction information and Grad-CAM visualisation.

The application is intended for decision support. Uncertain or potentially out-of-scope recordings should be reviewed by a human.

## Reproducibility and Limitations

- The project uses a fixed experimental split and a separate held-out test evaluation.
- The final test evaluation excludes one clip that had previously been exposed during application testing.
- DataSEC does not provide reliable original per-file source identifiers. Therefore, a verified source-level grouped split could not be established.
- Acoustic similarity checks were used as limited screening for possible duplicate or similar recordings; they do not establish recording provenance.
- CLAP pretraining overlap with online-source DataSEC material cannot be ruled out.
- The dataset contains pre-segmented sound clips. The system is not a complete long-recording event-detection solution.
- Model performance on this dataset does not establish performance in every real-world acoustic environment.


### Reproducing the Application Model

The trained model and calibration artefacts are excluded from Git because they are generated files and include a large model checkpoint.

To reproduce the application from a fresh clone, first ensure that the DataSEC dataset, the PANNs CNN14 pretrained checkpoint and the two required metadata files are available.

The following commands reproduce the final CNN14 configuration used in this project.

Run them from the repository root, in this order:

**1. Train the final CNN14 model**

```bash
python src/optimise_cnn14.py --experiment-id FINAL-123 --optimizer adamw --learning-rate 0.0003 --batch-size 4 --fine-tune-depth 1 --epochs 15 --seed 123
```

**2. Perform temperature calibration**

```bash
python src/calibrate_cnn14.py
```

**3. Generate conformal prediction parameters**

```bash
python src/conformal_cnn14.py
```

**4. Start the Streamlit application**

```bash
streamlit run app/app.py
```

The training configuration uses AdamW, a learning rate of 0.0003, batch size 4, one fine-tuned convolutional block, 15 epochs and seed 123.

The CNN14 optimisation script creates the `FINAL-123` experiment directory and protects existing experiment results from accidental overwriting. Run the reproduction commands on a fresh clone where the output directory does not already exist.

The application requires the generated CNN14 checkpoint, temperature calibration metrics, conformal parameters and hierarchical label mapping.

The LAION-CLAP checkpoint is required for the CLAP experiments, but not for running the CNN14 Streamlit application.

Results may vary slightly across hardware and software environments.


## Dataset Citation

Fredianelli, L., Artuso, F., Pompei, G., Licitra, G., Iannace, G., & Akbaba, A. (2026). *DataSEC - Dataset for Sound Event Classification of environmental noise* (Version 4). Zenodo.

https://doi.org/10.5281/zenodo.23097290

Please consult the Zenodo record for the dataset's current licence and attribution requirements.

## Third-Party Components

This project uses third-party libraries and pretrained models, including PANNs and LAION-CLAP. Their original licences, model terms and attribution requirements remain applicable.

The MIT licence in this repository applies only to original project code and documentation, not to the dataset or third-party components.

## Academic Use

This repository supports an individual AI coursework project. The reported results describe the experiments conducted for this project and should not be interpreted as a formal benchmark of all possible DataSEC classification methods.






