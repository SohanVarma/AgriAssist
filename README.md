# AgriAssist: Multimodal Crop Disease Diagnosis and Localized Advisory System

## Problem Summary
Farmers often struggle to identify crop diseases early and receive reliable, region-specific treatment advice. Existing AI tools usually stop at image classification and do not provide grounded, localized, or safety-aware recommendations.

AgriAssist is a prototype for a multimodal GenAI system that combines:

- crop disease image classification,
- synthetic rare-disease image generation for data scarcity,
- RAG-based agricultural advisory generation,
- India-specific localization,
- ethical safety disclaimers and confidence reporting.

## Project Components

- **Domain research note:** `domain_note.md` / `domain_note.pdf`
- **Data pipeline:** `src/02_data_pipeline.py`
- **Initial disease classification model:** `src/03_train_initial_model.py`
- **RAG-style advisory demo:** `src/04_rag_advisory_demo.py`
- **Synthetic rare-disease data demo:** `src/05_synthetic_data_demo.py`

## Data Pipeline
Run:
```bash
python3 src/01_create_sample_dataset.py
python3 src/02_data_pipeline.py
```

This creates a sample crop disease image dataset, validates image paths, loads metadata, and creates train/validation splits.

Expected output files:
```text
results/dataset_summary.json
results/train_metadata.csv
results/val_metadata.csv
```

## Model Training
Run:
```bash
python3 src/03_train_initial_model.py
```

This trains a small CNN for crop disease classification and logs preliminary results.

Expected output files:
```text
results/training_metrics.csv
results/training_loss.png
results/initial_model.pt
```

## Repository Structure
```text
AgriAssist/
├── README.md
├── domain_note.md
├── domain_note.pdf
├── requirements.txt
├── data/
├── src/
├── notebooks/
├── results/
└── docs/
```

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate       # Mac/Linux
pip install -r requirements.txt
```

## Run Full Demo
```bash
python3 src/01_create_sample_dataset.py
python3 src/02_data_pipeline.py
python3 src/03_train_initial_model.py
python3 src/04_rag_advisory_demo.py
python3 src/05_synthetic_data_demo.py
```

## Outputs
Generated files are saved in `results/`:

- `dataset_summary.json`
- `train_metadata.csv`
- `val_metadata.csv`
- `training_metrics.csv`
- `training_loss.png`
- `sample_advisory.md`
- `synthetic_generation_summary.json`

## Future Improvements
This prototype can be extended to:

- use PlantVillage or field crop disease datasets,
- train a stronger CNN/ViT classifier,
- generate rare disease images using a diffusion model,
- retrieve advisories from ICAR/agriculture bulletins,
- compare baseline classifier vs classifier trained with synthetic augmentation,
- evaluate advisory quality with and without RAG.
