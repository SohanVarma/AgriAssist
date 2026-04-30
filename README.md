# AgriAssist: Multimodal Crop Disease Diagnosis and Localized Advisory System

## Problem Summary
Farmers often struggle to identify crop diseases early and receive reliable, region-specific treatment advice. Existing AI tools usually stop at image classification and do not provide grounded, localized, or safety-aware recommendations.

AgriAssist is a milestone-1 prototype for a future multimodal GenAI system that combines:

- crop disease image classification,
- synthetic rare-disease image generation for data scarcity,
- RAG-based agricultural advisory generation,
- India-specific localization,
- ethical safety disclaimers and confidence reporting.

## Milestone 1 Required Deliverables

- **Domain research note submitted (1 page):** `domain_note.md` / `domain_note.pdf`
- **Data pipeline working and data loaded:** `src/02_data_pipeline.py`
- **Initial model running with preliminary results:** `src/03_train_initial_model.py`

## Milestone 1 Evidence

### Data pipeline working and data loaded
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

### Initial model running with preliminary results
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

## Run Full Milestone 1 Demo
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

## Final Evaluation Plan
For the final version, this prototype will be extended to:

- use PlantVillage or field crop disease datasets,
- train a stronger CNN/ViT classifier,
- generate rare disease images using a diffusion model,
- retrieve advisories from ICAR/agriculture bulletins,
- compare baseline classifier vs classifier trained with synthetic augmentation,
- evaluate advisory quality with and without RAG.
