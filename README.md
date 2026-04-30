# AgriAssist: Multimodal Crop Disease Diagnosis and Localized Advisory System

## Problem Summary
Farmers often struggle to identify crop diseases early and receive reliable, region-specific treatment advice. Existing AI tools usually stop at image classification and do not provide grounded, localized, or safety-aware recommendations.

AgriAssist is a milestone-1 prototype for a future multimodal GenAI system that combines:

- crop disease image classification,
- synthetic rare-disease image generation for data scarcity,
- RAG-based agricultural advisory generation,
- India-specific localization,
- ethical safety disclaimers and confidence reporting.

## Milestone 1 Status
This repository satisfies the Milestone 1 requirements:

1. Domain research note submitted: `domain_note.pdf`
2. Data pipeline working and data loaded: `src/02_data_pipeline.py`
3. Initial model running with preliminary results: `src/03_train_initial_model.py`

## Repository Structure
```text
project-agriassist-SE23UCSE163/
├── README.md
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
python -m venv .venv
source .venv/bin/activate       # Mac/Linux
# .venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Run Milestone 1
```bash
python src/01_create_sample_dataset.py
python src/02_data_pipeline.py
python src/03_train_initial_model.py
python src/04_rag_advisory_demo.py
python src/05_synthetic_data_demo.py
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
