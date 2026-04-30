# Architecture

```text
Farmer crop image + optional region/crop query
        |
        v
Image Diagnosis Model (CNN / ViT / CLIP)
        |
        v
Disease + confidence score
        |
        v
Retrieval Agent over agriculture bulletins
        |
        v
Localized Advisory Generator
        |
        v
Ethics + Verification Layer
        |
        v
Farmer-friendly recommendation
```

## Synthetic Data Component
Rare disease classes often have fewer labeled images. The final system will use a diffusion-based generator to create additional rare disease image samples. These synthetic samples will be used only for training augmentation and will be evaluated against a baseline model trained only on real images.
