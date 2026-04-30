"""Milestone 1 RAG-style advisory demo.

This is a lightweight local retrieval prototype. Final version will retrieve
from ICAR/agriculture bulletins and use an LLM for farmer-friendly generation.
"""

from pathlib import Path
import json

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

KNOWLEDGE_BASE = [
    {
        "disease": "leaf_spot",
        "region": "India",
        "advice": "Remove heavily infected leaves, avoid overhead irrigation, improve spacing, and consult a local agriculture officer for recommended fungicide usage.",
    },
    {
        "disease": "leaf_blight",
        "region": "India",
        "advice": "Use disease-free seed, remove infected debris, avoid excess nitrogen, and follow region-approved fungicide schedules.",
    },
    {
        "disease": "healthy_leaf",
        "region": "India",
        "advice": "Continue regular monitoring, balanced irrigation, and preventive field sanitation.",
    },
]


def retrieve_advice(disease: str, region: str = "India") -> dict:
    for record in KNOWLEDGE_BASE:
        if record["disease"] == disease and record["region"] == region:
            return record
    return {
        "disease": disease,
        "region": region,
        "advice": "No exact advisory found. Consult a certified agricultural extension expert.",
    }


def main() -> None:
    predicted_disease = "leaf_spot"
    confidence = 0.82
    region = "India"
    retrieved = retrieve_advice(predicted_disease, region)

    advisory = f"""# Sample Localized Crop Advisory

## Predicted Disease
{predicted_disease}

## Confidence
{confidence:.2f}

## Region
{region}

## Retrieved Advisory
{retrieved['advice']}

## Ethics and Safety Note
This is AI-assisted guidance. Farmers should confirm pesticide or chemical use with local agricultural experts before applying treatment.
"""

    (RESULTS_DIR / "sample_advisory.md").write_text(advisory, encoding="utf-8")
    print(advisory)


if __name__ == "__main__":
    main()
