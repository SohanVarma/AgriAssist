# Domain Note - Agriculture

## Title
AgriAssist: Multimodal Crop Disease Diagnosis and Localized Advisory System using Generative AI, RAG, and Agentic Reasoning

## Motivation
Agriculture is a critical sector in India, and many farmers depend on timely crop-health decisions to protect yield and income. Crop diseases can spread quickly, but access to expert diagnosis and reliable treatment guidance is limited in many rural areas. A scalable AI-based system can help farmers identify possible crop diseases earlier and receive actionable recommendations.

## Real Problem
Farmers often struggle to identify crop diseases from visual symptoms and usually receive generic advice that may not match their crop, disease severity, or region. Existing tools often stop at image classification and do not provide grounded, localized, or safety-aware treatment support. Data scarcity is another challenge because rare diseases have fewer labeled images, which makes AI models less reliable in real field conditions.

## Who It Affects
This problem affects small and marginal farmers, agricultural extension workers, agri-tech advisory platforms, and government agricultural support programs. Farmers with limited access to experts are especially affected because delayed or incorrect action can reduce yield and increase input costs.

## Why Existing Solutions Fall Short
Most crop disease systems focus only on classifying an uploaded leaf image. They do not retrieve verified agricultural guidance, explain confidence, or adapt recommendations to Indian regional conditions. Many models are trained on controlled datasets, so they may fail when used on noisy field images. Existing systems also do not address rare disease data imbalance effectively.

## Research Gap
There is a gap for an integrated agricultural AI system that combines multimodal image diagnosis, RAG-based advisory generation, synthetic data generation for rare diseases, India-specific localization, and ethical safeguards. Such a system goes beyond prediction and supports real decision-making.

## Proposed Approach
AgriAssist will use a vision model to detect crop disease from an image and provide a confidence score. A retrieval module will search agricultural knowledge sources such as crop bulletins and disease management notes. An LLM-based advisory module will generate farmer-friendly treatment and prevention steps. A localization layer will adapt advice based on Indian region and crop conditions. A synthetic data module will use diffusion-based generation in the final version to create rare disease images and improve classifier robustness. An ethics layer will display confidence, limitations, and a recommendation to consult agricultural experts before chemical use.

## Justification
This approach is justified because it connects the technical model to the real agricultural problem. Multimodal AI supports disease diagnosis from images, RAG grounds the advice in agricultural knowledge, generative models address rare disease data scarcity, and the ethics layer improves trust and responsible use. The project is suitable for GenAI evaluation because it spans generative models, LLMs, RAG, multimodal AI, and ethics.
