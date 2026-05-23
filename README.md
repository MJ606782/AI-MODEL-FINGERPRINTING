# AI Model Fingerprinting & Forensic Stylometry

An end-to-end natural language processing pipeline built on top of a fine-tuned **DistilBERT** transformer architecture to detect, decode, and map the distinct stylistic authorship signatures of 5 prominent Large Language Models (ChatGPT, Claude, Gemini, LLaMA, and Mistral).

##  Project Overview & Engineering Breakthroughs
During initial baseline tracking, the classifier encountered an optimization bottleneck (stuck at ~13% accuracy) due to categorical target index mismatches and structural formatting biases where the network relied heavily on raw markdown elements (like tables and hashes) rather than linguistic stylometry.

### Core Architecture Upgrades:
1. **Dynamic Token Augmentation:** Implemented a robust pre-processing pipeline that strips volatile structural patterns to force the transformer to evaluate underlying syntactic and lexical habits.
2. **Layer Liberation & Tuning:** Unfroze the base encoder blocks of the DistilBERT architecture and applied an optimized learning rate ($2\times 10^{-5}$) over 5 epochs to break out of the low-accuracy local minimum.
3. **Robust Environment Routing:** Architected the deployment pipeline to dynamically recognize its execution context—loading fine-tuned weights locally while falling back to a clean sandbox wrapper in restricted cloud environments.

As a direct result of these modifications, overall validation accuracy shot up to a stellar **90.0%**.

---

##  Performance & Validation Metrics
The model was rigorously validated against an untouched verification partition. Below is the validated classification report and true positive boundary mapping:

### Forensic Metrics Breakdown
* **Claude (F1: 0.92):** High separation accuracy; successfully maps complex semantic structures and empathetic pacing.
* **Mistral (F1: 0.95):** Achieved the highest individual style resolution performance.
* **Gemini (Precision: 0.97):** Exceptional precision marker—minimal false positive generation matches.
* **Overall Metrics Accuracy:** `90.0%`

### Confirmed Confusion Matrix Heatmap
The true positive vs. false positive classification boundaries across all 5 LLM targets:

![Confusion Matrix](confusion_matrix.png)

---

##  Advanced Features & Diagnostics
Beyond pure classification, this repository integrates cutting-edge diagnostics to stress-test and interpret the model's performance:
* **Explainable AI (XAI):** Integrated SHAP (SHapley Additive exPlanations) visualizers (`shap_original_claude.png`) to extract and analyze exactly which tokens heavily influence a specific model's fingerprint calculation.
* **Adversarial Attack Simulation:** Features a dedicated testing module (`adversarial_attacks.py`) designed to simulate stylistic camouflage and evaluate the classifier's defensive boundaries against prompt-spoofing techniques.

---

##  Repository Layout
```text
AI-model-fingerprinting/
│
├── src/                         # Core execution pipeline files
│   ├── local_tune.py            # 5-epoch unfrozen training engine
│   ├── evaluate_matrix.py       # Metrics extraction and Seaborn plotting
│   └── adversarial_attacks.py   # Robustness and prompt-spoofing benchmarks
│
├── app.py                       # Front-facing Streamlit dashboard interface
├── requirements.txt             # Environment dependency manifests
├── confusion_matrix.png        # Evaluated performance visual artifact
└── README.md                    # Project documentation
