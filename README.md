# Online Shop Classifier

**Machine Learning for E-Commerce Domain Classification**  
*MSc in Artificial Intelligence for Business Intelligence · University of Leicester*  
*In collaboration with GfK / NielsenIQ*

---

## 1. Project Overview

In digital market intelligence, assessing commercial presence across international markets requires determining which web domains operate as active e-commerce storefronts. Historically, this verification has relied on human annotators manually inspecting search discovery links one by one to verify checkout capability—a workflow that is labor-intensive, costly, and difficult to scale across tens of thousands of domains.

This project develops, benchmarks, and deploys **two distinct machine learning classification methods** alongside a **cascading hybrid architecture** to replace manual verification with an auditable, reproducible classification pipeline.

---

## 2. System Architecture

The classification pipeline ingests domain URLs and discovery metadata, passes them through a resilient multi-tier content extraction layer, and classifies them using a dual-stage decision architecture.

```
                           [ Discovered Domain / URL ]
                                       │
                                       ▼
     ┌──────────────────────────────────────────────────────────────────┐
     │            Tiered Content Extraction & Fallback Layer            │
     │  Tier 1: Playwright Headless Browser (Full JS DOM Rendering)     │
     │  Tier 2: curl-cffi TLS Fingerprint Impersonation (WAF Bypass)    │
     │  Tier 3: Wayback Machine API (Historical Snapshot Recovery)      │
     │  Tier 4: Search Snippet Fallback (Query & Metadata Ingestion)    │
     └─────────────────────────────────┬────────────────────────────────┘
                                       │
                                       ▼
     ┌──────────────────────────────────────────────────────────────────┐
     │             STAGE 1: Structural Heuristics Classifier            │
     │   - Schema.org JSON-LD extraction (@type: Product, Offer)        │
     │   - Known non-shop entity exclusion (Doc readers, SaaS, blogs)   │
     │   - Marketplace URI subpath routing                              │
     └─────────────────────────────────┬────────────────────────────────┘
                                       │
                         Is Structured Proof Detected?
                                      / \
                              YES    /   \    NO / INCONCLUSIVE
                                    /     \
                                   ▼       ▼
                       ┌───────────────┐ ┌───────────────────────────────┐
                       │ INSTANT EXIT  │ │ STAGE 2: Probabilistic ML     │
                       │  (High Conf)  │ │ Transactional Keyword Model   │
                       └───────────────┘ └───────────────┬───────────────┘
                                                         │
                                                         ▼
                                          [ Final Class Verdict & Reason ]
```

### Classification Approaches

* **Solution 1: Structural Heuristics Engine**  
  Inspects high-confidence structured DOM signals. It parses Schema.org JSON-LD microdata (`@type: Product`, `@type: Offer`), validates platform-specific URI structures, and screens against known non-shop signatures. If structural proof is identified, it emits a classification immediately.
* **Solution 2: Probabilistic ML Text Model**  
  Acts as a statistical content classifier when structured markup is absent or custom-coded. It evaluates multi-lingual transactional keyword densities across page content and metadata, weighting high-intent purchase triggers (`w = 0.40`) and general merchandising indicators (`w = 0.15`).
* **Cascading Hybrid Pipeline**  
  Integrates both stages sequentially: domains first pass through Solution 1. If Solution 1 is inconclusive, the payload cascades to Solution 2.

---

## 3. Empirical Benchmark Results

All configurations were evaluated on a benchmark set of **1,500 ground-truth domains** provided and verified with GfK. Evaluation metrics were generated using `evaluate_benchmark.py` and logged to `data/processed/model_comparison_metrics.csv`.

### Performance Comparison

| Model / Configuration | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | TP | FP | FN | TN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Solution 1 (Structural Heuristics)** | **72.47%** | **65.11%** | 86.69% | **74.36%** | 599 | 321 | 92 | **488** |
| **Solution 2 (Probabilistic ML Text)** | 72.07% | 64.66% | 86.83% | 74.12% | 600 | 328 | 91 | 481 |
| **Combined Hybrid Pipeline** | 68.73% | 61.28% | **87.26%** | 72.00% | **603** | 381 | **88** | 428 |

### Key Findings & Trade-offs

1. **Discovery Maximization (High Recall):** The Combined Hybrid pipeline achieves the highest storefront discovery recall (**87.26%**), successfully identifying 603 of the 691 ground-truth shops and minimizing missed commercial stores (88 False Negatives).
2. **Precision vs. Recall Trade-off:** Cascading inconclusive heuristic cases to the probabilistic text model introduces 60 additional false alarms (FP increases from 321 to 381). As a result, overall accuracy and precision are slightly lower in the hybrid configuration than in standalone heuristics.
3. **Resilience Recovery:** Across the test corpus, 84% of offline or unreachable domains were successfully classified via historical Wayback Machine snapshots and search snippet fallbacks, preventing missing-data drops.

---

## 4. Repository Structure

```
online-shop-classifier/
├── app.py                                  # Streamlit web application & benchmark UI
├── evaluate_benchmark.py                   # Script to independently benchmark S1, S2, and Hybrid
├── requirements.txt                        # Python dependencies
├── src/
│   ├── scrapers/
│   │   └── fetcher.py                      # Multi-tier extraction (Playwright, TLS, Wayback, Snippet)
│   └── models/
│       └── classifier.py                   # Solution 1, Solution 2, and Cascading Hybrid models
└── data/
    └── processed/
        ├── model_comparison_metrics.csv    # Logged summary metrics (Acc, Prec, Rec, F1, Matrix)
        ├── benchmark_results_evaluated.csv # Per-domain itemized predictions & decision reasons
        └── benchmark_results.csv           # 1,500 ground-truth evaluation domains
```

---

## 5. Installation & Setup

### Prerequisites

* Python 3.10+
* Git

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/HARSHITH20092002/online-shop-classifier.git](https://github.com/HARSHITH20092002/online-shop-classifier.git)
   cd online-shop-classifier
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   playwright install chromium
   ```

---

## 6. Execution & Usage

### 1. Run the Empirical Benchmark Suite
To re-evaluate Solution 1, Solution 2, and the Hybrid pipeline across all 1,500 benchmark domains:
```bash
python evaluate_benchmark.py
```
Outputs generated:
* `data/processed/model_comparison_metrics.csv`
* `data/processed/benchmark_results_evaluated.csv`

### 2. Launch the Streamlit Dashboard
To launch the interactive application:
```bash
streamlit run app.py
```

The application provides three operational views:
* **Single Domain Inspection:** Live classification of individual URLs with source tracing, latency metrics, and transparent decision audit strings.
* **Batch Dataset Results:** Batch processing of discovery datasets with filtered views and exportable CSV reports.
* **Model Benchmark Metrics:** Dynamic inspection of the 1,500-domain benchmark, including confusion matrices and comparative performance breakdowns.

---

## 7. Contributors & Acknowledgements

* **Author:** Harshith Kaithoju (MSc Artificial Intelligence for Business Intelligence, University of Leicester)
* **Industry Supervisor:** Maria Chernova (GfK / NielsenIQ)
* **Academic Supervisor:** Paul (University of Leicester, School of Computing and Mathematical Sciences)