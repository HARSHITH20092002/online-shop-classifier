import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.scrapers.fetcher import fetch_website_content
from src.models.classifier import ShopClassifier

BENCHMARK_INPUT = "data/processed/benchmark_results.csv"
DATASET_RAW = "data/FR_online_market_discovery_2024.parquet" 
OUTPUT_RESULTS = "data/processed/benchmark_results_evaluated.csv"
OUTPUT_SUMMARY = "data/processed/model_comparison_metrics.csv"

def compute_metrics(name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[True, False])
    tp, fn = cm[0][0], cm[0][1]
    fp, tn = cm[1][0], cm[1][1]
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        "Model / Method": name,
        "Accuracy (%)": round(acc * 100, 2),
        "Precision (%)": round(prec * 100, 2),
        "Recall (%)": round(rec * 100, 2),
        "F1-Score (%)": round(f1 * 100, 2),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn
    }

def run_evaluation():
    print("=" * 70)
    print("🔍 RUNNING RIGOROUS EVALUATION: SOLUTION 1 vs SOLUTION 2 vs HYBRID")
    print("=" * 70)

    # 1. Load ground-truth benchmark set
    if not os.path.exists(BENCHMARK_INPUT):
        print(f"❌ Could not find {BENCHMARK_INPUT}. Please check the path.")
        return

    df = pd.read_csv(BENCHMARK_INPUT)
    target_col = "ground_truth" if "ground_truth" in df.columns else "is_online_shop"
    domain_col = "domain" if "domain" in df.columns else df.columns[0]

    y_true = df[target_col].astype(str).str.lower().map({
        "true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False
    }).tolist()

    classifier = ShopClassifier()
    
    preds_s1 = []
    preds_s2 = []
    preds_hybrid = []

    print(f"Evaluating {len(df)} domains...")

    for idx, (_, row) in enumerate(df.iterrows(), 1):
        domain_val = str(row[domain_col])
        url = domain_val if domain_val.startswith("http") else f"https://{domain_val}"
        
        # Ingest content via historical snippet or live fetcher
        fetched = fetch_website_content(url, historical_row=row.to_dict())

        # --- Test Solution 1 Standalone (Structural Heuristics) ---
        # Rule: If heuristics identify proof, it predicts True/False.
        # If inconclusive (None), standalone heuristic CANNOT confirm a shop -> predicts False.
        res_s1 = classifier.solution_1_heuristics(fetched)
        if res_s1 is not None:
            preds_s1.append(res_s1.get("is_shop", False))
        else:
            preds_s1.append(False)

        # --- Test Solution 2 Standalone (Probabilistic ML Text Scoring) ---
        # Evaluates transactional keywords across the entire text payload directly
        res_s2 = classifier.solution_2_ml_engine(fetched)
        preds_s2.append(res_s2.get("is_shop", False))

        # --- Test Combined Hybrid Pipeline ---
        res_hybrid = classifier.predict(fetched)
        preds_hybrid.append(res_hybrid.get("is_shop", False))

        if idx % 250 == 0 or idx == len(df):
            print(f"  Processed {idx}/{len(df)} domains...")

    # 2. Compute exact metrics
    m1 = compute_metrics("Solution 1 (Structural Heuristics)", y_true, preds_s1)
    m2 = compute_metrics("Solution 2 (Probabilistic ML Text)", y_true, preds_s2)
    m_hyb = compute_metrics("Combined Hybrid Pipeline", y_true, preds_hybrid)

    summary_df = pd.DataFrame([m1, m2, m_hyb])
    
    print("\n" + "=" * 70)
    print("📊 BENCHMARK EVALUATION RESULTS (1,500 GROUND-TRUTH DOMAINS)")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70)

    # 3. Save predictions to CSV so every number is audit-ready
    df["pred_solution_1"] = preds_s1
    df["pred_solution_2"] = preds_s2
    df["pred_hybrid"] = preds_hybrid
    
    df.to_csv(OUTPUT_RESULTS, index=False)
    summary_df.to_csv(OUTPUT_SUMMARY, index=False)
    
    print(f"\n💾 Saved full itemized predictions to: {OUTPUT_RESULTS}")
    print(f"💾 Saved summary metrics table to:     {OUTPUT_SUMMARY}")

if __name__ == "__main__":
    run_evaluation()