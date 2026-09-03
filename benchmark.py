import os
import time
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.scrapers.fetcher import fetch_website_content
from src.models.classifier import ShopClassifier

# ==========================================
# BENCHMARK CONFIGURATION
# ==========================================
DATASET_PATH = "data/FR_online_market_discovery_2024.parquet"  # Or DE / ES files
SAMPLE_SIZE = 1500  # Evaluate 1,500 domains (Satisfies instructor's 1000-2000 requirement)
# ==========================================

def run_benchmark():
    print("==================================================================")
    print(f"🚀 Launching Benchmark Evaluation on {SAMPLE_SIZE} Domains...")
    print(f"📂 Source Dataset: {DATASET_PATH}")
    print("==================================================================")

    if DATASET_PATH.endswith(".parquet"):
        df = pd.read_parquet(DATASET_PATH)
    else:
        df = pd.read_csv(DATASET_PATH)

    # Normalize ground truth column
    target_col = next((c for c in ["is_online_shop", "is_shop", "label", "target"] if c in df.columns), None)
    if not target_col:
        print("❌ Ground truth column not found in dataset. Ensure 'is_online_shop' exists.")
        return

    domain_col = next((c for c in ["domain", "url", "website", "Root Domain"] if c in df.columns), df.columns[0])
    
    # Deduplicate to domain level and sample required size
    eval_df = df.drop_duplicates(subset=[domain_col]).dropna(subset=[target_col]).sample(n=min(SAMPLE_SIZE, len(df)), random_state=42).copy()
    eval_df["ground_truth"] = eval_df[target_col].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False})

    classifier = ShopClassifier()
    predictions_hybrid = []
    predictions_s1_only = []
    predictions_s2_only = []
    reasons = []
    methods = []
    data_sources = []

    start_time = time.time()

    for idx, (_, row) in enumerate(eval_df.iterrows(), 1):
        raw_domain = str(row[domain_col])
        url = raw_domain if raw_domain.startswith("http") else f"https://{raw_domain}"
        
        # Fetch with historical recovery
        fetched = fetch_website_content(url, historical_row=row.to_dict())
        
        # 1. Hybrid Prediction
        pred = classifier.predict(fetched)
        predictions_hybrid.append(pred["is_shop"])
        reasons.append(pred["reason"])
        methods.append(pred["method"])
        data_sources.append(fetched.get("data_source", "Unknown"))

        # 2. Isolated Solution 1 Evaluation
        s1 = classifier.solution_1_heuristics(fetched)
        predictions_s1_only.append(s1["is_shop"] if s1 else False)

        # 3. Isolated Solution 2 Evaluation
        s2 = classifier.solution_2_ml_engine(fetched)
        predictions_s2_only.append(s2["is_shop"])

        if idx % 100 == 0 or idx == len(eval_df):
            print(f"  [{idx}/{len(eval_df)}] Processed... Elapsed: {time.time() - start_time:.1f}s")

    y_true = eval_df["ground_truth"].tolist()

    # Metric Computation Helper
    def calc_metrics(y_actual, y_pred):
        cm = confusion_matrix(y_actual, y_pred, labels=[True, False])
        tp, fn = cm[0][0], cm[0][1]
        fp, tn = cm[1][0], cm[1][1]
        return {
            "Accuracy": accuracy_score(y_actual, y_pred),
            "Precision": precision_score(y_actual, y_pred, zero_division=0),
            "Recall": recall_score(y_actual, y_pred, zero_division=0),
            "F1-Score": f1_score(y_actual, y_pred, zero_division=0),
            "TP": tp, "FP": fp, "TN": tn, "FN": fn
        }

    m_hybrid = calc_metrics(y_true, predictions_hybrid)
    m_s1 = calc_metrics(y_true, predictions_s1_only)
    m_s2 = calc_metrics(y_true, predictions_s2_only)

    print("\n==================================================================")
    print("📊 BENCHMARK EVALUATION RESULTS (1,500 DOMAINS)")
    print("==================================================================")
    
    summary_table = pd.DataFrame([
        {"Method": "Solution 1 (Heuristic Engine)", **m_s1},
        {"Method": "Solution 2 (Probabilistic ML Engine)", **m_s2},
        {"Method": "Hybrid Pipeline (Solution 1 + 2)", **m_hybrid}
    ])
    print(summary_table.to_string(index=False))
    print("==================================================================\n")

    # Save detailed domain results with explicit reasons
    eval_df["predicted_is_shop"] = predictions_hybrid
    eval_df["classification_method"] = methods
    eval_df["data_source_used"] = data_sources
    eval_df["classification_reason"] = reasons

    os.makedirs("data/processed", exist_ok=True)
    eval_df.to_csv("data/processed/benchmark_results.csv", index=False)
    print("💾 Saved full itemized evaluation data to 'data/processed/benchmark_results.csv'")

if __name__ == "__main__":
    run_benchmark()