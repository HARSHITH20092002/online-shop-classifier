import os
import sys
import pandas as pd
from src.scrapers.fetcher import fetch_website_content
from src.models.classifier import ShopClassifier

TARGET_DATASET = "ALL"  # Set to 'ALL' or a specific file (e.g. 'FR_online_market_discovery_2024.parquet')
ROW_LIMIT = 10         # Limit rows per dataset for rapid testing; set to None for full batches

def run_pipeline():
    classifier = ShopClassifier()
    data_dir = "data"
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    is_all_mode = TARGET_DATASET.strip().upper() == "ALL"

    if is_all_mode:
        files = [
            f for f in os.listdir(data_dir) 
            if f.endswith(('.parquet', '.csv')) and os.path.isfile(os.path.join(data_dir, f))
        ]
        dataset_label = "ALL"
    else:
        files = [TARGET_DATASET]
        dataset_label = TARGET_DATASET

    if not files:
        print("⚠️ No input files found in 'data/' directory.", flush=True)
        return

    all_results = []
    print("\n==================================================", flush=True)
    print("--- Online Shop Classification Pipeline ---", flush=True)
    print(f"--- Scope          : {dataset_label}", flush=True)
    print(f"--- Datasets Found : {len(files)} file(s)", flush=True)
    print("==================================================\n", flush=True)

    for f_idx, file_name in enumerate(files, 1):
        file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path):
            print(f"⚠️ [{f_idx}/{len(files)}] File '{file_path}' not found. Skipping...", flush=True)
            continue

        print(f">>> File {f_idx}/{len(files)}: Processing '{file_name}'...", flush=True)
        
        df = pd.read_parquet(file_path) if file_name.endswith(".parquet") else pd.read_csv(file_path)
        domain_col = next((col for col in ["domain", "url", "website", "Root Domain"] if col in df.columns), df.columns[0])
        domains = df[domain_col].dropna().unique()

        if ROW_LIMIT is not None:
            domains = domains[:ROW_LIMIT]

        total_domains = len(domains)
        print(f"    Evaluating {total_domains} unique domains...", flush=True)

        for d_idx, raw_domain in enumerate(domains, 1):
            url = str(raw_domain) if str(raw_domain).startswith("http") else f"https://{raw_domain}"
            print(f"  [{d_idx}/{total_domains}] Fetching: {url} ...", end="", flush=True)

            fetched = fetch_website_content(url)
            prediction = classifier.predict(fetched)

            print(f" -> {prediction['result']} ({prediction['confidence']*100:.0f}% - {prediction['method']})", flush=True)

            all_results.append({
                "dataset_examined": dataset_label,
                "source_file": file_name,
                "domain": raw_domain,
                "url": url,
                "is_shop": prediction["is_shop"],
                "result": prediction["result"],
                "confidence": prediction["confidence"],
                "method": prediction["method"],
                "classification_reason": prediction["reason"]
            })
        print("", flush=True)

    output_name = "classification_results_ALL.csv" if is_all_mode else f"classification_results_{TARGET_DATASET.split('.')[0]}.csv"
    output_path = os.path.join(processed_dir, output_name)

    output_df = pd.DataFrame(all_results)
    output_df.to_csv(output_path, index=False)

    print("==================================================", flush=True)
    print("  Pipeline Execution Complete:")
    print(f"  - Scope Examined   : {dataset_label}")
    print(f"  - Total Evaluated  : {len(output_df)} domains")
    print(f"  - Output Exported  : {output_path}")
    print("==================================================\n", flush=True)

if __name__ == "__main__":
    run_pipeline()