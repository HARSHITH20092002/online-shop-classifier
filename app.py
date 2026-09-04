import os
import sys
import time
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.scrapers.fetcher import fetch_website_content
from src.models.classifier import ShopClassifier

st.set_page_config(
    page_title="Online Shop Classifier",
    page_icon="🛍️",
    layout="wide"
)

st.sidebar.title("Navigation")
mode = st.sidebar.radio(
    "Select View:", 
    [
        "Single Domain Analysis", 
        "Batch Dataset Results", 
        "Model Benchmark Metrics (1,500 Domains)"
    ]
)

st.title("🛍️ Online Shop Classifier")
st.caption("Enterprise Market Discovery: Rule Heuristics + Probabilistic Scoring + Historical Recovery")
st.markdown("---")

# VIEW 1: SINGLE DOMAIN ANALYSIS
if mode == "Single Domain Analysis":
    st.subheader("Single Domain Inspection")
    
    url_input = st.text_input(
        "Enter Web Domain / URL:", 
        placeholder="e.g., temu.com, weldom.fr, facebook.com/marketplace"
    )
    
    if st.button("Classify Domain", type="primary"):
        if not url_input.strip():
            st.warning("Please enter a valid domain or URL.")
        else:
            target_url = url_input.strip()
            if not target_url.startswith("http://") and not target_url.startswith("https://"):
                target_url = "https://" + target_url

            start_t = time.time()
            with st.spinner(f"Analyzing `{target_url}`..."):
                fetched = fetch_website_content(target_url)
                classifier = ShopClassifier()
                pred = classifier.predict(fetched)
                elapsed = time.time() - start_t

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if pred.get("is_shop", False):
                    st.success(f"**Result: {pred.get('result', 'SHOP')}**")
                else:
                    st.error(f"**Result: {pred.get('result', 'NOT A SHOP')}**")
            with col2:
                conf = pred.get("confidence", 0.0) * 100
                st.metric("Confidence Score", f"{conf:.0f}%")
            with col3:
                st.metric("Data Source", fetched.get("data_source", "Live DOM"))
            with col4:
                st.metric("Latency", f"{elapsed:.2f}s")

            st.markdown("---")
            st.markdown(f"**Decision Reason:** `{pred.get('reason', 'N/A')}`")
            
            st.subheader("Inspection Metadata")
            st.json({
                "domain": target_url,
                "data_source": fetched.get("data_source"),
                "is_reachable": fetched.get("is_reachable", False),
                "http_status_code": fetched.get("status_code", 0),
                "method_executed": pred.get("method"),
                "classification_reason": pred.get("reason"),
                "confidence_score": pred.get("confidence")
            })

# VIEW 2: BATCH DATASET RESULTS
elif mode == "Batch Dataset Results":
    st.subheader("Batch Classification Records")
    
    p_dir = os.path.join("data", "processed")
    files = [f for f in os.listdir(p_dir) if f.endswith(".csv")] if os.path.exists(p_dir) else []

    if files:
        files.sort(key=lambda x: 0 if "ALL" in x else 1)
        selected_file = st.selectbox("📂 Select Processed File to Inspect:", files, index=0)
        csv_path = os.path.join(p_dir, selected_file)
        df = pd.read_csv(csv_path)
        
        if "dataset_examined" in df.columns:
            st.info(f"**Examined Scope:** `{df['dataset_examined'].iloc[0]}` | **File Loaded:** `{selected_file}`")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Domains", len(df))
        
        if "is_shop" in df.columns:
            if df["is_shop"].dtype == object:
                df["is_shop"] = df["is_shop"].astype(str).str.lower().map({'true': True, 'false': False})
            shops = int(df["is_shop"].sum())
        elif "result" in df.columns:
            shops = len(df[df["result"] == "SHOP"])
        else:
            shops = 0
            
        c2.metric("Shops Detected", shops)
        c3.metric("Non-Shops / Excluded", len(df) - shops)

        st.markdown("---")
        st.dataframe(df, use_container_width=True)
        
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Selected Classification CSV",
            data=csv_data,
            file_name=selected_file,
            mime="text/csv"
        )
    else:
        st.info("No processed batches found in `data/processed/`. Run `python main.py` or `python benchmark.py` in your terminal.")

# VIEW 3: MODEL BENCHMARK METRICS (1,500 DOMAINS)
elif mode == "Model Benchmark Metrics (1,500 Domains)":
    st.subheader("📊 Comparative Performance Evaluation")
    bm_path = os.path.join("data", "processed", "benchmark_results.csv")

    if os.path.exists(bm_path):
        bdf = pd.read_csv(bm_path)
        
        y_true = bdf["ground_truth"].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False})
        y_pred = bdf["predicted_is_shop"].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False})

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[True, False])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Overall Accuracy", f"{acc*100:.2f}%")
        col2.metric("Precision", f"{prec*100:.2f}%")
        col3.metric("Recall (Sensitivity)", f"{rec*100:.2f}%")
        col4.metric("F1-Score", f"{f1*100:.2f}%")

        st.markdown("---")
        st.markdown("#### Confusion Matrix Breakdown")
        cm_df = pd.DataFrame(
            cm, 
            index=["Actual: SHOP (1)", "Actual: NOT A SHOP (0)"], 
            columns=["Predicted: SHOP (1)", "Predicted: NOT A SHOP (0)"]
        )
        st.table(cm_df)

        st.markdown("#### Itemized Evaluation Records with Decision Reasons")
        display_cols = [c for c in ["domain", "ground_truth", "predicted_is_shop", "classification_method", "data_source_used", "classification_reason"] if c in bdf.columns]
        st.dataframe(bdf[display_cols], use_container_width=True)
    else:
        st.info("Benchmark file not generated yet. Run `python benchmark.py` in your terminal to evaluate 1,500 domains.")