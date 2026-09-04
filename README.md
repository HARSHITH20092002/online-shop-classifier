# Online Shop Classifier

So this is an intelligent dual way system that tries to discover , then fetch, and later classify web domains into **Shop** vs. **Non-Shop**. It kinda mixes heuristic web scraping rules with machine learning too, specifically TF-IDF plus Logistic Regression, in a hybrid manner.

---

## Key Features

- **Dual-Engine Architecture:**
  - **Solution 1 (Rule-Based Heuristic Engine):** It goes through scraped page text and looks for e commerce signals like cart related words, checkout paths , payment provider clues, and those product price patterns that show up often.
  - **Solution 2 (Machine Learning Engine):** It takes TF-IDF n-grams from scraped page metadata, the actual body text, and also from structural pieces. After that it runs a Logistic Regression decision model so the results feel more stable.

- **Interactive Web Dashboard:** Built with Streamlit, mainly for single-domain checks, plus a live confidence visualizer so you can see what the system is “thinking” .  

- **Batch Benchmarking Suite:** There is an automated evaluation script that measures model accuracy, latency, and even the breakdown of decisions, across 50+ domains pulled from GfK market discovery Parquet datasets.

---

## Project Architecture & Directory Structure

```text
online-shop-classifier/
├── data/                         # Dataset directory (.gitignore tracked)
│   └── processed/                # Saved benchmark outputs and metrics
├── src/
│   ├── models/
│   │   ├── classifier.py         # Dual-solution prediction engine
│   │   ├── train.py              # ML model training script
│   │   ├── logistic_model.pkl    # Serialized ML classifier
│   │   └── tfidf_vectorizer.pkl  # Serialized vectorizer
│   └── scrapers/
│       └── fetcher.py            # Web scraper with timeout & user-agent handling
├── app.py                        # Streamlit interactive web dashboard
├── benchmark.py                  # Batch evaluation on 50 domains
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
