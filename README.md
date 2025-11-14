# 📊 Market Opportunity Analyzer

This project implements an advanced, end-to-end Machine Learning pipeline and a Streamlit dashboard designed to transform raw, unstructured customer feedback (reviews, comments, etc.) into quantifiable business opportunities.

It automatically identifies customer pain points, calculates the severity of those pains, and ranks them by actionable opportunity score.

---

## Core Value Proposition

This tool solves two major business problems:

### **For Existing Companies (e.g., Slack)**
Pinpoints critical product flaws (like **broken notifications** or **slow search**) that are leading to customer churn, allowing for high-impact resource allocation.

### **For Entrepreneurs & Startups**
Analyzes competitor review data to identify underserved market gaps (the “opportunity”), providing evidence for a new product’s core value proposition.

---

## System Architecture and Technology Stack

The analyzer is built as a modular pipeline (`src/`) that executes advanced clustering techniques and is presented via a highly interactive Streamlit application (`app/app.py`).

---

### 1. **Advanced Topic Modeling (The Core Engine)**

Instead of using simple keyword-counting methods, this project uses a modern clustering stack for superior context and accuracy:

- **SBERT (Sentence Transformers)**  
  Generates dense vector embeddings representing contextual meaning.

- **UMAP (Uniform Manifold Approximation and Projection)**  
  Reduces dimensionality while preserving important structure.

- **HDBSCAN (Hierarchical Density-Based Spatial Clustering)**  
  Automatically discovers natural clusters (topics) within feedback.

---

### 2. **Multi-Model Evaluation & Scoring**

| Model | Purpose | Role in Pipeline |
|-------|---------|------------------|
| **VADER** | Sentiment Analysis | Used for the pipeline’s final scoring due to its speed and effectiveness on short text. |
| **BERT (bert_sentiment)** | Benchmarking / Gold Standard | Used outside the pipeline to benchmark VADER’s accuracy. |

#### **Scoring Algorithm**
**Final Ranking:**  
Opportunity Score = *Mention Count × Negative Sentiment Fraction*

---

## Project Features & Structure

### **App Features (`app/app.py`)**

- **Live Analysis:**  
  Upload a CSV/TSV and execute the full SBERT → UMAP → HDBSCAN pipeline in-browser with Streamlit caching.

- **Interactive Dashboard:**  
  Shows ranked topics, opportunity scores, and negative sentiment fractions.

- **AI Analyst Chatbot:**  
  Uses OpenAI GPT-4o-mini to answer questions about pain points and opportunity scores.

---

## ✅ Quality & Robustness

This project was built with production readiness in mind:

### **Unit Testing**
Includes a pytest suite ensuring ETL and text normalization handle messy real-world data.

### **Benchmarking**
Shows why VADER was chosen over BERT for performance reasons—demonstrating engineering trade-offs.

### **Scalability**
Architecture supports large datasets and easy model upgrades for higher accuracy.

---
