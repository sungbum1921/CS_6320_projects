# Assignment 1 — N-gram Language Models

## 📘 Overview
This assignment implements **Uni-gram** and **Bi-gram** language models to analyze text corpora and estimate sequence probabilities.  
You will also experiment with **add-k smoothing** and evaluate model quality using **Perplexity (PP)** on held-out data.

---

## 🧠 Objectives
1. Implement probabilistic language models based on **n-gram frequency counts**.  
2. Apply **smoothing techniques** to handle unseen word combinations.  
3. Compute **sentence probabilities** and **perplexity scores** on validation data.  
4. Compare model performance across different smoothing constants and n-gram orders.

---

## 📂 Files Included
| File | Description |
|------|--------------|
| `train.txt` | Training corpus used to build n-gram counts. |
| `val.txt` | Validation corpus used to compute perplexity. |
| `assignment1.ipynb` | Jupyter notebook implementing the full pipeline (training, smoothing, evaluation). |

---

## ⚙️ Implementation Summary

### 1️⃣ Uni-gram Model
Estimates word probabilities as:  
> **P(wᵢ) = count(wᵢ) / N**  
> where **N** is the total number of tokens in the training corpus.

---

### 2️⃣ Bi-gram Model
Estimates conditional probabilities as:  
> **P(wᵢ | wᵢ₋₁) = count(wᵢ₋₁, wᵢ) / count(wᵢ₋₁)**  

This measures how likely word *wᵢ* follows *wᵢ₋₁* based on observed pairs.

---

### 3️⃣ Add-k (Laplace) Smoothing
Handles zero probabilities using a constant *k > 0*:  
> **P(wᵢ | wᵢ₋₁) = (count(wᵢ₋₁, wᵢ) + k) / (count(wᵢ₋₁) + kV)**  
> where **V** is the vocabulary size.

Smoothing ensures that unseen word pairs receive small non-zero probabilities,  
preventing total probability collapse for unseen events.

---

### 4️⃣ Perplexity (PP)
Evaluates model predictive performance on unseen data:  
> **PP(W) = 2^(-1/N × Σ log₂ P(wᵢ | wᵢ₋₁))**  

Lower perplexity → better model (indicating higher average likelihood on test data).

---

## 🧩 Experiment Plan
| Model | Smoothing k | Evaluation Metric | Expected Trend |
|--------|--------------|------------------|----------------|
| Uni-gram | 0 | High PP | Baseline |
| Uni-gram | 0.1, 0.01, 0.001 | Moderate PP | Smaller PP than baseline |
| Bi-gram | 0 | Overfitted PP | Better than Uni-gram |
| Bi-gram | 0.1, 0.01, 0.001 | PP decreases | Optimal k near small positive value |

---

## 🚀 How to Run
1. Open `assignment1.ipynb` in Jupyter Notebook or VSCode.  
2. Run all cells in order.  
3. The notebook will:
   - Build n-gram counts from `train.txt`
   - Apply smoothing
   - Compute perplexity on `val.txt`
4. The output will show tables comparing PP across settings.

---

## 🧾 Example Output
| Model | k | Perplexity |
|--------|---|-------------|
| Uni-gram | – | 293.23 |
| Bi-gram | 0 | 2156.58 |
| Bi-gram | 0.1 | 185.06 |
| Bi-gram | 1.0 | 429.08 |

---

## ✨ Discussion Points
- Analyze how **smoothing** improves generalization on unseen words.  
- Compare **Uni-gram vs Bi-gram** behavior on rare tokens.  
- Comment on trade-off between model complexity and data sparsity.

---
