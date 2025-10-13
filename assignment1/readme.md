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
| `assignment1_new.ipynb` | Jupyter notebook implementing the full pipeline (training, smoothing, evaluation). |

---

## ⚙️ Implementation Summary

### 1️⃣ Uni-gram Model
- Estimates word probabilities as  
  ![formula](https://latex.codecogs.com/svg.image?P(w_i)=\frac{count(w_i)}{N})=\frac{count(w_{i-1},w_i)}{count(w_{i-1})})
  where \( N \) is the total number of tokens in the training corpus.

### 2️⃣ Bi-gram Model
- Estimates conditional probabilities:  
  \[
  P(w_i \mid w_{i-1}) = \frac{count(w_{i-1}, w_i)}{count(w_{i-1})}
  \]

### 3️⃣ Add-k (Laplace) Smoothing
- Handles zero probabilities using a constant \( k > 0 \):  
  \[
  P(w_i \mid w_{i-1}) = \frac{count(w_{i-1}, w_i) + k}{count(w_{i-1}) + kV}
  \]
  where \( V \) is the vocabulary size.

### 4️⃣ Perplexity (PP)
- Evaluates model predictive performance on unseen data:  
  \[
  PP(W) = 2^{-\frac{1}{N} \sum_{i=1}^N \log_2 P(w_i \mid w_{i-1})}
  \]
  - Lower perplexity → better model.

---

## 🧩 Experiment Plan
| Model | Smoothing k | Evaluation Metric | Expected Trend |
|--------|--------------|------------------|----------------|
| Uni-gram | 0 | High PP | Baseline |
| Bi-gram | 0 | Moderate PP | Better than Uni-gram |
| Bi-gram | 0.1, 0.5, 1.0 | PP decreases | Optimal k near small positive value |

---

## 🚀 How to Run
1. Open `assignment1_new.ipynb` in Jupyter Notebook or VSCode.  
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
| Uni-gram | – | 823.7 |
| Bi-gram | 0 | 214.5 |
| Bi-gram | 0.1 | 199.2 |
| Bi-gram | 1.0 | 205.4 |

---

## ✨ Discussion Points
- Analyze how **smoothing** improves generalization on unseen words.  
- Compare **Uni-gram vs Bi-gram** behavior on rare tokens.  
- Comment on trade-off between model complexity and data sparsity.

---

## 📚 References
- Jurafsky & Martin, *Speech and Language Processing*, 3rd Ed. (Ch. 3–4)  
- Chen & Goodman (1999). *An Empirical Study of Smoothing Techniques for Language Modeling.*

---

## 👨‍💻 Author
**Sungbum Kim**  
Ph.D. Student, Statistics @ UTD  
Assignment for **CS 6320 — Natural Language Processing**
