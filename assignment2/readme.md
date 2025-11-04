# Assignment 2: Feedforward and Recurrent Neural Networks

This repository contains the implementation of **Assignment 2** for **CS6320: Natural Language Processing (Fall 2025)** at the **University of Texas at Dallas**.

The goal of this assignment is to implement and compare two neural architectures — a **Feedforward Neural Network (FFNN)** and a **Recurrent Neural Network (RNN)** — for a text classification task using tokenized JSON datasets and pre-trained word embeddings.

---

## 🧠 Project Overview

The project consists of two independent PyTorch models:

- **`ffnn_gpu.py`** — Implements a Feedforward Neural Network with ReLU activation and log-softmax output.  
  The model is fully connected and trained on GPU.

- **`rnn_gpu.py`** — Implements a vanilla Recurrent Neural Network trained on GPU.  
  Sequential token dependencies are explicitly modeled through hidden state propagation.

Both models include:
- **Negative Log-Likelihood Loss (NLLLoss)** for classification  
- **LogSoftmax activation** at the output layer  
- **Automatic checkpoint saving** for the best validation performance  

---

## 📂 Repository Structure

```

assignment2/
│
├── ffnn_gpu.py               # Feedforward Neural Network (FFNN) implementation
├── rnn_gpu.py                # Recurrent Neural Network (RNN) implementation
├── training.json              # Tokenized training dataset
├── validation.json             # Validation dataset
├── test.json                    # Test dataset
├── word_embedding.pkl    # Pre-trained word embeddings
└── README.md               # Documentation file (this file)

````

---

## ⚙️ Environment Setup

Both models were implemented and tested using **PyTorch** and **NumPy**.  
The following commands set up the required environment:

```bash
pip install torch tqdm numpy
````

GPU acceleration (CUDA) is automatically used if available.
The code runs seamlessly on **Google Colab (GPU runtime)** or any local GPU setup.

---

## 🚀 How to Run the Models

### 1️⃣ Train the Feedforward Neural Network (FFNN)

```bash
python ffnn_gpu.py \
  --hidden_dim 128 \
  --epochs 16 \
  --base_dir "./assignment2" \
  --do_train
```

### 2️⃣ Train the Recurrent Neural Network (RNN)

```bash
python rnn_gpu.py \
  --hidden_dim 128 \
  --epochs 16 \
  --base_dir "./assignment2" \
  --do_train
```

After training:

* The best-performing model is automatically saved as:

  ```
  best_model_ffnn_128.pt
  best_model_rnn_128.pt
  ```
* A summary file named `results.txt` is created, recording:

  * Hidden dimension
  * Training Accuracy
  * Validation Accuracy
  * Test Accuracy

Example output snippet:

```
FFNN
hidden_dim : 128
Training Accuracy : 0.7561
Validation Accuracy : 0.5750
Test Accuracy : 0.4575
```

---

## 💾 Word Embeddings

The file **`word_embedding.pkl`** contains pre-trained word embeddings.
These vectors are automatically loaded by both models during initialization.
Make sure the file resides in the same directory as the `.py` scripts when executing the code.

---

## 📊 Evaluation Protocol

Each model is evaluated based on:

* **Training Accuracy**
* **Validation Accuracy**
* **Test Accuracy**

The model achieving the **highest validation accuracy** is selected as the final model for testing.

---

## 🧩 Implementation Notes

* Developed and tested using **PyTorch 3.x** on **Google Colab GPU runtime**.
* Compatible with both **CPU** and **CUDA** execution environments.
* JSON files (`training.json`, `validation.json`, `test.json`) must be structured consistently with the embedding indices.
* The scripts automatically detect and utilize GPU if `torch.cuda.is_available()` is `True`.

---

## 🧪 Experimental Insights

* FFNN performs well for shorter, fixed-length representations.
* RNN captures sequential dependencies better but is computationally heavier.
* **FFNN** tends to show **overfitting** behavior, especially when the hidden layer dimension is large.  
  It quickly adapts to the training data but exhibits a noticeable drop in validation accuracy.  
  Reducing the hidden size or introducing dropout can help regularize the model.
* **RNN** exhibits signs of **underfitting** and **training instability**, particularly in the early epochs.  
  Gradient vanishing occasionally slows convergence, and validation accuracy improves slowly over time.  
  Increasing the hidden dimension or using gated variants (e.g., LSTM, GRU) may alleviate this issue.
* Overall, FFNN converges faster but generalizes less effectively, whereas RNN learns more gradually and struggles with stability under limited training data.

---

## 🧰 Potential Extensions

* Replace vanilla RNN with **LSTM** or **GRU** for longer sequence handling.
* Integrate **embedding fine-tuning** during training.
* Add **dropout regularization** for better generalization.

---

## 👤 Authors

**Group 4**

- Dhruv Jaiprakash Shelke  
- Sidhartha Pulluri  
- Smriti Sunil  
- Sungbum Kim

---

## 🏫 Course Information

**Course:** CS6320 - Natural Language Processing
**Assignment:** #2 (Feedforward and Recurrent Neural Networks)
**Semester:** Fall 2025
