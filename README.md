# Spam Detection

A dual-method approach for spam/phishing email detection using both modern transformer-based models and traditional machine learning techniques. This project demonstrates how to process raw email data, train models, and deploy them in ONNX format.

---

## Table of Contents

- [Overview](#overview)
- [Methods & Literature Review](#methods--literature-review)
- [Implementation – Programming](#implementation--programming)
- [Dataset](#dataset)
- [Evaluation Metrics](#evaluation-metrics)
- [Implementation Details](#implementation-details)
  - [Tokenization and Vectorization](#tokenization-and-vectorization)
  - [Training](#training)
  - [ONNX Conversion and Testing](#onnx-conversion-and-testing)
- [Results](#results)
- [Code Examples](#code-examples)
- [Project Structure](#project-structure)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

This repository presents two methods for detecting phishing emails and spam:

1. **Transformer-based Classification:**  
   A DistilBERT transformer is trained for sequence classification to distinguish between phishing and safe emails. The email text is preprocessed, tokenized, and fed into the model. Early stopping is employed during training and the model is later exported to ONNX format for efficient inference.

2. **Traditional Machine Learning Text Classification:**  
   Email texts are vectorized using TF-IDF, and classification is performed using Random Forest, Naïve Bayes, and Logistic Regression models. The best performing model is also exported to ONNX format.

---

## Methods & Literature Review

### Method 1: Transformer-based Classification
- **Model:** DistilBERT for sequence classification
- **Preprocessing:** Email text is preprocessed and tokenized before being passed to the model.
- **Training:**  
  - Early stopping and scheduled learning rate decay are used.
  - The model is optimized and later exported to ONNX format for inference.

### Method 2: Traditional ML Text Classification
- **Model Types:**  
  - TF-IDF vectorization combined with Random Forest, Naïve Bayes, and Logistic Regression.
- **Preprocessing:**  
  Email texts are vectorized and used to train the model.
- **Export:**  
  The final model is exported in ONNX format.

---

## Implementation – Programming

- **Programming Language:** Python  
- **Development Platform:** Google Colab  
- **Libraries Used:**
  - `transformers`
  - `sklearn`
  - `onnx`
  - `train_test_split` (from `sklearn.model_selection`)
  - Additional libraries: `pandas`, `numpy`, `re`, etc.

---

## Dataset

- **Name:** [subhajournal/phishingemails](https://www.kaggle.com/datasets/subhajournal/phishingemails)
- **Samples:**  
  - **Original:** 18,650 emails  
  - **After Cleaning:** 15,895 emails
- **Distribution:**  
  - **Safe Emails:** ~61%  
  - **Phishing Emails:** ~39%
- **Features:**  
  - **Email Text:** The raw content of the email.
  - **Email Type:** Label indicating if the email is "Safe Email" or "Phishing Email".

---

## Evaluation Metrics

- **Training Metrics:**  
  - **F1 Score**

- **Post-Evaluation Metrics:**  
  - Training Time  
  - Prediction Time  

- **Other Metrics:**  
  - Accuracy  
  - Log Loss  
  - Confusion Matrix  
  - Precision  
  - Recall  
  - ROC AUC

---

## Implementation Details

### 🔤 Tokenization and Vectorization

- **Transformer Model:**  
  Uses the DistilBERT tokenizer to convert email text into tokens.
- **Traditional Models:**  
  Applies TF-IDF vectorization to convert text into a numerical representation.  
  *Example:*  
  - If the term "data" appears 20 times in a 100-word document, TF = 0.2.  
  - If "data" appears in 100 out of 10,000 documents, then the IDF is calculated accordingly.

### 🏋️‍♂️ Training

- **Transformer-Based Model:**  
  Trained with early stopping and a linear learning rate schedule.
- **Traditional Models:**  
  Trained using standard `scikit-learn` pipelines. Training times and F1 scores are recorded for evaluation.

### ↪️ ONNX Conversion and Testing

Both approaches have been exported to the ONNX format to:
- Improve inference speed.
- Allow for easier deployment in a production environment.

Exported models are tested with ONNX Runtime to ensure consistency with the original models.

---

## 📄 Results

| Model                         | F1 Score (Approx.) | Training Time (s) | Prediction Time (s) |
|-------------------------------|--------------------|-------------------|---------------------|
| **BERT Transformer Model**    | 0.9839 (GPU)       | 1122.24 (GPU)     | 0.04                |
| **Naïve Bayes – Multinomial** | 0.9571             | 0.08              | 0.0004              |
| **Logistic Regression**       | 0.9655             | 1.09              | 0.0002              |
| **Random Forest**             | 0.9522             | 26.95             | 0.0007              |

---

## Code Examples

### 📌 Traditional ML Implementation (Excerpt)

```python
import pandas as pd
import numpy as np
import re
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = pd.read_csv("Phishing_Email.csv")

# Data cleaning and preprocessing...
# Vectorize email text using TF-IDF
vectorizer = TfidfVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(df['Email Text'])
# Continue with train_test_split, model training, and evaluation

rf_model = RandomForestClassifier(random_state=123, n_jobs=-1)
start_time = time.time()
rf_model.fit(X_train, y_train)
print(f"Training Time: {time.time() - start_time:.2f} seconds")
```

### 📌 Transformer-Based Implementation (Excerpt)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import datasets

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenization and dataset preparation...
tokenized_dataset = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="save_model",
    num_train_epochs=10,
    per_device_train_batch_size=32,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    fp16=torch.cuda.is_available()
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['valid'],
    compute_metrics=compute_metrics,
)

trainer.train()
```

---

## 📓 Notebooks

For full code details, please see the notebooks included:

- [`Traditional NLP U599.ipynb`](notebooks/Traditional%20NLP%20U599.ipynb)  
  Traditional machine learning approach using TF-IDF and classical classifiers.

- [`Transformer-based NLP U599.ipynb`](notebooks/Transformer-based%20NLP%20U599.ipynb)  
  Modern approach using DistilBERT for email classification.

---

## 📁 Project Structure

```bash
├── README.md
├── .github
│   └── workflows
│       └── clean_widgets.yml               # GitHub Actions workflow for cleaning notebook metadata
├── notebooks
│   ├── Traditional NLP U599.ipynb          # Traditional ML approach notebook
│   └── Transformer-based NLP U599.ipynb    # Transformer-based approach notebook
├── onnx_models
│   └── random_forest_model.onnx            # Exported ONNX model for traditional ML
└── onnx_utils
    └── vectorizer.pkl                      # TF-IDF vectorizer for traditional ML
```

---

## 📜 License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for more details.

---

## 🙏 Acknowledgements

Thanks to the [subhajournal/phishingemails](https://www.kaggle.com/datasets/subhajournal/phishingemails) dataset for supporting this project.

Appreciation to the research community and open-source library maintainers that power projects like this — including libraries such as `transformers`, `scikit-learn`, and `onnx`.
