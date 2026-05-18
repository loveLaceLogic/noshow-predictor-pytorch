# 🏥 No-Show Predictor (PyTorch)

Predicting patient appointment no-shows using a real healthcare dataset and a PyTorch neural network, with evaluation focused on operational usefulness rather than raw accuracy.

---

## 🔍 Problem Statement
Missed appointments disrupt clinical workflows, waste staff time, and increase healthcare costs.  
This project explores whether patient and scheduling data can be used to predict appointment no-shows in advance.

---

## 📊 Dataset
- **110,527** medical appointments  
- **14 features** including:
  - Age
  - Gender
  - Medical conditions (diabetes, hypertension, alcoholism)
  - SMS reminders
  - Scheduling and appointment dates
- Binary target variable: **No-show**

> Dataset exhibits class imbalance (~80% show / 20% no-show), which informed evaluation strategy.

---

## 🧠 Model
- Feedforward neural network (MLP)
- Implemented in **PyTorch**
- Binary classification using **BCEWithLogitsLoss**
- Trained using **Apple Metal Performance Shaders (MPS)** on macOS

---

## 📈 Evaluation Strategy
Because no-shows are relatively rare, accuracy alone is misleading.

Primary evaluation metrics:
- **Recall** (minimizing missed no-shows)
- **Precision**
- **F1-score**

Threshold tuning was explored to balance false positives vs missed no-shows in a real healthcare scheduling context.

---

## 📁 Project Structure

### Key Files
- `dataset.py` – Data loading, preprocessing, encoding, and train/validation/test splitting  
- `model.py` – PyTorch neural network architecture  
- `train.py` – Training loop, loss tracking, and model persistence  
- `eval.py` – Model evaluation, metrics calculation, and confusion matrix generation  
- `eda.py` – Exploratory data analysis for class imbalance and target distribution

## Screenshots

### Web Interface
<img width="500" height="618" alt="web-form" src="https://github.com/user-attachments/assets/8209bb53-75fd-4b12-8874-2ef28c4e518f" />


### Prediction Result

<img width="400" height="576" alt="prediction-result" src="https://github.com/user-attachments/assets/c4bf3013-43a6-4465-88f0-48f7ff3e5fc2" />

### FastAPI Swagger Documentation

<img width="844" height="724" alt="fastapi-get-docs" src="https://github.com/user-attachments/assets/9ae64141-70a2-486e-8c3b-1d6148da0ea0" />

<img width="837" height="771" alt="fastapi-post-docs" src="https://github.com/user-attachments/assets/ada091ea-d435-47c4-9e47-3fb36e1da31e" />

### Running API Server

<img width="573" height="386" alt="serving-running" src="https://github.com/user-attachments/assets/77bdeb57-a7c5-482a-a437-6eae523c352f" />

