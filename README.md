![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-ML-red)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Healthcare](https://img.shields.io/badge/Healthcare-Analytics-purple)
![Conda](https://img.shields.io/badge/Conda-Environment-success)

# 🏥 No-Show Predictor (PyTorch)

Predicting patient appointment no-shows using a real healthcare dataset and a PyTorch neural network, with evaluation focused on operational usefulness rather than raw accuracy.

---

## ⭐ Project Highlights

- 🏥 Trained on **110,527** real healthcare appointments
- 🧠 Built a **PyTorch neural network (MLP)** for binary classification
- ⚡ Developed a **FastAPI REST API** for serving predictions
- 🌐 Created a simple **HTML frontend** for interactive predictions
- 📊 Evaluated performance using **Precision, Recall, and F1-score**
- 🍎 Optimized training with **Apple Silicon (MPS)** acceleration
- 📦 Supports both **Conda** and **pip** environments
- 🔄 Includes a **Jenkins CI pipeline** for automated builds
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

## 🏗️ System Architecture

```mermaid
flowchart TD

A[Healthcare Appointment Dataset] --> B[Data Preprocessing]
B --> C[PyTorch Neural Network Training]
C --> D[Saved Model (.pt)]

D --> E[FastAPI REST API]
E --> F[HTML Frontend]
F --> G[Patient No-Show Prediction]

D --> H[Model Evaluation]
H --> I[Precision • Recall • F1 Score]
```
## 📸 Screenshots

### Web Interface
<img width="500" alt="web-interface" src="https://github.com/user-attachments/assets/8209bb53-75fd-4b12-8874-2ef28c4e518f" />

---

### Prediction Result
<img width="400" alt="prediction-result" src="https://github.com/user-attachments/assets/c4bf3013-43a6-4465-88f0-48f7ff3e5fc2" />

---

### FastAPI Swagger Documentation

#### GET Endpoint
<img width="844" alt="fastapi-get-docs" src="https://github.com/user-attachments/assets/9ae64141-70a2-486e-8c3b-1d6148da0ea0" />

#### POST Endpoint
<img width="837" alt="fastapi-post-docs" src="https://github.com/user-attachments/assets/ada091ea-d435-47c4-9e47-3fb36e1da31e" />

---

### Running API Server
<img width="573" alt="server-running" src="https://github.com/user-attachments/assets/77bdeb57-a7c5-482a-a437-6eae523c352f" />


---

## ⚙️ Technologies Used

- Python
- PyTorch
- FastAPI
- HTML/CSS/JavaScript
- Jenkins (CI/CD pipeline setup)
- Git & GitHub
- Pandas
- Scikit-learn
- Matplotlib

---

## 🚀 Future Improvements

- Deploy API to cloud hosting
- Integrate trained PyTorch model into live predictions
- Expand frontend styling and validation
- Add automated Jenkins testing pipeline
- Containerize application using Docker

---
### Jenkins CI Pipeline

<img width="717" height="762" alt="jenkins-success" src="https://github.com/user-attachments/assets/a4463d32-0ab6-46c4-8731-3d61c15f6732" />


