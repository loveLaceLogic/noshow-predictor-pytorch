![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-ML-red)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Healthcare](https://img.shields.io/badge/Healthcare-Analytics-purple)
![Conda](https://img.shields.io/badge/Conda-Environment-success)

# 🏥 No-Show Predictor (PyTorch)

Predicting patient appointment no-shows using a real healthcare dataset and a PyTorch neural network, with evaluation focused on operational usefulness rather than raw accuracy.
---

## ⭐ Project Highlights

* 🏥 Trained on **110,527** real healthcare appointments
* 🧠 Built a **PyTorch neural network (MLP)** for binary classification
* ⚡ Developed a **FastAPI REST API** for serving predictions
* 🌐 Created a lightweight **HTML frontend** for interactive predictions
* 📊 Evaluated performance using **Precision, Recall, and F1-score**
* 🍎 Optimized training with **Apple Silicon (MPS)** acceleration
* 📦 Supports both **Conda** and **pip** environments
* 🔄 Configured a **Jenkins CI pipeline** for automated builds

---

## 🔍 Problem Statement

Missed appointments disrupt clinical workflows, waste staff time, and increase healthcare costs.

This project explores whether patient demographic and scheduling information can be used to predict appointment no-shows before a scheduled visit, helping healthcare organizations better allocate resources and improve operational efficiency.

---

## 📊 Dataset

* **110,527** healthcare appointments
* **14 predictive features**, including:

  * Age
  * Gender
  * Diabetes
  * Hypertension
  * Alcoholism
  * Handicap
  * SMS reminders
  * Scheduling and appointment dates

**Target Variable**

* Binary classification:

  * Show
  * No-show

> The dataset is naturally imbalanced (~80% show vs ~20% no-show), making Precision, Recall, and F1-score more meaningful than overall accuracy.

---

## 🧠 Model

* Feedforward Neural Network (MLP)
* Implemented using **PyTorch**
* Binary classification with **BCEWithLogitsLoss**
* Trained using **Apple Metal Performance Shaders (MPS)** on macOS

---

## 📈 Evaluation Strategy

Because appointment no-shows are relatively uncommon, overall accuracy alone provides limited insight into model performance.

The project emphasizes:

* Precision
* Recall
* F1-score

Threshold tuning was explored to balance false positives against missed no-show predictions in a realistic healthcare scheduling environment.

---

## 📈 Results

The model successfully learned scheduling patterns from more than **110,000** real healthcare appointments.

Rather than optimizing solely for accuracy, evaluation focused on operationally meaningful metrics suitable for imbalanced healthcare datasets.

The trained model was exported for inference through a FastAPI REST API and integrated with a lightweight HTML frontend for interactive prediction.

---

## 📁 Repository Structure

### Key Files

* `dataset.py` — Data loading, preprocessing, encoding, and train/validation/test splitting
* `model.py` — PyTorch neural network architecture
* `train.py` — Training loop, loss tracking, and model persistence
* `eval.py` — Model evaluation, metrics calculation, and confusion matrix generation
* `eda.py` — Exploratory data analysis for class imbalance and target distribution

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    A["Healthcare Appointment Dataset"] --> B["Data Preprocessing"]
    B --> C["PyTorch Neural Network Training"]
    C --> D["Trained PyTorch Model (.pt)"]

    D --> E["FastAPI REST API"]
    E --> F["HTML Frontend"]
    F --> G["Patient No-Show Prediction"]

    D --> H["Model Evaluation"]
    H --> I["Precision, Recall, F1 Score"]
```
## 📸 Screenshots

### Patient Prediction Interface

Interactive HTML application allowing users to enter patient information and receive a predicted appointment no-show risk.

<img width="500" alt="web-interface" src="https://github.com/user-attachments/assets/8209bb53-75fd-4b12-8874-2ef28c4e518f" />

---

### Prediction Output

Example prediction returned by the trained PyTorch model through the FastAPI backend.

<img width="400" alt="prediction-result" src="https://github.com/user-attachments/assets/c4bf3013-43a6-4465-88f0-48f7ff3e5fc2" />

---

### FastAPI API Documentation

#### GET Endpoint

Health check endpoint used to verify API availability.

<img width="844" alt="fastapi-get-docs" src="https://github.com/user-attachments/assets/9ae64141-70a2-486e-8c3b-1d6148da0ea0" />

#### POST Endpoint

Prediction endpoint accepting patient information and returning a no-show prediction.

<img width="837" alt="fastapi-post-docs" src="https://github.com/user-attachments/assets/ada091ea-d435-47c4-9e47-3fb36e1da31e" />

---

### Running API Server

FastAPI development server running locally with automatic reload enabled.

<img width="573" alt="server-running" src="https://github.com/user-attachments/assets/77bdeb57-a7c5-482a-a437-6eae523c352f" />

---

## 🔄 Continuous Integration (Jenkins)

A Jenkins pipeline was configured during development to automate project builds and demonstrate familiarity with continuous integration workflows.

<img width="717" height="762" alt="jenkins-success" src="https://github.com/user-attachments/assets/a4463d32-0ab6-46c4-8731-3d61c15f6732" />

---

## ⚙️ Technologies Used

* Python
* PyTorch
* FastAPI
* HTML
* CSS
* JavaScript
* Pandas
* Scikit-learn
* Matplotlib
* Jenkins
* Git
* GitHub

---

## 💡 What I Learned

This project strengthened my experience with:

* Building machine learning models using PyTorch
* Developing REST APIs with FastAPI
* Structuring machine learning applications
* Evaluating imbalanced healthcare datasets
* Using Git and GitHub for version control
* Configuring Jenkins for continuous integration
* Creating reproducible Conda environments

---

## 🚀 Future Improvements

* Deploy the API to cloud hosting
* Replace the HTML frontend with React
* Containerize the application using Docker
* Experiment with XGBoost and LightGBM models
* Add SHAP feature importance visualization
* Expand automated testing within the Jenkins pipeline
