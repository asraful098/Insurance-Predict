# Insurance Predict

A Machine Learning regression project that predicts individual medical insurance costs based on personal, demographic, and lifestyle information.

## Project Overview

**Insurance Predict** uses Machine Learning regression techniques to estimate medical insurance charges based on features such as age, BMI, number of children, smoking status, sex, and region.

The project follows an end-to-end Machine Learning workflow, including data preprocessing, model training, hyperparameter tuning, model evaluation, model saving, and deployment with Streamlit.

## Features

* Data cleaning and preprocessing
* Handling numerical and categorical features
* Feature scaling and encoding
* Multiple regression models
* XGBoost regression
* Hyperparameter tuning
* Model evaluation using regression metrics
* Best model selection
* Model serialization using Joblib
* Interactive Streamlit prediction interface

## Machine Learning Workflow

```text
Raw Data
   ↓
Data Cleaning
   ↓
Train / Test Split
   ↓
Data Preprocessing
   ↓
Feature Engineering
   ↓
Model Training
   ↓
Hyperparameter Tuning
   ↓
Model Evaluation
   ↓
Best Model Selection
   ↓
Save Model with Joblib
   ↓
Streamlit Deployment
```

## Dataset

The model uses the following features:

| Feature    | Description                              |
| ---------- | ---------------------------------------- |
| `age`      | Age of the individual                    |
| `sex`      | Gender                                   |
| `bmi`      | Body Mass Index                          |
| `children` | Number of children/dependents            |
| `smoker`   | Smoking status                           |
| `region`   | Residential region                       |
| `charges`  | Medical insurance cost — target variable |

The target variable is:

```text
charges
```

## Models

Several regression algorithms are evaluated during the project, including:

* Linear Regression
* Ridge Regression
* Random Forest Regressor
* Gradient Boosting Regressor
* XGBoost Regressor

Hyperparameter tuning is performed to improve model performance and identify the best-performing model for deployment.

## Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **XGBoost**
* **Joblib**
* **Streamlit**
* **Matplotlib**

## Project Structure

```text
Insurance-Predict/
│
├── data/
│   └── insurance.csv
│
├── src/
│   ├── ...
│   └── ...
│
├── outputs/
│   └── best_insurance_model.joblib
│
├── app.py
├── .gitignore
├── requirements.txt
└── README.md
```

The **Streamlit application** is located in the project root:

```text
app.py
```

Run the application with:

```bash
streamlit run app.py
```


## Installation

### 1. Clone the repository

```bash
git clone https://github.com/asraful098/Insurance-Predict.git
```

### 2. Navigate to the project

```bash
cd Insurance-Predict
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the Streamlit App

Run the application using:

```bash
streamlit run app.py
```

The application will open in your browser, where you can enter the required information and receive a predicted insurance cost.

## Model

The best-performing trained model is saved using **Joblib**:

```text
best_insurance_model.joblib
```

The Streamlit application loads this saved model to generate predictions without retraining the model every time the application starts.

## Project Goal

The main goal of this project is to build a complete Machine Learning regression workflow—from raw data preprocessing and model development to hyperparameter optimization and deployment as an interactive web application.

## Future Improvements

* Add model explainability
* Improve the Streamlit UI
* Add prediction visualizations
* Add confidence/uncertainty information
* Deploy the application publicly
* Add automated model retraining

## Author

**Sheikh Md. Asraful Islam Robin**

AI/ML Student & Developer

* GitHub: [@asraful098](https://github.com/asraful098)
* Portfolio: [Asraful | AI/ML Developer](https://asraful098.github.io/asaful-site/)

---

⭐ If you find this project useful, consider giving the repository a star!
