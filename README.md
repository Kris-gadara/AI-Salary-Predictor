# AI Salary Predictor - Machine Learning Prediction App

An end-to-end, production-grade Machine Learning application that predicts developer salaries using regression models trained on survey datasets. The project implements clean software architecture principles, sharing the preprocessing and validation pipelines across both a **FastAPI REST API** and a **Streamlit Interactive Dashboard**.

---

## 🚀 Tech Stack & Skills Demonstrated

*   **Python**: Core programming language.
*   **Machine Learning (Regression)**: XGBoost Regressor for non-linear regression modelling.
*   **Scikit-Learn & Pandas**: Feature engineering, outlier removal, cardinality reduction, and data manipulation.
*   **Streamlit**: Interactive user dashboard for predictions and model insights.
*   **FastAPI & Uvicorn**: High-performance asynchronous API endpoints and serving a lightweight HTML client.
*   **Pydantic**: Robust schema validation and runtime type enforcement.
*   **PyYAML**: Externalized configuration management for model and data pipelines.
*   **Pytest**: Unit testing for preprocessing, schema validation, and inference modules.

---

## 📁 Repository Structure

The project has been refactored into a clean, flat, enterprise-friendly structure:

```text
AI-Salary-Predictor/
├── config/                  # External YAML configuration files
│   ├── currency_rates.yaml  # Computed currency conversion rates per country
│   ├── model_parameters.yaml# Hyperparameters, outlier bounds, & training settings
│   └── valid_categories.yaml# Reduced categorical categories allowed for inference
├── models/                  # Stored model artifacts (XGBoost pickle file)
│   └── model.pkl            # Trained model + feature column list
├── src/                     # Core python codebase
│   ├── __init__.py          # Package initializer
│   ├── infer.py             # Shared inference module
│   ├── preprocessing.py     # Shared preprocessing pipelines (Pandas/Scikit-learn)
│   ├── schema.py            # Pydantic schemas for data validation
│   └── train.py             # Model training & cross-validation pipeline
├── tests/                   # Pytest suite
│   ├── conftest.py          # Pytest fixtures
│   ├── test_infer.py        # Inference validation tests
│   ├── test_preprocessing.py# Preprocessing logic tests
│   └── test_schema.py       # Pydantic schema validation tests
├── app.py                   # FastAPI application (API endpoints + lightweight HTML UI)
├── streamlit_app.py         # Streamlit dashboard application (interactive predictor)
├── requirements.txt         # Project package dependencies
└── README.md                # Interview prep and project documentation (this file)
```

---

## ⚙️ How to Setup & Run

### 1. Install Dependencies
Ensure you have Python 3.10+ installed. In your terminal, run:
```bash
pip install -r requirements.txt
```

### 2. Run the FastAPI Application
FastAPI serves the JSON prediction endpoints as well as a lightweight web form:
```bash
uvicorn app:app --reload --port 8000
```
*   **Web Interface**: Access at [http://localhost:8000](http://localhost:8000)
*   **Interactive API Docs (Swagger UI)**: Access at [http://localhost:8000/docs](http://localhost:8000/docs)
*   **Health Check Endpoint**: [http://localhost:8000/health](http://localhost:8000/health)

### 3. Run the Streamlit Dashboard
Streamlit serves a beautiful interactive dashboard with sidebar options, predictions, and model feature importances:
```bash
streamlit run streamlit_app.py
```
*   **Dashboard URL**: Typically runs on [http://localhost:8501](http://localhost:8501)

### 4. Run the Test Suite
Ensure code quality and pipeline consistency using pytest:
```bash
pytest tests/
```

---

## 🏗️ Production Architecture: The DRY Principle

A common issue in ML engineering is **Training-Serving Skew** (where preprocessing/validation differs between model training and real-time model serving). 

To solve this, this repository implements a shared architecture where the training script (`src/train.py`), the FastAPI app (`app.py`), and the Streamlit app (`streamlit_app.py`) reuse the exact same:
1.  **Pydantic Schema Validation** (`src/schema.py`): Validates input bounds (e.g. non-negative years of coding experience).
2.  **Pandas / Scikit-Learn Preprocessing** (`src/preprocessing.py`): Normalizes text encoding differences, handles "Other" catch-all categories, and handles one-hot encoding columns consistently.
3.  **Inference Layer** (`src/infer.py`): Re-indexes input feature matrices to align with training columns and performs currency conversions.

---

## 🎯 Key ML Engineering Concepts (Interview Talking Points)

### 1. Outlier Removal Per Country (Not Global)
*   *Why?* Salary distributions vary heavily between countries (e.g. median developer salary in the USA vs India). A global 2-98% percentile cut would wipe out high-earning US entries and low-earning Indian entries.
*   *How?* We group the training dataset by `Country` and apply local percentile bounds (2% to 98%) inside each group. This preserves the representative distribution of each region while removing extreme noise.

### 2. High-Cardinality Reduction
*   *Why?* Categorical variables like `Country` or `DevType` have dozens of unique categories. One-hot encoding them directly would create hundreds of sparse columns, causing overfitting and slower training.
*   *How?* We analyze frequency distributions, keeping only the top 20 most frequent categories and grouping all others into an `"Other"` category.

### 3. Currency conversion
*   *Why?* Developers report compensation in local currency.
*   *How?* During training, we compute median conversion rates (using reported `CompTotal` vs USD `ConvertedCompYearly`) and cache them in `config/currency_rates.yaml`. In production, predictions are in USD, but the UI automatically converts values to the developer's local currency.

---

## 💬 Interview Q&A Cheatsheet

### Q1: Why did you use both FastAPI and Streamlit?
> **Answer**: They serve different business requirements. **FastAPI** is a high-performance, asynchronous REST API framework suitable for production system integration (e.g., exposing predictions to mobile applications, SaaS products, or other microservices). **Streamlit** is optimized for rapid internal prototyping, enabling product managers and stakeholders to quickly interact with the model, perform what-if analyses, and visualize insights (like feature importance) without writing HTML/JS.

### Q2: How did you prevent Training-Serving Skew in this codebase?
> **Answer**: I refactored the project to keep preprocessing and validation strictly modular. Both `app.py` (FastAPI) and `streamlit_app.py` import the exact same Pydantic schemas from `src/schema.py` and preprocessing logic from `src/preprocessing.py`. When an inference request is received, the input goes through the same normalization, filling of missing values, and one-hot alignment as the training pipeline, ensuring the model receives inputs exactly as it expects.

### Q3: Why did you choose XGBoost instead of Linear Regression?
> **Answer**: Developer salaries are highly non-linear. For example, the difference in salary between 10 and 15 years of coding is typically less dramatic than the difference between 0 and 5 years. Linear models fail to capture these diminishing returns or complex combinations of features (e.g. the interaction of living in Switzerland *and* being an Engineering Manager). XGBoost, being an ensemble of decision trees, naturally captures non-linear relationships and feature interactions without requiring manual polynomial feature engineering.

### Q4: How is the model evaluated and validated?
> **Answer**: The model training pipeline in `src/train.py` performs **5-Fold Cross-Validation** to measure model generalization and detect overfitting. We compute the R² score across all folds. The final model is trained on the full dataset, with early stopping (configured via `config/model_parameters.yaml` to 50 rounds) applied on a 10% validation split.
