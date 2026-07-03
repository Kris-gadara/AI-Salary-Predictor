# AI Salary Predictor

Developer salary prediction app built with Python, scikit-learn, Pandas, Pydantic, XGBoost, and FastAPI.

The project predicts salaries from structured developer profile inputs and serves a lightweight browser UI plus a JSON API.

## Quick Start

From the `developer_salary_prediction` directory:

```bash
uv sync
uv run python src/train.py
uv run python app.py
```

Open `http://localhost:8000` in your browser.

## What Changed

- `app.py` now serves a FastAPI page and `/api/predict` JSON endpoint.
- The existing model pipeline, validation, and inference logic were kept.

## Project Files

- [developer_salary_prediction/app.py](developer_salary_prediction/app.py)
- [developer_salary_prediction/src/infer.py](developer_salary_prediction/src/infer.py)
- [developer_salary_prediction/src/schema.py](developer_salary_prediction/src/schema.py)
- [developer_salary_prediction/src/train.py](developer_salary_prediction/src/train.py)
