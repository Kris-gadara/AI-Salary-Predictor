# Claude Development Guide

## Project Overview

This repository contains a local-first developer salary prediction app built with Python, Pandas, scikit-learn preprocessing, Pydantic validation, XGBoost regression, and FastAPI.

## Tech Stack

- Python 3.11+
- uv
- Pandas
- scikit-learn
- Pydantic
- FastAPI
- XGBoost

## Key Files

- [app.py](app.py) - FastAPI web UI and JSON API
- [src/infer.py](src/infer.py) - Prediction logic
- [src/schema.py](src/schema.py) - Input validation
- [src/train.py](src/train.py) - Training pipeline

## Common Commands

```bash
uv sync
uv run python src/train.py
uv run python app.py
```

## Notes

- The browser UI is now served directly by FastAPI.
- Keep the HTML UI lightweight and local-first.
