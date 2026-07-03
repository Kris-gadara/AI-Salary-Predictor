# Developer Salary Prediction

FastAPI-based salary prediction app that uses scikit-learn preprocessing, Pandas, Pydantic validation, and an XGBoost regression model trained on Stack Overflow survey data.

## Run

```bash
uv sync
uv run python src/train.py
uv run python app.py
```

Then open `http://localhost:8000`.

## API

- `GET /` renders the browser UI
- `POST /api/predict` returns JSON predictions
- `GET /health` returns a simple status check

## UI Fields

- Country
- Years of coding
- Years of professional work experience
- Education level
- Developer type
- Industry
- Age range
- IC or PM

## Notes

- The existing model and validation files are still used.
- `models/model.pkl` and `config/valid_categories.yaml` must exist before launching the app.
