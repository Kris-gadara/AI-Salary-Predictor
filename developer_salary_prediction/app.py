"""FastAPI web app for salary prediction."""

import sys
from html import escape
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

# Ensure the 'developer_salary_prediction' directory is on sys.path so imports
# work regardless of where the server is started from.
_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from src.infer import get_local_currency, predict_salary, valid_categories
from src.schema import SalaryInput


app = FastAPI(title="Developer Salary Predictor", version="1.0.0")

VALID_COUNTRIES = valid_categories["Country"]
VALID_EDUCATION_LEVELS = valid_categories["EdLevel"]
VALID_DEV_TYPES = valid_categories["DevType"]
VALID_INDUSTRIES = valid_categories["Industry"]
VALID_AGES = valid_categories["Age"]
VALID_IC_OR_PM = valid_categories["ICorPM"]


def _default_value(options: list[str], preferred: str) -> str:
    return preferred if preferred in options else options[0]


DEFAULT_COUNTRY = _default_value(VALID_COUNTRIES, "United States of America")
DEFAULT_EDUCATION = _default_value(
    VALID_EDUCATION_LEVELS, "Bachelor's degree (B.A., B.S., B.Eng., etc.)"
)
DEFAULT_DEV_TYPE = _default_value(VALID_DEV_TYPES, "Developer, back-end")
DEFAULT_INDUSTRY = _default_value(VALID_INDUSTRIES, "Software Development")
DEFAULT_AGE = _default_value(VALID_AGES, "25-34 years old")
DEFAULT_IC_OR_PM = _default_value(VALID_IC_OR_PM, "Individual contributor")


def _option_html(options: list[str], selected: str) -> str:
    return "\n".join(
        f'<option value="{escape(option)}" {"selected" if option == selected else ""}>'
        f"{escape(option)}</option>"
        for option in options
    )


def _card(title: str, body: str) -> str:
    return f"""
    <div class="card">
      <div class="card-title">{escape(title)}</div>
      <div class="card-body">{body}</div>
    </div>
    """


def _metric(label: str, value: str, subtitle: str = "") -> str:
    subtitle_html = f'<div class="metric-subtitle">{escape(subtitle)}</div>' if subtitle else ""
    return f"""
    <div class="metric">
      <div class="metric-label">{escape(label)}</div>
      <div class="metric-value">{escape(value)}</div>
      {subtitle_html}
    </div>
    """


def _result_panel(salary: float, country: str) -> str:
    local = get_local_currency(country, salary)
    result_cards = [
        _metric("Annual Salary (USD)", f"${salary:,.0f}", "Predicted yearly compensation in USD"),
        _metric("Monthly (USD)", f"${salary / 12:,.0f}", "Approximate monthly salary"),
        _metric("Hourly (USD)", f"${salary / (52 * 40):,.0f}", "Assumes a 40 hour work week"),
        _metric("Weekly (USD)", f"${salary / 52:,.0f}", "Approximate weekly salary"),
    ]

    if local and local.get("code") != "USD":
        result_cards.insert(
            1,
            _metric(
                f"Annual Salary ({local['code']})",
                f"{local['salary_local']:,.0f} {local['code']}",
                f"Converted using 1 USD = {local['rate']} {local['code']} ({local['name']})",
            ),
        )

    return """
    <section class="panel success">
      <div class="panel-kicker">Prediction complete</div>
      <h2>Your estimated salary</h2>
      <div class="metrics-grid">
        {cards}
      </div>
      <p class="note">
        This result is based on the trained model and should be treated as an estimate.
        Real compensation can vary by company, location, and scope.
      </p>
    </section>
    """.format(cards="\n".join(result_cards))


def _error_panel(message: str) -> str:
    return f"""
    <section class="panel error">
      <div class="panel-kicker">Unable to predict</div>
      <h2>Check the input or model file</h2>
      <p>{escape(message)}</p>
    </section>
    """


def _build_page(
    *,
    submitted: bool,
    country: str,
    years_code: float,
    work_exp: float,
    education_level: str,
    dev_type: str,
    industry: str,
    age: str,
    ic_or_pm: str,
    result_html: str = "",
) -> str:
    page_result = result_html or """
    <section class="panel empty">
      <div class="panel-kicker">Ready when you are</div>
      <h2>Generate a prediction</h2>
      <p>Fill in the form and press Predict Salary to estimate annual compensation.</p>
    </section>
    """

    form_html = f"""
    <form class="form" method="get" action="/">
      <input type="hidden" name="submitted" value="1">
      <div class="section-title">Personal Info</div>
      <div class="grid two-col">
        <label>
          <span>Country</span>
          <select name="country">{_option_html(VALID_COUNTRIES, country)}</select>
        </label>
        <label>
          <span>Age Range</span>
          <select name="age">{_option_html(VALID_AGES, age)}</select>
        </label>
        <label>
          <span>Education Level</span>
          <select name="education_level">{_option_html(VALID_EDUCATION_LEVELS, education_level)}</select>
        </label>
        <label>
          <span>Role Type</span>
          <select name="ic_or_pm">{_option_html(VALID_IC_OR_PM, ic_or_pm)}</select>
        </label>
      </div>

      <div class="section-title">Professional Info</div>
      <div class="grid two-col">
        <label>
          <span>Total Years of Coding</span>
          <input type="number" name="years_code" min="0" max="50" step="1" value="{years_code}">
        </label>
        <label>
          <span>Years of Professional Experience</span>
          <input type="number" name="work_exp" min="0" max="50" step="1" value="{work_exp}">
        </label>
        <label>
          <span>Developer Type</span>
          <select name="dev_type">{_option_html(VALID_DEV_TYPES, dev_type)}</select>
        </label>
        <label>
          <span>Industry</span>
          <select name="industry">{_option_html(VALID_INDUSTRIES, industry)}</select>
        </label>
      </div>

      <div class="actions">
        <button type="submit">Predict Salary</button>
      </div>
    </form>
    """

    overview_cards = """
    <div class="cards">
      {card_1}
      {card_2}
      {card_3}
    </div>
    """.format(
        card_1=_card("FastAPI", "Serves the UI and prediction API."),
        card_2=_card("Pandas + scikit-learn", "Handle feature preparation and model inputs."),
        card_3=_card("XGBoost", "Backs the trained salary prediction model."),
    )

    body_classes = "submitted" if submitted else "idle"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Developer Salary Predictor</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0d1117;
      --panel: #161b22;
      --panel-2: #1c2333;
      --border: #30363d;
      --text: #e6edf3;
      --muted: #8b949e;
      --accent: #f0b429;
      --accent-2: #d4990a;
      --good: #10b981;
      --bad: #f43f5e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at top, rgba(240, 180, 41, 0.08), transparent 34%), var(--bg);
      color: var(--text);
    }}
    .shell {{ max-width: 1180px; margin: 0 auto; padding: 32px 20px 48px; }}
    .hero {{ text-align: center; margin-bottom: 24px; }}
    .eyebrow {{ color: var(--accent); font-weight: 700; letter-spacing: 0.18em; text-transform: uppercase; font-size: 0.78rem; }}
    h1 {{ margin: 8px 0 10px; font-size: clamp(2rem, 4vw, 3.2rem); line-height: 1.05; }}
    .subtitle {{ color: var(--muted); max-width: 820px; margin: 0 auto; font-size: 1.02rem; line-height: 1.6; }}
    .cards {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin: 28px 0; }}
    .card, .panel, .form {{ background: rgba(22, 27, 34, 0.98); border: 1px solid var(--border); border-radius: 18px; box-shadow: 0 24px 80px rgba(0, 0, 0, 0.22); }}
    .card {{ padding: 18px; }}
    .card-title, .section-title, .panel-kicker {{ color: var(--accent); font-weight: 700; }}
    .card-body, .panel p {{ color: var(--muted); line-height: 1.55; }}
    .layout {{ display: grid; grid-template-columns: 1.2fr 0.95fr; gap: 18px; align-items: start; }}
    .form {{ padding: 22px; }}
    .section-title {{ margin: 10px 0 12px; font-size: 0.92rem; letter-spacing: 0.08em; text-transform: uppercase; }}
    .grid {{ display: grid; gap: 14px; }}
    .two-col {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    label {{ display: grid; gap: 8px; font-size: 0.92rem; color: var(--text); }}
    label span {{ color: var(--muted); font-weight: 600; }}
    input, select {{ width: 100%; min-height: 46px; border-radius: 12px; border: 1px solid var(--border); background: var(--panel-2); color: var(--text); padding: 0.8rem 0.9rem; font-size: 1rem; }}
    input:focus, select:focus {{ outline: 2px solid rgba(240, 180, 41, 0.28); border-color: var(--accent); }}
    .actions {{ margin-top: 18px; }}
    button {{ width: 100%; min-height: 50px; border: none; border-radius: 14px; background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%); color: #0d1117; font-size: 1rem; font-weight: 800; cursor: pointer; box-shadow: 0 10px 30px rgba(240, 180, 41, 0.25); }}
    .result-column {{ display: grid; gap: 18px; }}
    .panel {{ padding: 22px; }}
    .panel h2 {{ margin: 8px 0 10px; font-size: 1.5rem; }}
    .metrics-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-top: 16px; }}
    .metric {{ background: rgba(28, 35, 51, 0.88); border: 1px solid var(--border); border-radius: 16px; padding: 16px; }}
    .metric-label {{ color: var(--muted); font-size: 0.9rem; }}
    .metric-value {{ font-size: 1.5rem; font-weight: 800; margin-top: 6px; }}
    .metric-subtitle {{ color: var(--muted); font-size: 0.84rem; margin-top: 6px; line-height: 1.45; }}
    .success {{ border-color: rgba(16, 185, 129, 0.36); }}
    .error {{ border-color: rgba(244, 63, 94, 0.38); }}
    .empty {{ border-style: dashed; }}
    .note {{ margin-top: 14px; font-size: 0.93rem; }}
    .footer {{ text-align: center; color: var(--muted); margin-top: 22px; font-size: 0.92rem; }}
    .hero-meta {{ margin-top: 16px; color: var(--muted); }}
    @media (max-width: 980px) {{
      .layout, .cards {{ grid-template-columns: 1fr; }}
      .two-col, .metrics-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body class="{body_classes}">
  <main class="shell">
    <section class="hero">
      <div class="eyebrow">Developer Salary Predictor</div>
      <h1>FastAPI salary predictions with a simple browser UI</h1>
      <p class="subtitle">
        Enter your background and role details, then get an estimated annual salary from the trained regression model.
      </p>
      <div class="hero-meta">FastAPI · Pandas · scikit-learn · XGBoost · Pydantic</div>
    </section>

    {overview_cards}

    <section class="layout">
      <div>
        {form_html}
      </div>
      <div class="result-column">
        {page_result}
      </div>
    </section>
    <div class="footer">Built for local prediction workflows and lightweight deployment.</div>
  </main>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def home(
    submitted: bool = Query(False),
    country: str = Query(DEFAULT_COUNTRY),
    years_code: float = Query(5.0, ge=0),
    work_exp: float = Query(3.0, ge=0),
    education_level: str = Query(DEFAULT_EDUCATION),
    dev_type: str = Query(DEFAULT_DEV_TYPE),
    industry: str = Query(DEFAULT_INDUSTRY),
    age: str = Query(DEFAULT_AGE),
    ic_or_pm: str = Query(DEFAULT_IC_OR_PM),
) -> HTMLResponse:
    result_html = ""

    if submitted:
        try:
            input_data = SalaryInput(
                country=country,
                years_code=years_code,
                work_exp=work_exp,
                education_level=education_level,
                dev_type=dev_type,
                industry=industry,
                age=age,
                ic_or_pm=ic_or_pm,
            )
            salary = predict_salary(input_data)
            result_html = _result_panel(salary, input_data.country)
        except Exception as exc:
            result_html = _error_panel(str(exc))

    return HTMLResponse(
        _build_page(
            submitted=submitted,
            country=country,
            years_code=years_code,
            work_exp=work_exp,
            education_level=education_level,
            dev_type=dev_type,
            industry=industry,
            age=age,
            ic_or_pm=ic_or_pm,
            result_html=result_html,
        )
    )


@app.post("/api/predict")
def api_predict(payload: SalaryInput) -> JSONResponse:
    salary = predict_salary(payload)
    local = get_local_currency(payload.country, salary)
    return JSONResponse(
        {
            "salary_usd": round(salary, 2),
            "salary_local": local,
            "input": payload.model_dump(),
        }
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)