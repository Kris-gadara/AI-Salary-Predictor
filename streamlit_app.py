import streamlit as st
import pickle
import pandas as pd
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="AI Salary Predictor",
    page_icon="💰",
    layout="wide",
)

# Title and description
st.title("💰 AI Salary Predictor")
st.markdown("""
Predict salaries using regression models trained on developer survey datasets. 
This application leverages **scikit-learn**, **Pandas**, and **XGBoost** to make predictions, 
and is deployed using **Streamlit** and **FastAPI**.
""")

# Load model and categories
try:
    from src.infer import predict_salary, get_local_currency, valid_categories
    from src.schema import SalaryInput
    
    # Load model for feature importance visualization
    model_path = Path("models/model.pkl")
    with open(model_path, "rb") as f:
        artifacts = pickle.load(f)
        xgb_model = artifacts["model"]
        feature_columns = artifacts["feature_columns"]
except Exception as e:
    st.error(f"Error loading model or helper modules: {e}")
    st.info("Make sure you have run training or have 'models/model.pkl' generated.")
    st.stop()

# Layout: Two columns for inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Personal Profile")
    country = st.selectbox("Country", sorted(valid_categories["Country"]), index=sorted(valid_categories["Country"]).index("United States of America") if "United States of America" in valid_categories["Country"] else 0)
    age = st.selectbox("Age Range", valid_categories["Age"], index=valid_categories["Age"].index("25-34 years old") if "25-34 years old" in valid_categories["Age"] else 0)
    education_level = st.selectbox("Education Level", valid_categories["EdLevel"], index=valid_categories["EdLevel"].index("Bachelor's degree (B.A., B.S., B.Eng., etc.)") if "Bachelor's degree (B.A., B.S., B.Eng., etc.)" in valid_categories["EdLevel"] else 0)
    ic_or_pm = st.selectbox("Role Type", valid_categories["ICorPM"], index=valid_categories["ICorPM"].index("Individual contributor") if "Individual contributor" in valid_categories["ICorPM"] else 0)

with col2:
    st.subheader("💼 Professional Profile")
    dev_type = st.selectbox("Developer Type", valid_categories["DevType"], index=valid_categories["DevType"].index("Developer, back-end") if "Developer, back-end" in valid_categories["DevType"] else 0)
    industry = st.selectbox("Industry", valid_categories["Industry"], index=valid_categories["Industry"].index("Software Development") if "Software Development" in valid_categories["Industry"] else 0)
    years_code = st.slider("Total Years of Coding", min_value=0.0, max_value=50.0, value=5.0, step=1.0)
    work_exp = st.slider("Professional Work Experience (Years)", min_value=0.0, max_value=50.0, value=3.0, step=1.0)

# Prediction Section
st.markdown("---")

if st.button("🔮 Predict Salary", type="primary"):
    try:
        # Prepare inputs
        input_data = SalaryInput(
            country=country,
            years_code=years_code,
            work_exp=work_exp,
            education_level=education_level,
            dev_type=dev_type,
            industry=industry,
            age=age,
            ic_or_pm=ic_or_pm
        )
        
        # Predict
        predicted_val = predict_salary(input_data)
        
        # Display Results
        st.subheader("🎉 Predicted Salary Estimate")
        
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric(label="Annual Salary (USD)", value=f"${predicted_val:,.2f}")
            st.write(f"**Monthly:** ${predicted_val/12:,.2f} | **Hourly:** ${predicted_val/(52*40):,.2f}")
            
        with res_col2:
            # Convert to local currency if applicable
            local_info = get_local_currency(country, predicted_val)
            if local_info and local_info.get("code") != "USD":
                st.metric(
                    label=f"Annual Salary ({local_info['code']})",
                    value=f"{local_info['salary_local']:,.2f} {local_info['code']}"
                )
                st.caption(f"Rate: 1 USD = {local_info['rate']} {local_info['code']} ({local_info['name']})")
            else:
                st.metric(label="Local Currency", value="USD (Same as base)")
                
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

# Model Analysis Tab / Section
st.markdown("---")
with st.expander("📊 Model Insights & Feature Importance"):
    st.markdown("Here are the top 10 feature categories that influence the XGBoost model's salary predictions:")
    try:
        # Get feature importances
        importances = xgb_model.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": feature_columns,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(10)
        
        # Plot
        st.bar_chart(data=importance_df, x="Feature", y="Importance")
    except Exception as e:
        st.write(f"Could not compute feature importances: {e}")
