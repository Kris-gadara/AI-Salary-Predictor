"""Streamlit web app for salary prediction."""

import streamlit as st

from src.infer import predict_salary, get_local_currency, valid_categories
from src.schema import SalaryInput

# Page configuration
st.set_page_config(
    page_title="Developer Salary Predictor | AI-Powered Predictions",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for enhanced UI
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    /* Card-like container */
    .stApp {
        background: white;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    
    /* Headers */
    h1 {
        color: #667eea;
        font-weight: 800;
        font-size: 3rem !important;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    h2 {
        color: #764ba2;
        font-weight: 700;
        border-left: 5px solid #667eea;
        padding-left: 1rem;
        margin-top: 2rem;
    }
    
    h3 {
        color: #667eea;
        font-weight: 600;
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.3rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Input fields */
    .stSelectbox, .stNumberInput {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 1rem 3rem;
        font-size: 1.2rem;
        font-weight: 600;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 800;
        color: #667eea;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Success/Info boxes */
    .stSuccess {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        border-radius: 15px;
        padding: 1rem;
    }
    
    .stInfo {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        border: 2px solid rgba(255, 255, 255, 0.4);
    }
    
    /* Divider */
    hr {
        border: 0;
        height: 3px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        margin: 2rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #999;
        font-size: 0.9rem;
        margin-top: 3rem;
        padding: 1rem;
        border-top: 2px solid #eee;
    }
    
    /* Columns */
    [data-testid="column"] {
        background: #fafafa;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    }
    
    /* Animation for result */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .result-container {
        animation: slideIn 0.5s ease-out;
    }
    </style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("<h1>💰 Developer Salary Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-Powered Salary Predictions Based on Stack Overflow Developer Survey Data</p>", unsafe_allow_html=True)

# Feature highlights
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown("### 🎯 Accurate")
    st.markdown("XGBoost ML model trained on real survey data")
with col_b:
    st.markdown("### 🌍 Global")
    st.markdown(f"Support for {len(valid_categories['Country'])}+ countries")
with col_c:
    st.markdown("### ⚡ Instant")
    st.markdown("Get predictions in seconds")

# Sidebar with info
with st.sidebar:
    st.markdown("## 📊 About This App")
    st.markdown("---")
    
    st.markdown("""
        ### 🤖 Machine Learning Model
        This app uses an **XGBoost** (gradient boosting) model trained on 
        Stack Overflow Developer Survey data.
        
        ### 📈 Prediction Factors
        The model analyzes:
        
        - 🌍 **Country** - Geographic location
        - 💻 **Coding Experience** - Years of coding
        - 👔 **Work Experience** - Professional years
        - 🎓 **Education** - Academic background
        - 🔧 **Developer Type** - Role specialization
        - 🏢 **Industry** - Work sector
        - 👤 **Age** - Age range
        - 👥 **Role Type** - IC vs Manager
    """)
    
    st.info("💡 **Tip:** Results are estimates based on survey averages and may vary from actual salaries.")
    
    st.markdown("---")
    st.markdown("### 📋 Model Coverage")
    
    coverage_data = {
        "🌍 Countries": len(valid_categories['Country']),
        "🎓 Education Levels": len(valid_categories['EdLevel']),
        "👨‍💻 Developer Types": len(valid_categories['DevType']),
        "🏢 Industries": len(valid_categories['Industry']),
        "📅 Age Ranges": len(valid_categories['Age']),
        "👥 Role Types": len(valid_categories['ICorPM'])
    }
    
    for label, count in coverage_data.items():
        st.markdown(f"**{label}:** `{count}`")
    
    st.caption("✨ Only validated values from training data are available")
    
    st.markdown("---")
    st.markdown("### 🚀 Tech Stack")
    st.markdown("""
        - **Frontend:** Streamlit
        - **ML Model:** XGBoost
        - **Data:** Stack Overflow Survey
        - **Validation:** Pydantic
    """)

# Main input form
st.markdown("---")
st.header("🔍 Enter Developer Information")
st.markdown("Fill in the details below to get your salary prediction")

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs(["👤 Personal Info", "💼 Professional Info", "🎯 Generate Prediction"])

# Get valid categories from training
valid_countries = valid_categories["Country"]
valid_education_levels = valid_categories["EdLevel"]
valid_dev_types = valid_categories["DevType"]
valid_industries = valid_categories["Industry"]
valid_ages = valid_categories["Age"]
valid_icorpm = valid_categories["ICorPM"]

# Set default values (if available)
default_country = (
    "United States of America"
    if "United States of America" in valid_countries
    else valid_countries[0]
)
default_education = (
    "Bachelor's degree (B.A., B.S., B.Eng., etc.)"
    if "Bachelor's degree (B.A., B.S., B.Eng., etc.)" in valid_education_levels
    else valid_education_levels[0]
)
default_dev_type = (
    "Developer, back-end"
    if "Developer, back-end" in valid_dev_types
    else valid_dev_types[0]
)
default_industry = (
    "Software Development"
    if "Software Development" in valid_industries
    else valid_industries[0]
)
default_age = "25-34 years old" if "25-34 years old" in valid_ages else valid_ages[0]
default_icorpm = (
    "Individual contributor"
    if "Individual contributor" in valid_icorpm
    else valid_icorpm[0]
)

# Tab 1: Personal Information
with tab1:
    st.markdown("### 📍 Location & Demographics")
    col1, col2 = st.columns(2)
    
    with col1:
        country = st.selectbox(
            "🌍 Country",
            options=valid_countries,
            index=valid_countries.index(default_country),
            help="Your country of residence (impacts salary significantly)",
        )
        
        age = st.selectbox(
            "👤 Age Range",
            options=valid_ages,
            index=valid_ages.index(default_age),
            help="Your current age range",
        )
    
    with col2:
        education = st.selectbox(
            "🎓 Education Level",
            options=valid_education_levels,
            index=valid_education_levels.index(default_education),
            help="Your highest level of education completed",
        )
        
        ic_or_pm = st.selectbox(
            "👥 Role Type",
            options=valid_icorpm,
            index=valid_icorpm.index(default_icorpm),
            help="Are you an individual contributor or people manager?",
        )

# Tab 2: Professional Information
with tab2:
    st.markdown("### 💼 Experience & Specialization")
    
    col3, col4 = st.columns(2)
    
    with col3:
        years = st.number_input(
            "💻 Total Years of Coding",
            min_value=0,
            max_value=50,
            value=5,
            step=1,
            help="Including education, how many years have you been coding?",
        )
        
        dev_type = st.selectbox(
            "🔧 Developer Type",
            options=valid_dev_types,
            index=valid_dev_types.index(default_dev_type),
            help="Your primary developer role or specialization",
        )
    
    with col4:
        work_exp = st.number_input(
            "👔 Years of Professional Experience",
            min_value=0,
            max_value=50,
            value=3,
            step=1,
            help="Years of professional work experience (not including education)",
        )
        
        industry = st.selectbox(
            "🏢 Industry",
            options=valid_industries,
            index=valid_industries.index(default_industry),
            help="The industry sector you work in",
        )

# Tab 3: Prediction
with tab3:
    st.markdown("### 🎯 Ready to Predict?")
    st.markdown("Review your information and click the button below to get your salary prediction.")
    
    # Display summary
    st.markdown("#### 📋 Summary")
    summary_col1, summary_col2 = st.columns(2)
    
    with summary_col1:
        st.markdown(f"""
        - **Country:** {country}
        - **Age:** {age}
        - **Education:** {education}
        - **Role:** {ic_or_pm}
        """)
    
    with summary_col2:
        st.markdown(f"""
        - **Total Coding Years:** {years}
        - **Work Experience:** {work_exp} years
        - **Developer Type:** {dev_type}
        - **Industry:** {industry}
        """)
    
    st.markdown("---")

    # Prediction button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_button = st.button("🔮 Predict My Salary", type="primary", use_container_width=True)
    
    if predict_button:
        try:
            # Create input model
            input_data = SalaryInput(
                country=country,
                years_code=years,
                work_exp=work_exp,
                education_level=education,
                dev_type=dev_type,
                industry=industry,
                age=age,
                ic_or_pm=ic_or_pm,
            )

            # Make prediction
            with st.spinner("🤖 AI is analyzing your profile..."):
                salary = predict_salary(input_data)

            # Display result with animation
            st.markdown("<div class='result-container'>", unsafe_allow_html=True)
            st.balloons()
            st.success("✅ Prediction Complete!")
            
            st.markdown("### 💵 Your Predicted Salary")

            # Show USD and local currency side by side
            local = get_local_currency(country, salary)
            if local and local["code"] != "USD":
                col_usd, col_local = st.columns(2)
                with col_usd:
                    st.metric(
                        label="💵 Annual Salary (USD)",
                        value=f"${salary:,.0f}",
                        help="Predicted annual compensation in US Dollars",
                    )
                with col_local:
                    st.metric(
                        label=f"💰 Annual Salary ({local['code']})",
                        value=f"{local['salary_local']:,.0f} {local['code']}",
                        help=f"Converted using survey rate: 1 USD = {local['rate']} {local['code']} ({local['name']})",
                    )
                
                # Additional insights
                st.markdown("---")
                st.markdown("#### 📊 Additional Insights")
                insight_col1, insight_col2, insight_col3 = st.columns(3)
                
                with insight_col1:
                    monthly_usd = salary / 12
                    st.metric("📅 Monthly (USD)", f"${monthly_usd:,.0f}")
                
                with insight_col2:
                    hourly_usd = salary / (52 * 40)  # Assuming 40 hours/week
                    st.metric("⏰ Hourly (USD)", f"${hourly_usd:,.0f}")
                
                with insight_col3:
                    if local:
                        monthly_local = local['salary_local'] / 12
                        st.metric(f"📅 Monthly ({local['code']})", f"{monthly_local:,.0f}")
            else:
                st.metric(
                    label="💵 Estimated Annual Salary",
                    value=f"${salary:,.0f}",
                    help="Predicted annual compensation in USD",
                )
                
                # Additional insights for USD only
                st.markdown("---")
                st.markdown("#### 📊 Salary Breakdown")
                insight_col1, insight_col2, insight_col3 = st.columns(3)
                
                with insight_col1:
                    monthly_usd = salary / 12
                    st.metric("📅 Monthly", f"${monthly_usd:,.0f}")
                
                with insight_col2:
                    hourly_usd = salary / (52 * 40)
                    st.metric("⏰ Hourly", f"${hourly_usd:,.0f}")
                
                with insight_col3:
                    weekly_usd = salary / 52
                    st.metric("📆 Weekly", f"${weekly_usd:,.0f}")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Disclaimer
            st.info("ℹ️ **Note:** This prediction is based on survey data and represents an estimate. Actual salaries may vary based on company size, specific skills, location within country, and other factors not captured in this model.")

        except FileNotFoundError:
            st.error(
                """
                ❌ **Model Not Found!** 
                
                Please train the model first by running:
                ```bash
                python src/train.py
                ```
                """
            )
        except Exception as e:
            st.error(f"❌ **Error:** {str(e)}")
            st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
    <div class='footer'>
        <p><strong>Developer Salary Predictor</strong> | AI-Powered Predictions</p>
        <p>
            🚀 Built with <strong>Streamlit</strong> | 
            📊 Data from <strong>Stack Overflow Developer Survey</strong> | 
            🤖 Model: <strong>XGBoost</strong>
        </p>
        <p style='font-size: 0.8rem; color: #bbb;'>
            © 2026 AI-Salary-Predictor | Made with ❤️ by Developers, for Developers
        </p>
    </div>
""", unsafe_allow_html=True)

