# 💰 AI Salary Predictor

<div align="center">

![Developer Salary Predictor](https://img.shields.io/badge/AI-Salary%20Predictor-blueviolet?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=for-the-badge&logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-3.1+-green?style=for-the-badge)

**AI-Powered Developer Salary Predictions Based on Real Survey Data**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Tech Stack](#-tech-stack) • [Contributing](#-contributing)

</div>

---

## 🌟 Overview

An intelligent web application that predicts developer salaries using machine learning, trained on Stack Overflow Developer Survey data. Get accurate salary estimates based on your location, experience, education, and role.

## ✨ Features

- 🎯 **Accurate Predictions** - XGBoost ML model trained on real survey data
- 🌍 **Global Coverage** - Support for 100+ countries
- 💱 **Multi-Currency** - Automatic conversion to local currencies
- ⚡ **Instant Results** - Get predictions in seconds
- 📊 **Detailed Insights** - Hourly, monthly, and annual breakdowns
- 🎨 **Modern UI** - Beautiful, intuitive interface
- 📱 **Responsive Design** - Works on desktop and mobile

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- pip or uv package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Kris-gadara/AI-Salary-Predictor.git
cd AI-Salary-Predictor/developer_salary_prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

Or using uv:
```bash
uv sync
```

3. **Run the application**
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## 📖 Usage

1. **Open the application** in your web browser
2. **Fill in your details** across three tabs:
   - 👤 Personal Info (Country, Age, Education, Role)
   - 💼 Professional Info (Experience, Developer Type, Industry)
   - 🎯 Generate Prediction (Review and predict)
3. **Click "Predict My Salary"** to get your estimate
4. **View results** with detailed breakdowns in USD and local currency

## 🛠️ Tech Stack

### Frontend
- **Streamlit** - Interactive web interface
- **Custom CSS** - Enhanced styling and animations

### Backend & ML
- **XGBoost** - Gradient boosting ML model
- **Pandas** - Data manipulation
- **scikit-learn** - ML utilities
- **Pydantic** - Data validation

### Data
- **Stack Overflow Developer Survey** - Training dataset
- **YAML** - Configuration files

## 📊 Model Details

The XGBoost model analyzes 8 key factors:

1. 🌍 **Country** - Geographic location
2. 💻 **Coding Experience** - Total years of coding
3. 👔 **Work Experience** - Professional years
4. 🎓 **Education Level** - Academic background
5. 🔧 **Developer Type** - Role specialization
6. 🏢 **Industry** - Work sector
7. 👤 **Age** - Age range
8. 👥 **Role Type** - IC vs Manager

## 📁 Project Structure

```
developer_salary_prediction/
├── app.py                  # Main Streamlit application
├── src/
│   ├── infer.py           # Prediction logic
│   ├── schema.py          # Data models
│   └── preprocessing.py   # Feature engineering
├── models/
│   └── model.pkl          # Trained ML model
├── config/
│   ├── currency_rates.yaml
│   ├── model_parameters.yaml
│   └── valid_categories.yaml
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🎨 UI Features

- **Modern Gradient Design** - Eye-catching purple gradient theme
- **Tab-Based Navigation** - Organized input sections
- **Interactive Metrics** - Large, readable salary displays
- **Responsive Layout** - Adapts to screen size
- **Smooth Animations** - Enhanced user experience
- **Informative Sidebar** - Detailed app information

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Stack Overflow** - For providing the Developer Survey data
- **XGBoost Team** - For the excellent ML library
- **Streamlit** - For making web apps easy

## 📧 Contact

**Kris Gadara** - [@Kris-gadara](https://github.com/Kris-gadara)

Project Link: [https://github.com/Kris-gadara/AI-Salary-Predictor](https://github.com/Kris-gadara/AI-Salary-Predictor)

---

<div align="center">

Made with ❤️ by Developers, for Developers

⭐ Star this repo if you find it helpful!

</div>
