# 🏏 IPL Oracle

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**AI-Powered Decision Support System for IPL Match Prediction**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [How It Works](#-how-it-works) • [Screenshots](#-screenshots)

</div>

---

## 📖 Overview

IPL Oracle is a professional-grade Streamlit dashboard that leverages machine learning to predict IPL match outcomes. Built with historical data from 2008-2024, it combines **Elo rating systems**, **Random Forest classifiers**, and **advanced feature engineering** to provide accurate win probability predictions.

Whether you're a cricket enthusiast analyzing team performance or looking to make data-driven decisions, IPL Oracle offers comprehensive insights into team strengths, venue characteristics, and match dynamics.

---

## ✨ Features

### 🎯 Match Simulator
- **Win Probability Predictions**: Get AI-powered predictions for any IPL matchup
- **Dynamic Elo Ratings**: Team strength calculated from historical performance
- **Toss & Venue Factors**: Incorporates toss decisions and venue-specific advantages
- **Value Finder**: Calculate Expected Value (EV) for betting opportunities by comparing model probabilities with market odds

### 📊 Team Analysis
- **Elo Rating Trajectory**: Interactive charts showing team strength evolution from 2008-2024
- **Current Rankings**: Real-time Elo-based power rankings of all 10 IPL teams
- **Head-to-Head Records**: Historical matchup statistics between any two teams

### 🏟️ Venue Insights
- **Chase vs Defend Analysis**: Identify venues that favor chasing or defending teams
- **Toss Advantage Statistics**: Discover where winning the toss matters most
- **Scoring Patterns**: Average first innings scores and their correlation with chase success

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Deepak-Moger/IPLOracle.git
   cd IPLOracle
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser**
   
   Navigate to `http://localhost:8501` to access the dashboard.

---

## 💻 Usage

### Match Simulator
1. Select **Team A** and **Team B** from the dropdown menus
2. Choose the **Venue** where the match will be played
3. Set the **Toss Winner** and **Toss Decision** (bat/field)
4. Click **"Predict Match"** to get win probabilities
5. (Optional) Enter market odds in the **Value Finder** to identify profitable betting opportunities

### Team Analysis
- Use the multi-select to compare Elo trajectories of multiple teams
- View current power rankings based on Elo ratings
- Select two teams to see their head-to-head historical record

### Venue Insights
- Adjust the minimum matches slider to filter venues
- Explore chase-friendly vs defend-friendly venues
- Analyze toss advantage patterns across different grounds

---

## 🧠 How It Works

### Elo Rating System
IPL Oracle implements a dynamic Elo rating system inspired by chess rankings:
- All teams start at a base rating of **1500**
- Ratings update after each match based on:
  - Match outcome (win/loss)
  - Expected vs actual result
  - Rating difference between teams
- K-factor of **32** ensures responsive rating changes

### Machine Learning Model
The prediction engine uses a **Random Forest Classifier** trained on features including:
- **Elo Ratings**: Current team strength indicators
- **Form Guide**: Recent match performance (last 5 games)
- **Home Advantage**: City-based home ground factor
- **Toss Factor**: Historical impact of toss decisions at specific venues
- **Head-to-Head**: Historical matchup performance

### Model Performance
- Trained on **1000+ IPL matches** (2008-2024)
- Test set accuracy: **~65-70%**
- Validated against 2023-24 season data

---

## 📁 Project Structure

```
IPLOracle/
├── app.py                 # Main Streamlit application
├── model_engine.py        # ML model, Elo system, and data processing
├── requirements.txt       # Python dependencies
├── ipl_model.pkl         # Trained model (auto-generated)
├── README.md             # Documentation
└── archive/
    └── Datasets/
        ├── matches_2008-2024.csv      # Match data
        └── deliveries_2008-2024.csv   # Ball-by-ball data
```

---

## 📊 Data Sources

The application uses IPL match data from 2008 to 2024, including:
- **Match Results**: Teams, venues, toss details, winners
- **Delivery Data**: Ball-by-ball information for detailed analysis

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3.8+** | Core programming language |
| **Streamlit** | Web application framework |
| **scikit-learn** | Machine learning models |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computations |
| **Plotly** | Interactive visualizations |

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](archive/LICENSE) file for details.

---

## 🙏 Acknowledgments

- IPL match data sourced from publicly available cricket statistics
- Inspired by Elo rating systems used in competitive sports
- Built with ❤️ for cricket fans everywhere

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Made with 🏏 by [Deepak Moger](https://github.com/Deepak-Moger)

</div>
