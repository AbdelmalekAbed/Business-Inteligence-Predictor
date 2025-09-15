# AI-Driven Business Intelligence Predictor (Streamlit)

This project builds a predictive model and interactive dashboard to forecast sales and provide business insights.

## Features
- End-to-end ML pipeline (preprocessing → modeling → evaluation)
- Interactive Streamlit dashboard (actual vs predicted, feature importance)
- Synthetic dataset included (`data/retail_sales.csv`) so you can run immediately
- Ready for deployment to Streamlit Cloud

## Quickstart

```bash
# 1) Clone or unzip
cd ai_bi_predictor

# 2) Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Run the dashboard
streamlit run app.py
```

The app loads `data/retail_sales.csv` by default.

## Project Structure
```text
ai_bi_predictor/
├── app.py
├── requirements.txt
├── README.md
└── data/
    └── retail_sales.csv
```

## Deploy to Streamlit Cloud
1. Push this folder to a GitHub repo.
2. Go to Streamlit Community Cloud and create a new app, selecting your repo.
3. Set `app.py` as the entry point; it will auto-install `requirements.txt`.

## Notes
- Swap the synthetic CSV with a real dataset for your case study (same schema or update `app.py` feature definitions accordingly).
- You can add more models and metrics to compare performance (e.g., GradientBoosting).
