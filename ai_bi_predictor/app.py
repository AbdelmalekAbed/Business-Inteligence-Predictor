"""
AI-Driven Business Intelligence Predictor
=========================================
A comprehensive machine learning application for business forecasting and analytics.
Built for enterprise-level data science applications with focus on retail sales prediction.

Author: Data Science Portfolio Project
Target: PwC Technology Acceleration Center (TAC) Internship Application
"""
import joblib
import json
import os
import requests
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import logging
import warnings
warnings.filterwarnings('ignore')
import re
import chardet
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
from joblib import Parallel, delayed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI-Driven BI Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #000000 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #e1e5e9;
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .stAlert > div {
        border-radius: 10px;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        border-top: 1px solid #e1e5e9;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

class DataConnector:
    """Mock enterprise data connector for real-time updates"""
    
    def __init__(self):
        self.mock_sources = {
            "Sales API": "https://api.example.com/sales",
            "Marketing API": "https://api.example.com/marketing", 
            "Economic API": "https://api.example.com/economics"
        }
    
    def simulate_real_time_data(self, base_df, n_new_records=10):
        """Simulate new data arriving from APIs"""
        
        # Generate new records with realistic patterns
        latest_date = base_df['date'].max()
        new_dates = pd.date_range(
            start=latest_date + timedelta(days=1),
            periods=n_new_records,
            freq='D'
        )
        
        # Create new data with some trend
        new_data = []
        for date in new_dates:
            record = {
                'date': date,
                'store_id': np.random.randint(1, 21),
                'region': np.random.choice(['North', 'South', 'East', 'West', 'Central']),
                'product_category': np.random.choice(['Electronics', 'Clothing', 'Grocery', 'Home_Garden', 'Sports']),
                'marketing_spend': np.random.gamma(2, 500),
                'price_index': np.random.normal(100, 15),
                'unemployment_rate': np.random.normal(8, 3),
                'promotions': np.random.choice([0, 1], p=[0.7, 0.3])
            }
            
            # Generate sales with seasonal pattern
            day_of_year = date.timetuple().tm_yday
            seasonal_factor = 1 + 0.5 * np.sin(2 * np.pi * day_of_year / 365.25)
            weekend_factor = 1.5 if date.weekday() >= 5 else 1.0
            
            base_sales = (
                record['marketing_spend'] * 0.7 +
                (110 - record['unemployment_rate']) * 100 +
                record['price_index'] * 5 +
                record['promotions'] * 1500 +
                np.random.choice([800, 1200, 1500, 2000, 2500])
            )
            
            record['sales'] = max(100, base_sales * seasonal_factor * weekend_factor + np.random.normal(0, 500))
            new_data.append(record)
        
        return pd.DataFrame(new_data)
    
    def get_data_quality_metrics(self, df):
        """Calculate data quality metrics"""
        metrics = {
            'total_records': len(df),
            'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100,
            'data_freshness_hours': (datetime.now() - df['date'].max()).total_seconds() / 3600,
            'completeness_score': (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        }
        return metrics

def create_real_time_monitoring_dashboard(predictor, df):
    """Create real-time monitoring dashboard"""
    
    st.subheader("📡 Real-Time Data Monitoring")
    
    # Initialize session state for monitoring
    if 'last_update' not in st.session_state:
        st.session_state['last_update'] = datetime.now()
    if 'monitoring_active' not in st.session_state:
        st.session_state['monitoring_active'] = False
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Simulate Data Update"):
            connector = DataConnector()
            new_data = connector.simulate_real_time_data(df, n_new_records=5)
            
            # Update dataset
            updated_df = pd.concat([df, new_data], ignore_index=True)
            st.session_state['df'] = updated_df
            st.session_state['last_update'] = datetime.now()
            
            st.success(f"✅ Added {len(new_data)} new records")
            st.rerun()
    
    with col2:
        auto_refresh = st.checkbox("Auto-refresh (Demo)", value=False)
        if auto_refresh:
            time.sleep(30)  # In real app, use st.rerun() with timer
            st.rerun()
    
    with col3:
        st.write(f"**Last Update:** {st.session_state['last_update'].strftime('%H:%M:%S')}")
    
    # Data quality monitoring
    connector = DataConnector()
    quality_metrics = connector.get_data_quality_metrics(df)
    
    st.subheader("📊 Data Quality Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        completeness = quality_metrics['completeness_score']
        color = "green" if completeness > 95 else "orange" if completeness > 85 else "red"
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; border: 2px solid {color}; border-radius: 10px;">
            <h3 style="color: {color};">{completeness:.1f}%</h3>
            <p>Data Completeness</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        freshness = quality_metrics['data_freshness_hours']
        color = "green" if freshness < 24 else "orange" if freshness < 48 else "red"
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; border: 2px solid {color}; border-radius: 10px;">
            <h3 style="color: {color};">{freshness:.0f}h</h3>
            <p>Data Freshness</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        records = quality_metrics['total_records']
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; border: 2px solid blue; border-radius: 10px;">
            <h3 style="color: blue;">{records:,}</h3>
            <p>Total Records</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        missing_pct = quality_metrics['missing_percentage']
        color = "green" if missing_pct < 5 else "orange" if missing_pct < 15 else "red"
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; border: 2px solid {color}; border-radius: 10px;">
            <h3 style="color: {color};">{missing_pct:.1f}%</h3>
            <p>Missing Values</p>
        </div>
        """, unsafe_allow_html=True)

def create_api_documentation():
    """Create mock API documentation for enterprise integration"""
    
    st.subheader("🔌 API Integration Guide")
    
    with st.expander("📖 API Documentation", expanded=False):
        st.markdown("""
        ## Production Deployment APIs
        
        ### 1. Model Prediction API
        **POST** `/api/v1/predict`
        ```json
        {
            "features": {
                "marketing_spend": 1000,
                "price_index": 105.2,
                "unemployment_rate": 7.5,
                "promotions": 1,
                "store_id": 15,
                "region": "North"
            }
        }
        ```
        
        **Response:**
        ```json
        {
            "prediction": 25847.32,
            "confidence_interval": [23200.15, 28494.49],
            "model_version": "v2.1.3",
            "timestamp": "2024-01-15T10:30:00Z"
        }
        ```
        
        ### 2. Model Performance API
        **GET** `/api/v1/model/metrics`
        ```json
        {
            "r2_score": 0.847,
            "rmse": 2341.56,
            "mae": 1876.23,
            "last_training": "2024-01-10T14:22:00Z",
            "training_samples": 15420
        }
        ```
        
        ### 3. Data Health API
        **GET** `/api/v1/data/health`
        ```json
        {
            "status": "healthy",
            "completeness": 98.5,
            "freshness_hours": 2.3,
            "quality_score": 97.2,
            "alerts": []
        }
        ```
        
        ### 4. Batch Prediction API
        **POST** `/api/v1/predict/batch`
        - Upload CSV file with features
        - Returns CSV with predictions
        - Supports up to 10,000 records per request
        
        ### Authentication
        All APIs require Bearer token authentication:
        ```
        Authorization: Bearer <your-api-token>
        ```
        
        ### Rate Limits
        - Standard: 1000 requests/hour
        - Batch: 100 requests/hour
        - Premium: 10000 requests/hour
        """)
    
    # Mock deployment checklist
    with st.expander("✅ Production Deployment Checklist", expanded=False):
        checklist_items = [
            "Model validation on holdout dataset",
            "A/B testing framework setup", 
            "Monitoring and alerting configuration",
            "Data pipeline validation",
            "API security and authentication",
            "Load testing and performance optimization",
            "Backup and disaster recovery",
            "Documentation and training materials",
            "Regulatory compliance review",
            "Stakeholder approval and sign-off"
        ]
        
        for item in checklist_items:
            checked = st.checkbox(item, value=np.random.choice([True, False]), disabled=True)
    
    return True

class BusinessIntelligencePredictor:
    """
    Professional-grade Business Intelligence Predictor with enterprise features.
    
    Features:
    - Multiple ML algorithms with hyperparameter tuning
    - Comprehensive data validation and preprocessing
    - Advanced feature engineering
    - Interactive prediction interface
    - Business insights generation
    """
    
    def __init__(self, model_dir="models"):
        # Reduced param grids for faster tuning
        self.models = {
            "Random Forest": {
                "model": RandomForestRegressor(random_state=42, n_jobs=-1),
                "params": {"n_estimators": [50, 100], "max_depth": [10, 15]}
            },
            "Gradient Boosting": {
                "model": GradientBoostingRegressor(random_state=42),
                "params": {"n_estimators": [50, 100], "learning_rate": [0.1, 0.2]}
            },
            "Linear Regression": {
                "model": LinearRegression(),
                "params": {}
            }
        }
        self.pipeline = None
        self.feature_names = None
        self.performance_metrics = {}
        self.feature_importance = None
        self.processed_feature_names = None
        self.training_X = None
        
        # Model persistence
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
    
    def save_model(self, model_name: str = None):
        """Save trained model and metadata to disk"""
        if self.pipeline is None:
            raise ValueError("No trained model to save")
        
        if model_name is None:
            model_name = f"bi_model_{datetime.now().strftime('%Y%m%d_%H%M')}"
        
        model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
        metadata_path = os.path.join(self.model_dir, f"{model_name}_metadata.json")
        
        # Save model
        joblib.dump(self.pipeline, model_path)
        
        # Save metadata
        metadata = {
            "feature_names": self.feature_names,
            "processed_feature_names": self.processed_feature_names,
            "performance_metrics": self.performance_metrics,
            "training_date": datetime.now().isoformat(),
            "model_type": str(type(self.pipeline['model']).__name__)
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return model_path, metadata_path
    
    def load_model(self, model_path: str):
        """Load trained model and metadata from disk"""
        self.pipeline = joblib.load(model_path)
        
        # Load metadata if available
        metadata_path = model_path.replace('.joblib', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.feature_names = metadata.get("feature_names")
            self.processed_feature_names = metadata.get("processed_feature_names")
            self.performance_metrics = metadata.get("performance_metrics", {})
        
        return self.pipeline
    
    def get_available_models(self):
        """Get list of saved models"""
        if not os.path.exists(self.model_dir):
            return []
        
        model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.joblib')]
        models_info = []
        
        for model_file in model_files:
            metadata_file = model_file.replace('.joblib', '_metadata.json')
            metadata_path = os.path.join(self.model_dir, metadata_file)
            
            model_info = {"filename": model_file, "name": model_file.replace('.joblib', '')}
            
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    model_info.update(metadata)
                except:
                    pass
            
            models_info.append(model_info)
        
        return models_info
        
    def generate_sample_data(self, n_samples: int = 2000) -> pd.DataFrame:
        """Generate realistic retail sales dataset for demonstration"""
        np.random.seed(42)
        
        # Date range for 2 years
        start_date = datetime(2022, 1, 1)
        date_range = pd.date_range(start_date, periods=n_samples, freq='D')
        
        # Store and region data
        stores = np.random.randint(1, 21, n_samples)
        regions = np.random.choice(['North', 'South', 'East', 'West', 'Central'], n_samples)
        categories = np.random.choice(['Electronics', 'Clothing', 'Grocery', 'Home_Garden', 'Sports'], n_samples)
        
        # Economic indicators
        marketing_spend = np.random.gamma(2, 500, n_samples)
        price_index = np.random.normal(100, 15, n_samples)
        unemployment_rate = np.random.normal(8, 3, n_samples)
        promotions = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        
        # Seasonal patterns
        day_of_year = np.array([d.timetuple().tm_yday for d in date_range])
        seasonal_factor = 1 + 0.5 * np.sin(2 * np.pi * day_of_year / 365.25)
        
        # Weekend effect
        weekend_factor = np.array([1.5 if d.weekday() >= 5 else 1.0 for d in date_range])
        
        # Generate realistic sales
        base_sales = (
            marketing_spend * 0.7 +
            (110 - unemployment_rate) * 100 +
            price_index * 5 +
            promotions * 1500 +
            np.random.choice([800, 1200, 1500, 2000, 2500], n_samples)
        )
        
        sales = base_sales * seasonal_factor * weekend_factor + np.random.normal(0, 500, n_samples)
        sales = np.maximum(sales, 100)
        
        df = pd.DataFrame({
            'date': date_range,
            'store_id': stores,
            'region': regions,
            'product_category': categories,
            'marketing_spend': np.round(marketing_spend, 2),
            'price_index': np.round(price_index, 2),
            'unemployment_rate': np.round(unemployment_rate, 2),
            'promotions': promotions,
            'sales': np.round(sales, 2)
        })
        
        # Ensure date is datetime64[ns]
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Comprehensive data validation with enhanced checks for column consistency"""
        issues = []
        
        if df.empty:
            issues.append("Dataset is empty")
            return False, issues
        
        if len(df) < 100:
            issues.append("Dataset too small (minimum 100 records required)")
        
        # Check for excessive missing values
        missing_pct = (df.isnull().sum() / len(df)) * 100
        high_missing = missing_pct[missing_pct > 50]
        if not high_missing.empty:
            issues.append(f"High missing values in: {list(high_missing.index)}")
        
        # Check for date column
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        if not date_cols:
            issues.append("No date column found for time series analysis")
        
        # Numeric column validation (outliers and invalid values)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col.lower() in ['sales', 'marketing_spend', 'price_index'] and (df[col] < 0).any():
                issues.append(f"Invalid negative values found in {col}")
            
            # Outlier detection using IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            if len(outliers) > len(df) * 0.1:
                issues.append(f"Excessive outliers in {col}: {len(outliers)} values outside [{lower_bound:.2f}, {upper_bound:.2f}]")
            
            # Domain-specific checks
            if col.lower() == 'unemployment_rate' and df[col].notna().any():
                if (df[col] < 0).any() or (df[col] > 100).any():
                    issues.append(f"Unemployment rate in {col} has invalid values (must be 0-100)")
            if col.lower() == 'price_index' and df[col].notna().any():
                if (df[col] < 50).any() or (df[col] > 150).any():
                    issues.append(f"Price index in {col} has unrealistic values (expected 50-150)")
        
        # Categorical column validation
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            unique_count = df[col].nunique()
            if unique_count > len(df) * 0.5:
                issues.append(f"Column {col} has excessive unique values ({unique_count}), possible data entry errors")
            if unique_count == 1:
                issues.append(f"Column {col} has only one unique value, not useful for modeling")
        
        return len(issues) == 0, issues
    
    def add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced feature engineering for business intelligence"""
        df = df.copy()
        
        # Store training data statistics for features that require aggregation
        if hasattr(self, 'training_X') and self.training_X is not None:
            training_store_performance = self.training_X.groupby("store_id")["sales"].mean() if "store_id" in self.training_X.columns and "sales" in self.training_X.columns else None
        else:
            training_store_performance = None
        
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors='coerce')
            
            # Time-based features
            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month
            df["quarter"] = df["date"].dt.quarter
            df["dayofweek"] = df["date"].dt.dayofweek
            df["day_of_month"] = df["date"].dt.day
            df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
            df["is_month_end"] = (df["date"].dt.day >= 28).astype(int)
            
            # Seasonal indicators
            df["season"] = pd.cut(df["month"], bins=[0, 3, 6, 9, 12], 
                                labels=['Winter', 'Spring', 'Summer', 'Fall'], duplicates='drop')
        
        # Business intelligence features
        if "marketing_spend" in df.columns and "sales" in df.columns:
            df["marketing_roi"] = df["sales"] / (df["marketing_spend"] + 1)
            df["marketing_efficiency"] = np.log1p(df["marketing_spend"])
        
        if "unemployment_rate" in df.columns:
            df["economic_health"] = np.where(df["unemployment_rate"] < 5, "Good", 
                                        np.where(df["unemployment_rate"] < 10, "Moderate", "Poor"))
        
        # Store performance metrics
        if "store_id" in df.columns and "sales" in df.columns:
            if len(df) > 1:
                # For multi-row DataFrames, compute store performance normally
                store_performance = df.groupby("store_id")["sales"].mean()
                df["store_avg_performance"] = df["store_id"].map(store_performance)
                try:
                    df["store_performance_category"] = pd.qcut(df["store_avg_performance"], 
                                                            q=3, labels=["Low", "Medium", "High"], duplicates='drop')
                except Exception as e:
                    logger.warning(f"Failed to compute store_performance_category: {str(e)}. Assigning default 'Medium'.")
                    df["store_performance_category"] = "Medium"
            else:
                # For single-row DataFrames, use training data or default value
                if training_store_performance is not None and df["store_id"].iloc[0] in training_store_performance.index:
                    df["store_avg_performance"] = training_store_performance[df["store_id"].iloc[0]]
                    # Determine category based on training data quantiles
                    try:
                        quantiles = training_store_performance.quantile([0.33, 0.66])
                        if df["store_avg_performance"].iloc[0] <= quantiles[0.33]:
                            df["store_performance_category"] = "Low"
                        elif df["store_avg_performance"].iloc[0] <= quantiles[0.66]:
                            df["store_performance_category"] = "Medium"
                        else:
                            df["store_performance_category"] = "High"
                    except Exception as e:
                        logger.warning(f"Failed to assign store_performance_category: {str(e)}. Assigning default 'Medium'.")
                        df["store_performance_category"] = "Medium"
                else:
                    # Fallback for unseen store_id or no training data
                    df["store_avg_performance"] = df["sales"].iloc[0] if "sales" in df.columns else 0
                    df["store_performance_category"] = "Medium"
        
        return df
    
    def build_preprocessing_pipeline(self, X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
        """Build sophisticated preprocessing pipeline"""
        # Identify column types
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove date column from features if present
        if 'date' in categorical_features:
            categorical_features.remove('date')
        
        # Create preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ],
            remainder='drop',
            verbose_feature_names_out=False
        )
        
        return preprocessor, numeric_features, categorical_features
    
    def get_feature_names_after_preprocessing(self, preprocessor, numeric_features, categorical_features, X_sample):
        """Get feature names after preprocessing transformation"""
        try:
            # Fit the preprocessor to get feature names
            preprocessor.fit(X_sample)
            
            # Get feature names from transformers
            feature_names = []
            
            # Add numeric feature names
            feature_names.extend(numeric_features)
            
            # Add categorical feature names (one-hot encoded)
            if categorical_features:
                cat_transformer = preprocessor.named_transformers_['cat']
                cat_feature_names = cat_transformer.get_feature_names_out(categorical_features)
                feature_names.extend(cat_feature_names)
            
            return feature_names
        except Exception as e:
            logger.warning(f"Could not get feature names: {e}")
            return [f"feature_{i}" for i in range(len(numeric_features) + len(categorical_features) * 5)]
    
    def train_and_evaluate(self, df: pd.DataFrame, target_col: str, 
                      model_name: str, test_size: float = 0.2, 
                      enable_tuning: bool = False) -> Dict:
        """Train model with comprehensive evaluation and progress indicators"""
        
        # Feature engineering
        df_processed = self.add_advanced_features(df)
        
        # Prepare features and target
        X = df_processed.drop(columns=[target_col, 'date'], errors='ignore')
        y = df_processed[target_col]
        
        # Ensure categorical columns are strings
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            X[col] = X[col].astype(str).fillna('Unknown')
        
        # Handle missing values for numeric columns only
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if numeric_cols.size > 0:
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        
        # Handle missing values in target
        y = pd.to_numeric(y, errors='coerce').fillna(y.median())
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Build preprocessing pipeline
        preprocessor, numeric_features, categorical_features = self.build_preprocessing_pipeline(X_train)
        
        # Get processed feature names
        self.processed_feature_names = self.get_feature_names_after_preprocessing(
            preprocessor, numeric_features, categorical_features, X_train
        )
        
        # Get model
        model_config = self.models[model_name]
        model = model_config["model"]
        
        # Create complete pipeline
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Progress indicator for training
        with st.spinner("Training model..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Hyperparameter tuning (optional)
            if enable_tuning and model_config["params"]:
                param_grid = {f'model__{k}': v for k, v in model_config["params"].items()}
                grid_search = GridSearchCV(
                    self.pipeline, param_grid, cv=3, scoring='r2', n_jobs=-1
                )
                # Simulate progress for tuning
                for i in range(100):
                    time.sleep(0.01)  # Placeholder for actual computation
                    progress_bar.progress(i + 1)
                    status_text.text(f'Tuning: {i+1}% complete')
                grid_search.fit(X_train, y_train)
                self.pipeline = grid_search.best_estimator_
                best_params = grid_search.best_params_
                progress_bar.empty()
                status_text.empty()
            else:
                # Simple model fit
                for i in range(100):
                    time.sleep(0.005)  # Simulate training progress
                    progress_bar.progress(i + 1)
                    status_text.text(f'Training: {i+1}% complete')
                self.pipeline.fit(X_train, y_train)
                best_params = None
                progress_bar.empty()
                status_text.empty()
        
        # Predictions
        y_pred_train = self.pipeline.predict(X_train)
        y_pred_test = self.pipeline.predict(X_test)
        
        # Comprehensive metrics
        train_metrics = self._calculate_metrics(y_train, y_pred_train)
        test_metrics = self._calculate_metrics(y_test, y_pred_test)
        
        # Cross-validation with progress
        with st.spinner("Running cross-validation..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            cv_scores = cross_val_score(self.pipeline, X_train, y_train, cv=3, scoring='r2', n_jobs=-1)
            # Simulate progress for cross-validation
            for i in range(100):
                time.sleep(0.01)  # Approximate CV progress
                progress_bar.progress(i + 1)
                status_text.text(f'Cross-validation: {i+1}% complete')
            progress_bar.empty()
            status_text.empty()
        
        # Store feature names for later use
        self.feature_names = X.columns.tolist()
        self.training_X = X
        
        # Calculate feature importance
        try:
            if hasattr(self.pipeline['model'], 'feature_importances_'):
                importances = self.pipeline['model'].feature_importances_
                n_features = len(importances)
                feature_names_subset = self.processed_feature_names[:n_features] if len(self.processed_feature_names) >= n_features else self.processed_feature_names
                
                self.feature_importance = pd.DataFrame({
                    'Feature': feature_names_subset,
                    'Importance': importances[:len(feature_names_subset)]
                }).sort_values('Importance', ascending=False)
            else:
                if hasattr(self.pipeline['model'], 'coef_'):
                    coefficients = np.abs(self.pipeline['model'].coef_)
                    n_features = len(coefficients)
                    feature_names_subset = self.processed_feature_names[:n_features] if len(self.processed_feature_names) >= n_features else self.processed_feature_names
                    
                    self.feature_importance = pd.DataFrame({
                        'Feature': feature_names_subset,
                        'Importance': coefficients[:len(feature_names_subset)]
                    }).sort_values('Importance', ascending=False)
                else:
                    self.feature_importance = None
        except Exception as e:
            logger.warning(f"Could not calculate feature importance: {e}")
            self.feature_importance = None
        
        results = {
            'model': self.pipeline,
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'y_pred_train': y_pred_train, 'y_pred_test': y_pred_test,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_scores': cv_scores,
            'best_params': best_params,
            'numeric_features': numeric_features,
            'categorical_features': categorical_features,
            'feature_importance': self.feature_importance
        }
        
        self.performance_metrics = test_metrics
        return results
    
    def _calculate_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        y_true = pd.to_numeric(y_true, errors='coerce').astype(float)
        y_pred = pd.to_numeric(y_pred, errors='coerce').astype(float)
        
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) < 2:
            logger.warning("Insufficient valid data for metrics calculation")
            return {
                'MAE': 0.0,
                'MSE': 0.0,
                'RMSE': 0.0,
                'R²': 0.0,
                'MAPE': 0.0
            }
        
        mape = 0.0
        if len(y_true_clean) > 0 and not np.any(y_true_clean == 0):
            mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
        elif len(y_true_clean) > 0:
            logger.warning("Zero values in y_true_clean, setting MAPE to 0")
        
        return {
            'MAE': mean_absolute_error(y_true_clean, y_pred_clean) if len(y_true_clean) > 0 else 0.0,
            'MSE': mean_squared_error(y_true_clean, y_pred_clean) if len(y_true_clean) > 0 else 0.0,
            'RMSE': np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)) if len(y_true_clean) > 0 else 0.0,
            'R²': r2_score(y_true_clean, y_pred_clean) if len(y_true_clean) > 0 else 0.0,
            'MAPE': mape
        }
    
    def generate_business_insights(self, results: Dict, df: pd.DataFrame) -> Dict[str, str]:
        """Generate actionable business insights from model results"""
        insights = {}
        
        r2 = results['test_metrics']['R²']
        if r2 > 0.8:
            insights['performance'] = "🎯 Excellent model accuracy - High confidence in predictions"
        elif r2 > 0.6:
            insights['performance'] = "✅ Good model performance - Reliable for business decisions"
        elif r2 > 0.0:
            insights['performance'] = "⚠️ Moderate accuracy - Consider additional features or data"
        else:
            insights['performance'] = "❌ Poor model performance - Model requires retraining or more data"
        
        if self.feature_importance is not None and not self.feature_importance.empty:
            top_feature = self.feature_importance.iloc[0]['Feature']
            insights['top_driver'] = f"💡 Key business driver: {top_feature.replace('_', ' ').title()}"
        else:
            insights['top_driver'] = "💡 Feature analysis completed"
        
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_pct < 5:
            insights['data_quality'] = "✨ Excellent data quality - Minimal missing values"
        else:
            insights['data_quality'] = f"⚠️ Data quality concern - {missing_pct:.1f}% missing values"
        
        return insights

def load_file_with_encoding_fallback(uploaded_file, encodings=None):
    """Load a file with multiple encoding fallbacks"""
    if encodings is None:
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
    
    if hasattr(uploaded_file, 'read'):
        temp_path = "temp_uploaded_file.csv"
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        uploaded_file.seek(0)
        
        for encoding in encodings:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=encoding)
                logger.info(f"Successfully loaded file with encoding: {encoding}")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.warning(f"Error with encoding {encoding}: {str(e)}")
                continue
        
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='utf-8', errors='replace')
            logger.warning("Loaded file with error replacement - some characters may be lost")
            return df
        except Exception as e:
            logger.error(f"Failed to load file even with error replacement: {str(e)}")
            raise
    else:
        for encoding in encodings:
            try:
                df = pd.read_csv(uploaded_file, encoding=encoding)
                logger.info(f"Successfully loaded file with encoding: {encoding}")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.warning(f"Error with encoding {encoding}: {str(e)}")
                continue
        
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8', errors='replace')
            logger.warning("Loaded file with error replacement - some characters may be lost")
            return df
        except Exception as e:
            logger.error(f"Failed to load file even with error replacement: {str(e)}")
            raise

def detect_target_column(df: pd.DataFrame) -> str:
    """Automatically detect the most likely target column"""
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    target_keywords = ['sales', 'revenue', 'price', 'amount', 'value', 'profit', 'target', 'y']
    
    for keyword in target_keywords:
        matches = [col for col in df.columns if keyword.lower() in col.lower()]
        if matches:
            return matches[0]
    
    if numeric_columns:
        return numeric_columns[0]
    
    return df.columns[-1]

def display_column_info(df: pd.DataFrame):
    """
    Display comprehensive dataset column information in a clean, professional layout.
    
    Args:
        df (pd.DataFrame): The dataset to analyze and display
    """
    
    # Professional Header
    st.markdown("# 📊 Dataset Column Analysis")
    st.markdown("*Comprehensive overview of your dataset structure and statistics*")
    st.markdown("---")
    
    # Dataset Overview Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    with col1:
        st.metric("📋 Total Columns", len(df.columns))
    with col2:
        st.metric("📈 Numeric", len(numeric_cols))
    with col3:
        st.metric("📝 Categorical", len(categorical_cols))
    with col4:
        st.metric("📊 Total Records", f"{len(df):,}")
    
    st.markdown("---")
    
    # Analysis Overview
    with st.container():
        st.subheader("📋 Analysis Overview")
        st.info(
            "This analysis provides detailed insights into your dataset's structure. "
            "**Numeric columns** are potential prediction targets, while "
            "**categorical columns** serve as contextual features for modeling."
        )
    
    # Numeric Columns Section
    if numeric_cols:
        st.subheader("📈 Numeric Columns (Potential Prediction Targets)")
        
        # Create cards for numeric columns
        cols_per_row = 3
        for i in range(0, len(numeric_cols), cols_per_row):
            cols = st.columns(cols_per_row)
            
            for j, col in enumerate(numeric_cols[i:i+cols_per_row]):
                col_stats = df[col].describe()
                missing_pct = df[col].isna().mean() * 100
                
                # Check if potential target
                target_keywords = ['sales', 'revenue', 'price', 'amount', 'value', 'profit', 'target', 'y']
                is_potential_target = any(keyword in col.lower() for keyword in target_keywords)
                
                with cols[j]:
                    # Card container
                    with st.container():
                        if is_potential_target:
                            st.success(f"🎯 **{col.replace('_', ' ').title()}** (Potential Target)")
                        else:
                            st.info(f"📊 **{col.replace('_', ' ').title()}**")
                        
                        # Statistics in two columns
                        c1, c2 = st.columns(2)
                        with c1:
                            st.write(f"**Mean:** {col_stats['mean']:.2f}")
                            st.write(f"**Min:** {col_stats['min']:.2f}")
                            st.write(f"**Max:** {col_stats['max']:.2f}")
                        with c2:
                            st.write(f"**Std:** {col_stats['std']:.2f}")
                            st.write(f"**Median:** {col_stats['50%']:.2f}")
                            if missing_pct > 10:
                                st.error(f"**Missing:** {missing_pct:.1f}%")
                            elif missing_pct > 0:
                                st.warning(f"**Missing:** {missing_pct:.1f}%")
                            else:
                                st.success(f"**Missing:** {missing_pct:.1f}%")
                        
                        st.markdown("---")
        
        # Detailed Statistics
        with st.expander("📊 Detailed Numeric Statistics", expanded=False):
            stats_df = df[numeric_cols].describe().T
            stats_df['Missing %'] = df[numeric_cols].isna().mean() * 100
            stats_df = stats_df.round(2)
            
            # Color-code the dataframe
            def highlight_missing(val):
                if val > 20:
                    return 'background-color: #ffebee'
                elif val > 10:
                    return 'background-color: #fff3e0'
                else:
                    return 'background-color: #e8f5e8'
            
            st.dataframe(
                stats_df.style.applymap(highlight_missing, subset=['Missing %']),
                use_container_width=True
            )
    
    # Categorical Columns Section
    if categorical_cols:
        st.subheader("📝 Categorical Columns (Contextual Features)")
        
        # Create cards for categorical columns
        cols_per_row = 4
        for i in range(0, len(categorical_cols), cols_per_row):
            cols = st.columns(cols_per_row)
            
            for j, col in enumerate(categorical_cols[i:i+cols_per_row]):
                if j < len(cols):
                    unique_count = df[col].nunique()
                    missing_pct = df[col].isna().mean() * 100
                    most_common = df[col].mode().iloc[0] if not df[col].empty else "N/A"
                    
                    # Determine cardinality
                    if unique_count > len(df) * 0.8:
                        cardinality = "🔴 High"
                        card_type = "error"
                    elif unique_count > 10:
                        cardinality = "🟡 Medium"
                        card_type = "warning"
                    else:
                        cardinality = "🟢 Low"
                        card_type = "success"
                    
                    with cols[j]:
                        if card_type == "error":
                            st.error(f"📝 **{col.replace('_', ' ').title()}**")
                        elif card_type == "warning":
                            st.warning(f"📝 **{col.replace('_', ' ').title()}**")
                        else:
                            st.success(f"📝 **{col.replace('_', ' ').title()}**")
                        
                        st.write(f"**Unique:** {unique_count:,}")
                        st.write(f"**Cardinality:** {cardinality}")
                        st.write(f"**Most Common:** {str(most_common)[:15]}{'...' if len(str(most_common)) > 15 else ''}")
                        
                        if missing_pct > 10:
                            st.error(f"**Missing:** {missing_pct:.1f}%")
                        elif missing_pct > 0:
                            st.warning(f"**Missing:** {missing_pct:.1f}%")
                        else:
                            st.success(f"**Missing:** {missing_pct:.1f}%")
                        
                        st.markdown("---")
        
        # Detailed Categorical Statistics
        with st.expander("📊 Detailed Categorical Statistics", expanded=False):
            cat_data = []
            for col in categorical_cols:
                unique_count = df[col].nunique()
                missing_pct = df[col].isna().mean() * 100
                most_common = df[col].mode().iloc[0] if not df[col].empty else "N/A"
                
                # Cardinality assessment
                if unique_count > len(df) * 0.8:
                    cardinality = "High"
                elif unique_count > 10:
                    cardinality = "Medium"
                else:
                    cardinality = "Low"
                
                cat_data.append({
                    'Column': col,
                    'Unique Values': unique_count,
                    'Cardinality': cardinality,
                    'Most Common': str(most_common)[:30] + ('...' if len(str(most_common)) > 30 else ''),
                    'Missing %': missing_pct,
                    'Data Type': str(df[col].dtype)
                })
            
            cat_df = pd.DataFrame(cat_data)
            
            # Color coding function
            def color_cardinality(val):
                if val == "High":
                    return 'background-color: #ffebee'
                elif val == "Medium":
                    return 'background-color: #fff3e0'
                else:
                    return 'background-color: #e8f5e8'
            
            def color_missing(val):
                if val > 20:
                    return 'background-color: #ffebee'
                elif val > 10:
                    return 'background-color: #fff3e0'
                else:
                    return 'background-color: #e8f5e8'
            
            st.dataframe(
                cat_df.style.applymap(color_cardinality, subset=['Cardinality'])
                      .applymap(color_missing, subset=['Missing %']),
                use_container_width=True
            )
    
    # Recommendations Section
    st.markdown("---")
    st.subheader("💡 Recommendations")
    
    recommendations = []
    
    # Check for potential targets
    potential_targets = [col for col in numeric_cols 
                        if any(keyword in col.lower() 
                              for keyword in ['sales', 'revenue', 'price', 'amount', 'value', 'profit', 'target', 'y'])]
    if potential_targets:
        recommendations.append(f"🎯 Consider these columns for prediction: **{', '.join(potential_targets)}**")
    
    # Check for high missing data
    high_missing_cols = [col for col in df.columns if df[col].isna().mean() > 0.2]
    if high_missing_cols:
        recommendations.append(f"⚠️ Review columns with high missing data (>20%): **{', '.join(high_missing_cols)}**")
    
    # Check for high cardinality categoricals
    high_card_cols = [col for col in categorical_cols if df[col].nunique() > len(df) * 0.8]
    if high_card_cols:
        recommendations.append(f"🔴 High cardinality columns may need special encoding: **{', '.join(high_card_cols)}**")
    
    # Check for low cardinality categoricals
    low_card_cols = [col for col in categorical_cols if df[col].nunique() <= 10]
    if low_card_cols:
        recommendations.append(f"🟢 Good candidates for one-hot encoding: **{', '.join(low_card_cols)}**")
    
    if not recommendations:
        recommendations.append("✅ Your dataset looks well-structured with no major issues detected!")
    
    for rec in recommendations:
        st.markdown(f"• {rec}")
    
    st.markdown("---")
    st.caption("💡 Use this analysis to guide your feature selection and data preprocessing decisions.")

def handle_dataset_loading_and_validation(data_source: str, uploaded_file, numeric_conversion_threshold=0.3, max_rows=100000) -> tuple[pd.DataFrame, list, str]:
    """Load and validate dataset, cleaning currency symbols and malformed numeric columns with configurable threshold and sampling for large datasets"""
    df = None
    errors = []
    suggested_target = None
    
    try:
        predictor = BusinessIntelligencePredictor()
        if data_source == "Retail Sales Dataset":
            df = predictor.generate_sample_data()
        elif uploaded_file is not None:
            df = load_file_with_encoding_fallback(uploaded_file)
        
        if df is None:
            errors.append("No dataset selected or uploaded")
            return None, errors, suggested_target
        
        # Sample for large datasets
        if len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
            logger.info(f"Sampled dataset to {max_rows} rows for performance")
            st.warning(f"⚠️ Dataset is large ({len(df):,}+ rows). Sampled to {max_rows:,} rows for faster processing.")
        
        # Clean numeric columns
        for col in df.columns:
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().head(5)
                if sample_values.empty:
                    continue
                try:
                    cleaned_series = df[col].str.replace(r'[₹$,]', '', regex=True).str.strip()
                    cleaned_series = cleaned_series.replace('', np.nan)
                    numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
                    if numeric_series.notna().sum() > len(df) * numeric_conversion_threshold:
                        df[col] = numeric_series
                        logger.info(f"Cleaned column {col} to numeric (threshold: {numeric_conversion_threshold*100}%)")
                    else:
                        logger.warning(f"Column {col} could not be converted to numeric (only {numeric_series.notna().sum()/len(df)*100:.1f}% valid)")
                except Exception as e:
                    logger.warning(f"Failed to clean column {col}: {str(e)}")
        
        # Convert date columns
        for col in df.columns:
            if df[col].dtype == 'object' and col.lower() in ['date', 'transaction_date', 'order_date']:
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    errors.append(f"Could not convert column '{col}' to datetime")
        
        # Validate dataset
        if len(df) < 10:
            errors.append("Dataset too small (minimum 10 records)")
        if len(df.columns) < 2:
            errors.append("Dataset must have at least 2 columns")
        
        # Detect suggested target
        suggested_target = detect_target_column(df)
        if suggested_target is None or suggested_target not in df.columns:
            errors.append("Could not detect a suitable target column")
        
        # Display column info
        display_column_info(df)
        
        # Additional validation for target
        if suggested_target and suggested_target in df.columns:
            if df[suggested_target].dtype not in ['int64', 'float64']:
                errors.append(f"Target column '{suggested_target}' must be numeric")
            if df[suggested_target].isna().sum() / len(df) > 0.5:
                errors.append(f"Target column '{suggested_target}' has too many missing values")
        
        return df, errors, suggested_target
    
    except Exception as e:
        errors.append(f"Failed to load dataset: {str(e)}")
        return None, errors, suggested_target

def create_advanced_visualizations(results: Dict, df: pd.DataFrame):
    """Create comprehensive visualization suite with sampling for large datasets"""
    
    sample_size = min(10000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df
    
    st.subheader("🎯 Model Performance Dashboard")
    
    metrics_df = pd.DataFrame([
        {'Dataset': 'Training', **results['train_metrics']},
        {'Dataset': 'Testing', **results['test_metrics']}
    ])
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_metrics = px.bar(
            metrics_df.melt(id_vars='Dataset', var_name='Metric', value_name='Value'),
            x='Metric', y='Value', color='Dataset',
            title="Training vs Testing Performance",
            barmode='group'
        )
        st.plotly_chart(fig_metrics, width='stretch')
    
    with col2:
        fig_pred = px.scatter(
            x=results['y_test'], 
            y=results['y_pred_test'],
            title="Actual vs Predicted Sales",
            labels={'x': 'Actual Sales', 'y': 'Predicted Sales'}
        )
        
        min_val, max_val = results['y_test'].min(), results['y_test'].max()
        fig_pred.add_trace(
            go.Scatter(
                x=[min_val, max_val], 
                y=[min_val, max_val],
                mode='lines', 
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            )
        )
        st.plotly_chart(fig_pred, width='stretch')
    
    if 'date' in df_sample.columns:
        st.subheader("📈 Time Series Analysis")
        
        daily_sales = df_sample.groupby('date')['sales'].sum().reset_index()
        
        fig_ts = px.line(
            daily_sales, x='date', y='sales',
            title="Sales Trend Over Time"
        )
        st.plotly_chart(fig_ts, width='stretch')
    
    if results.get('feature_importance') is not None and not results['feature_importance'].empty:
        st.subheader("🔍 Feature Importance Analysis")
        
        importance_df = results['feature_importance'].head(15)
        
        fig_importance = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Top 15 Feature Importance",
            color='Importance',
            color_continuous_scale='viridis'
        )
        fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_importance, width='stretch')
    else:
        st.info("Feature importance analysis not available for this model configuration")

def create_prediction_interface(predictor: BusinessIntelligencePredictor, df: pd.DataFrame, target_column: str):
    """Interactive prediction interface for new data points, adaptable to any dataset"""
    
    st.header(f"🔮 Interactive {target_column.title()} Prediction")
    
    with st.form("prediction_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)
        
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_features = df.select_dtypes(include=['datetime64[ns]']).columns.tolist()
        
        if target_column in numeric_features:
            numeric_features.remove(target_column)
        if target_column in categorical_features:
            categorical_features.remove(target_column)
        
        prediction_data = {}
        
        with col1:
            st.subheader("📅 Time Parameters")
            if date_features:
                prediction_date = st.date_input(
                    date_features[0].title(),
                    value=datetime.today() + timedelta(days=30),
                    min_value=datetime(2020, 1, 1),
                    max_value=datetime(2030, 12, 31)
                )
                prediction_data[date_features[0]] = pd.to_datetime(prediction_date)
            else:
                prediction_date = st.date_input(
                    "Prediction Date",
                    value=datetime.today() + timedelta(days=30),
                    min_value=datetime(2020, 1, 1),
                    max_value=datetime(2030, 12, 31)
                )
                prediction_data['date'] = pd.to_datetime(prediction_date)
        
            st.subheader("🔢 Input Features (1)")
            feature_subset = numeric_features[:len(numeric_features)//2] + categorical_features[:len(categorical_features)//2]
            for feature in feature_subset:
                try:
                    if feature in numeric_features:
                        default_val = float(df[feature].median()) if not df[feature].isna().all() else 0.0
                        min_val = float(df[feature].min()) if not df[feature].isna().all() else 0.0
                        max_val = float(df[feature].max()) if not df[feature].isna().all() else 10000.0
                        prediction_data[feature] = st.number_input(
                            feature.replace('_', ' ').title(),
                            min_value=min_val,
                            max_value=max_val,
                            value=default_val,
                            step=0.1 if df[feature].dtype == float else 1.0
                        )
                    elif feature in categorical_features:
                        unique_vals = sorted(df[feature].dropna().unique())
                        if unique_vals:
                            prediction_data[feature] = st.selectbox(
                                feature.replace('_', ' ').title(),
                                options=unique_vals
                            )
                        else:
                            prediction_data[feature] = st.text_input(
                                feature.replace('_', ' ').title(),
                                value="Unknown"
                            )
                except Exception as e:
                    st.warning(f"Could not process input for {feature}: {str(e)}")
                    prediction_data[feature] = 0 if feature in numeric_features else "Unknown"
        
        with col2:
            st.subheader("🔢 Input Features (2)")
            feature_subset = numeric_features[len(numeric_features)//2:] + categorical_features[len(categorical_features)//2:]
            for feature in feature_subset:
                try:
                    if feature in numeric_features:
                        default_val = float(df[feature].median()) if not df[feature].isna().all() else 0.0
                        min_val = float(df[feature].min()) if not df[feature].isna().all() else 0.0
                        max_val = float(df[feature].max()) if not df[feature].isna().all() else 10000.0
                        prediction_data[feature] = st.number_input(
                            feature.replace('_', ' ').title(),
                            min_value=min_val,
                            max_value=max_val,
                            value=default_val,
                            step=0.1 if df[feature].dtype == float else 1.0
                        )
                    elif feature in categorical_features:
                        unique_vals = sorted(df[feature].dropna().unique())
                        if unique_vals:
                            prediction_data[feature] = st.selectbox(
                                feature.replace('_', ' ').title(),
                                options=unique_vals
                            )
                        else:
                            prediction_data[feature] = st.text_input(
                                feature.replace('_', ' ').title(),
                                value="Unknown"
                            )
                except Exception as e:
                    st.warning(f"Could not process input for {feature}: {str(e)}")
                    prediction_data[feature] = 0 if feature in numeric_features else "Unknown"
        
        with col3:
            st.subheader("🎯 Strategy Inputs")
            promotions = st.selectbox(
                "Promotions Active",
                options=[0, 1],
                index=0
            )
            prediction_data['promotions'] = int(promotions)
            
            st.subheader("📊 Scenario Analysis")
            scenario = st.selectbox(
                "Business Scenario",
                ["Current", "Optimistic", "Pessimistic", "Custom"]
            )
        
        predict_button = st.form_submit_button("🚀 Generate Prediction", type="primary")
    
    if predict_button and predictor.pipeline is not None:
        try:
            if scenario == "Optimistic":
                for feature in numeric_features:
                    if feature in prediction_data:
                        prediction_data[feature] *= 1.2
            elif scenario == "Pessimistic":
                for feature in numeric_features:
                    if feature in prediction_data:
                        prediction_data[feature] *= 0.8
            
            pred_df = pd.DataFrame([prediction_data])
            pred_df_processed = predictor.add_advanced_features(pred_df)
            
            missing_features = set(predictor.feature_names) - set(pred_df_processed.columns)
            for feature in missing_features:
                if feature in predictor.training_X.columns:
                    if predictor.training_X[feature].dtype in ['object', 'category']:
                        mode_val = predictor.training_X[feature].mode().iloc[0] if not predictor.training_X[feature].mode().empty else 'Unknown'
                        pred_df_processed[feature] = mode_val
                    else:
                        pred_df_processed[feature] = predictor.training_X[feature].median()
                else:
                    pred_df_processed[feature] = 0
            
            pred_df_final = pred_df_processed[predictor.feature_names]
            
            for col in pred_df_final.columns:
                if pred_df_final[col].dtype in ['object', 'category']:
                    pred_df_final[col] = pred_df_final[col].astype(str).fillna('Unknown')
                else:
                    pred_df_final[col] = pd.to_numeric(pred_df_final[col], errors='coerce').fillna(0)
            
            prediction = predictor.pipeline.predict(pred_df_final)[0]
            
            rmse = predictor.performance_metrics.get('RMSE', 0)
            confidence_interval = 1.96 * rmse
            
            st.markdown(f"""
            <div class="prediction-result">
                <h2>🎯 Predicted {target_column.title()}</h2>
                <h1>${prediction:,.2f}</h1>
                <p><strong>Scenario:</strong> {scenario}</p>
                <p><strong>95% Confidence Interval:</strong> ${max(0, prediction - confidence_interval):,.2f} - ${prediction + confidence_interval:,.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                cost_features = [f for f in numeric_features if 'spend' in f.lower() or 'cost' in f.lower()]
                if cost_features:
                    cost_feature = cost_features[0]
                    cost_value = prediction_data.get(cost_feature, 1.0)
                    roi = prediction / cost_value if cost_value > 0 else 0
                    st.metric(f"{cost_feature.replace('_', ' ').title()} ROI", f"{roi:.2f}x")
                else:
                    st.metric("ROI", "N/A")
            
            with col2:
                model_confidence = predictor.performance_metrics.get('R²', 0)
                st.metric("Model Confidence", f"{model_confidence:.1%}")
            
            with col3:
                vs_average = ((prediction / df[target_column].mean()) - 1) * 100 if target_column in df.columns and not df[target_column].isna().all() else 0
                st.metric(f"vs Average {target_column.title()}", f"{vs_average:+.1f}%")
        
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.info("💡 Tip: Ensure all required features are provided and match the training data format")

def create_scenario_analysis_dashboard(predictor, df, target_column):
    """Advanced what-if analysis with real-time updates"""
    
    st.subheader("📊 Scenario Analysis Dashboard")
    
    if not st.session_state.get('model_trained', False):
        st.warning("Please train a model first to enable scenario analysis")
        return
    
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_features:
        numeric_features.remove(target_column)
    
    key_levers = ['marketing_spend', 'price_index', 'promotions', 'unemployment_rate']
    available_levers = [col for col in key_levers if col in numeric_features]
    
    st.write("**Adjust Key Business Levers:**")
    
    lever_values = {}
    col1, col2 = st.columns(2)
    
    with col1:
        for i, lever in enumerate(available_levers[:len(available_levers)//2]):
            min_val = float(df[lever].min())
            max_val = float(df[lever].max())
            default_val = float(df[lever].median())
            
            lever_values[lever] = st.slider(
                f"{lever.replace('_', ' ').title()}",
                min_value=min_val,
                max_value=max_val,
                value=default_val,
                step=(max_val - min_val) / 100,
                key=f"slider_{lever}"
            )
    
    with col2:
        for lever in available_levers[len(available_levers)//2:]:
            min_val = float(df[lever].min())
            max_val = float(df[lever].max())
            default_val = float(df[lever].median())
            
            lever_values[lever] = st.slider(
                f"{lever.replace('_', ' ').title()}",
                min_value=min_val,
                max_value=max_val,
                value=default_val,
                step=(max_val - min_val) / 100,
                key=f"slider_{lever}"
            )
    
    if lever_values:
        try:
            pred_data = {}
            
            for col in numeric_features:
                if col in lever_values:
                    pred_data[col] = lever_values[col]
                else:
                    pred_data[col] = df[col].median()
            
            categorical_features = df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_features:
                if col != target_column and col != 'date':
                    mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                    pred_data[col] = mode_val
            
            pred_data['date'] = pd.to_datetime('today')
            
            pred_df = pd.DataFrame([pred_data])
            pred_df_processed = predictor.add_advanced_features(pred_df)
            
            missing_features = set(predictor.feature_names) - set(pred_df_processed.columns)
            for feature in missing_features:
                if feature in predictor.training_X.columns:
                    if predictor.training_X[feature].dtype in ['object', 'category']:
                        mode_val = predictor.training_X[feature].mode().iloc[0] if not predictor.training_X[feature].mode().empty else 'Unknown'
                        pred_df_processed[feature] = mode_val
                    else:
                        pred_df_processed[feature] = predictor.training_X[feature].median()
            
            pred_df_final = pred_df_processed[predictor.feature_names]
            for col in pred_df_final.columns:
                if pred_df_final[col].dtype in ['object', 'category']:
                    pred_df_final[col] = pred_df_final[col].astype(str).fillna('Unknown')
                else:
                    pred_df_final[col] = pd.to_numeric(pred_df_final[col], errors='coerce').fillna(0)
            
            prediction = predictor.pipeline.predict(pred_df_final)[0]
            
            baseline_prediction = df[target_column].mean()
            change_pct = ((prediction / baseline_prediction) - 1) * 100
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    f"Predicted {target_column.title()}",
                    f"${prediction:,.0f}",
                    f"{change_pct:+.1f}% vs baseline"
                )
            
            with col2:
                st.metric("Baseline Average", f"${baseline_prediction:,.0f}")
            
            with col3:
                impact = prediction - baseline_prediction
                st.metric("Impact", f"${impact:+,.0f}")
            
            st.subheader("🔍 Sensitivity Analysis")
            
            sensitivity_data = []
            for lever in available_levers[:4]:
                base_val = lever_values[lever]
                
                for change_pct in [-20, -10, 10, 20]:
                    test_val = base_val * (1 + change_pct/100)
                    test_data = pred_data.copy()
                    test_data[lever] = test_val
                    
                    test_df = pd.DataFrame([test_data])
                    test_df_processed = predictor.add_advanced_features(test_df)
                    
                    for feature in missing_features:
                        if feature in predictor.training_X.columns:
                            if predictor.training_X[feature].dtype in ['object', 'category']:
                                mode_val = predictor.training_X[feature].mode().iloc[0] if not predictor.training_X[feature].mode().empty else 'Unknown'
                                test_df_processed[feature] = mode_val
                            else:
                                test_df_processed[feature] = predictor.training_X[feature].median()
                    
                    test_df_final = test_df_processed[predictor.feature_names]
                    for col in test_df_final.columns:
                        if test_df_final[col].dtype in ['object', 'category']:
                            test_df_final[col] = test_df_final[col].astype(str).fillna('Unknown')
                        else:
                            test_df_final[col] = pd.to_numeric(test_df_final[col], errors='coerce').fillna(0)
                    
                    test_prediction = predictor.pipeline.predict(test_df_final)[0]
                    
                    sensitivity_data.append({
                        'Lever': lever.replace('_', ' ').title(),
                        'Change %': change_pct,
                        'Predicted Sales': test_prediction,
                        'Impact': test_prediction - prediction
                    })
            
            sensitivity_df = pd.DataFrame(sensitivity_data)
            
            fig_sensitivity = px.bar(
                sensitivity_df,
                x='Change %',
                y='Impact',
                color='Lever',
                barmode='group',
                title="Sales Impact by Lever Changes",
                labels={'Impact': f'Change in {target_column.title()} ($)'}
            )
            
            st.plotly_chart(fig_sensitivity, width='stretch')
            
        except Exception as e:
            st.error(f"Scenario analysis failed: {str(e)}")

def monte_carlo_simulation(predictor, base_prediction_data, n_simulations=500):
    """Run Monte Carlo simulation for uncertainty quantification with parallelization"""
    
    st.subheader("🎲 Monte Carlo Risk Analysis")
    
    if not predictor.pipeline or not predictor.feature_names:
        st.error("❌ No trained model or feature names available. Train or load a model first.")
        return
    
    # Preprocess base_prediction_data with feature engineering
    try:
        pred_df = pd.DataFrame([base_prediction_data])
        pred_df_processed = predictor.add_advanced_features(pred_df)
        
        # Ensure all required features are present
        missing_features = set(predictor.feature_names) - set(pred_df_processed.columns)
        for feature in missing_features:
            if feature in predictor.training_X.columns:
                if predictor.training_X[feature].dtype in ['object', 'category']:
                    mode_val = predictor.training_X[feature].mode().iloc[0] if not predictor.training_X[feature].mode().empty else 'Unknown'
                    pred_df_processed[feature] = mode_val
                else:
                    pred_df_processed[feature] = predictor.training_X[feature].median()
            else:
                pred_df_processed[feature] = 0
        
        base_data = pred_df_processed[predictor.feature_names].iloc[0].to_dict()
        
        # Ensure correct data types
        for col in base_data:
            if col in predictor.training_X.columns:
                expected_type = predictor.training_X[col].dtype
                if str(expected_type).startswith(('object', 'category')):
                    base_data[col] = str(base_data[col])
                else:
                    try:
                        base_data[col] = float(base_data[col])
                    except (ValueError, TypeError):
                        st.error(f"❌ Invalid value for {col}. Expected numeric, got {base_data[col]}")
                        return
    except Exception as e:
        st.error(f"❌ Failed to preprocess base_prediction_data: {str(e)}")
        st.info("💡 Ensure all required features are provided and match the training data format.")
        return
    
    if st.button("Run Monte Carlo Simulation"):
        with st.spinner("Running simulations..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def single_simulation(sim_data, idx):
                try:
                    # Create a copy to avoid modifying the original
                    sim_data = sim_data.copy()
                    
                    # Perturb numeric features
                    for col in sim_data.keys():
                        if col != 'date' and col in predictor.training_X.select_dtypes(include=[np.number]).columns:
                            noise = np.random.normal(0, 0.1)
                            sim_data[col] = float(sim_data[col]) * (1 + noise)
                    
                    # Convert to DataFrame
                    pred_df = pd.DataFrame([sim_data])
                    
                    # Apply feature engineering
                    pred_df_processed = predictor.add_advanced_features(pred_df)
                    
                    # Add missing features from training
                    missing_features = set(predictor.feature_names) - set(pred_df_processed.columns)
                    for feature in missing_features:
                        if feature in predictor.training_X.columns:
                            if predictor.training_X[feature].dtype in ['object', 'category']:
                                mode_val = predictor.training_X[feature].mode().iloc[0] if not predictor.training_X[feature].mode().empty else 'Unknown'
                                pred_df_processed[feature] = mode_val
                            else:
                                pred_df_processed[feature] = predictor.training_X[feature].median()
                        else:
                            pred_df_processed[feature] = 0
                    
                    # Ensure correct feature order and types
                    pred_df_final = pred_df_processed[predictor.feature_names]
                    for col in pred_df_final.columns:
                        if pred_df_final[col].dtype in ['object', 'category']:
                            pred_df_final[col] = pred_df_final[col].astype(str).fillna('Unknown')
                        else:
                            pred_df_final[col] = pd.to_numeric(pred_df_final[col], errors='coerce').fillna(0)
                    
                    # Make prediction
                    prediction = predictor.pipeline.predict(pred_df_final)[0]
                    if not np.isfinite(prediction):
                        return np.nan
                    
                    return prediction
                except Exception as e:
                    logger.error(f"Simulation {idx} failed: {str(e)}")
                    return np.nan
            
            # Run simulations in parallel
            predictions = Parallel(n_jobs=-1, verbose=0)(
                delayed(single_simulation)(base_data.copy(), i) for i in range(n_simulations)
            )
            
            # Update progress bar
            for i in range(0, 100, int(100 / n_simulations * 10)):
                progress_bar.progress(min(i + 10, 100))
                status_text.text(f'Progress: {min(i + 10, 100)}%')
                time.sleep(0.05)
            
            progress_bar.empty()
            status_text.empty()
            
            # Filter valid predictions
            predictions = np.array([p for p in predictions if np.isfinite(p)])
            
            if len(predictions) == 0:
                st.error("❌ No valid predictions generated. Check model and data.")
                st.info("💡 Possible issues: mismatched feature names, invalid data types, or model pipeline errors.")
                logger.error("Monte Carlo simulation failed: No valid predictions")
                return
            
            # Calculate confidence intervals
            confidence_intervals = {
                '5th percentile': np.percentile(predictions, 5),
                '25th percentile': np.percentile(predictions, 25),
                'Median': np.percentile(predictions, 50),
                '75th percentile': np.percentile(predictions, 75),
                '95th percentile': np.percentile(predictions, 95)
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Risk Assessment:**")
                for label, value in confidence_intervals.items():
                    st.write(f"- {label}: ${value:,.0f}")
            
            with col2:
                fig_dist = px.histogram(
                    x=predictions,
                    nbins=50,
                    title=f"Prediction Distribution ({len(predictions)} simulations)",
                    labels={'x': 'Predicted Sales', 'count': 'Frequency'}
                )
                st.plotly_chart(fig_dist, width='stretch')
                
                # Display summary statistics
                st.write("**Simulation Summary:**")
                st.write(f"- Mean: ${np.mean(predictions):,.0f}")
                st.write(f"- Std Dev: ${np.std(predictions):,.0f}")
                st.write(f"- Success Rate: {(len(predictions) / n_simulations) * 100:.1f}%")

def create_sidebar_config(df, suggested_target="sales"):
    """Create sidebar configuration with dynamic target column detection and numeric threshold"""
    
    with st.sidebar:
        st.header("⚙️ Configuration Panel")
        
        st.subheader("📊 Data Source")
        data_source = st.radio(
            "Select data source:",
            ["Retail Sales Dataset", "Upload Custom CSV"],
            help="Use demo data for testing or upload your own CSV file"
        )
        
        uploaded_file = None
        if data_source == "Upload Custom CSV":
            uploaded_file = st.file_uploader(
                "Choose CSV file",
                type=['csv'],
                help="Upload a CSV file with date, features, and target variable"
            )
        
        st.subheader("🧹 Data Cleaning Settings")
        numeric_conversion_threshold = st.slider(
            "Numeric Conversion Threshold (%):",
            min_value=10.0,
            max_value=80.0,
            value=30.0,
            step=5.0,
            help="Minimum percentage of valid numeric values required to convert a column to numeric"
        ) / 100.0
        
        max_rows = st.slider(
            "Max Dataset Size (for sampling):",
            min_value=10000,
            max_value=500000,
            value=100000,
            step=10000,
            help="Sample large datasets to this size for better performance"
        )
        
        st.subheader("🤖 Model Settings")
        
        if df is not None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_columns:
                st.error("❌ No numeric columns found for prediction target")
                target_column = None
            else:
                default_idx = 0
                if suggested_target in numeric_columns:
                    default_idx = numeric_columns.index(suggested_target)
                
                target_column = st.selectbox(
                    "Target Column:",
                    options=numeric_columns,
                    index=default_idx,
                    help="Select the column you want to predict"
                )
                
                if target_column:
                    target_stats = df[target_column].describe()
                    st.write(f"**Target Stats:**")
                    st.write(f"Mean: {target_stats['mean']:.2f}")
                    st.write(f"Std: {target_stats['std']:.2f}")
                    st.write(f"Range: [{target_stats['min']:.2f}, {target_stats['max']:.2f}]")
        else:
            target_column = st.text_input(
                "Target Column:",
                value="sales",
                help="Name of the column to predict"
            )
        
        predictor = BusinessIntelligencePredictor()
        model_name = st.selectbox(
            "ML Algorithm:",
            options=list(predictor.models.keys()),
            help="Select the machine learning algorithm to use"
        )
        
        test_size = st.slider(
            "Test Set Size:",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            step=0.05,
            help="Percentage of data used for testing"
        )
        
        enable_tuning = st.checkbox(
            "Enable Hyperparameter Tuning",
            help="Automatically optimize model parameters (slower but better performance)"
        )
        
        run_analysis = st.button("🚀 Run Complete Analysis", type="primary")
        add_model_management_to_sidebar(predictor)
        return data_source, uploaded_file, model_name, target_column, test_size, enable_tuning, run_analysis, numeric_conversion_threshold, max_rows

def add_model_management_to_sidebar(predictor):
    """Add model save/load functionality to sidebar"""
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Model Management")
    
    if st.session_state.get('model_trained', False):
        model_name = st.sidebar.text_input(
            "Model Name", 
            f"sales_model_{datetime.now().strftime('%Y%m%d')}"
        )
        
        if st.sidebar.button("💾 Save Current Model"):
            try:
                model_path, metadata_path = predictor.save_model(model_name)
                st.sidebar.success(f"✅ Model saved!")
                st.sidebar.info(f"Path: {os.path.basename(model_path)}")
            except Exception as e:
                st.sidebar.error(f"❌ Save failed: {str(e)}")
    
    available_models = predictor.get_available_models()
    if available_models:
        st.sidebar.write("**Load Existing Model:**")
        
        model_options = ["Select a model..."] + [model["name"] for model in available_models]
        selected_model = st.sidebar.selectbox("Available Models", model_options)
        
        if selected_model != "Select a model..." and st.sidebar.button("📂 Load Model"):
            try:
                model_path = os.path.join(predictor.model_dir, f"{selected_model}.joblib")
                predictor.load_model(model_path)
                st.session_state['model_trained'] = True
                st.session_state['predictor'] = predictor
                st.sidebar.success("✅ Model loaded successfully!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"❌ Load failed: {str(e)}")
        
        if selected_model != "Select a model...":
            model_info = next((m for m in available_models if m["name"] == selected_model), None)
            if model_info:
                st.sidebar.write("**Model Info:**")
                if "training_date" in model_info:
                    training_date = datetime.fromisoformat(model_info["training_date"]).strftime("%Y-%m-%d %H:%M")
                    st.sidebar.write(f"- Trained: {training_date}")
                if "model_type" in model_info:
                    st.sidebar.write(f"- Type: {model_info['model_type']}")
                if "performance_metrics" in model_info and model_info["performance_metrics"]:
                    r2 = model_info["performance_metrics"].get("R²", 0)
                    st.sidebar.write(f"- R² Score: {r2:.3f}")

def main():
    """Main application interface"""
    
    # Initialize session state
    if 'model_trained' not in st.session_state:
        st.session_state['model_trained'] = False
    if 'predictor' not in st.session_state:
        st.session_state['predictor'] = BusinessIntelligencePredictor()
    if 'df' not in st.session_state:
        st.session_state['df'] = None
    if 'results' not in st.session_state:
        st.session_state['results'] = None
    
    # Use predictor from session state
    predictor = st.session_state['predictor']
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 AI-Driven Business Intelligence Predictor</h1>
        <p>Enterprise-Grade Sales Forecasting & Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Determine suggested target based on existing dataframe or default
    suggested_target = "sales"
    if st.session_state.get('df') is not None:
        suggested_target = detect_target_column(st.session_state['df'])
    
    # Sidebar configuration
    data_source, uploaded_file, model_name, target_column, test_size, enable_tuning, run_analysis, numeric_conversion_threshold, max_rows = create_sidebar_config(
        st.session_state.get('df'), 
        suggested_target=suggested_target
    )
    
    # Main content area
    try:
        # Load and validate dataset
        df, errors, suggested_target = handle_dataset_loading_and_validation(data_source, uploaded_file, numeric_conversion_threshold, max_rows)
        
        if df is None:
            for error in errors:
                st.error(f"❌ {error}")
            st.stop()
        
        st.session_state['df'] = df
        if errors:
            for error in errors:
                st.warning(f"⚠️ {error}")
        
        # Data overview
        st.header("📊 Dataset Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("📋 Total Records", f"{len(df):,}")
        with col2:
            st.metric("📈 Features", len(df.columns) - 1)
        with col3:
            st.metric("❓ Missing Values", f"{df.isnull().sum().sum():,}")
        with col4:
            st.metric("📅 Date Range", f"{(df['date'].max() - df['date'].min()).days} days" if 'date' in df.columns else "N/A")
        with col5:
            target_col = target_column if target_column in df.columns else suggested_target
            st.metric(f"💰 Avg {target_col.title()}", f"${df[target_col].mean():,.0f}" if target_col in df.columns else "N/A")
        
        # Data preview with filtering
        with st.expander("🔍 Interactive Data Explorer", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                if 'region' in df.columns:
                    selected_regions = st.multiselect("Filter by Region", df['region'].unique(), default=df['region'].unique())
                    df_filtered = df[df['region'].isin(selected_regions)]
                else:
                    df_filtered = df
            
            with col2:
                if 'product_category' in df.columns:
                    selected_categories = st.multiselect("Filter by Category", df['product_category'].unique(), default=df['product_category'].unique())
                    df_filtered = df_filtered[df_filtered['product_category'].isin(selected_categories)]
            
            # Convert date to string for display to avoid Arrow issues
            df_filtered_display = df_filtered.copy()
            if 'date' in df_filtered_display.columns:
                df_filtered_display['date'] = df_filtered_display['date'].dt.strftime('%Y-%m-%d')
            st.dataframe(df_filtered_display.head(20), width='stretch')
            
            # Quick statistics
            st.subheader("📊 Quick Statistics")
            numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
            stats_df = df_filtered[numeric_cols].describe().round(2)
            st.dataframe(stats_df, width='stretch')
            
            # Debug: Show dataset statistics
            st.write("Dataset statistics:")
            stats_df_all = df.describe(include='all').round(2)
            if 'date' in stats_df_all.columns:
                stats_df_all['date'] = stats_df_all['date'].astype(str)
            st.dataframe(stats_df_all)
            st.write(f"{target_col.title()} variance:", df[target_col].var() if target_col in df.columns else "N/A")
        
        # Exploratory Data Analysis
        st.header("📈 Exploratory Data Analysis")
        
        tab1, tab2, tab3 = st.tabs(["📊 Distributions", "🔗 Relationships", "📅 Time Trends"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_dist = px.histogram(
                    df, x=target_col,
                    nbins=50,
                    title=f"{target_col.title()} Distribution",
                    color_discrete_sequence=['#667eea']
                )
                fig_dist.update_layout(showlegend=False)
                st.plotly_chart(fig_dist, width='stretch')
            
            with col2:
                if 'region' in df.columns:
                    fig_box = px.box(
                        df, x='region', y=target_col,
                        title=f"{target_col.title()} Distribution by Region",
                        color='region'
                    )
                    st.plotly_chart(fig_box, width='stretch')
        
        with tab2:
            numeric_df = df.select_dtypes(include=[np.number])
            
            col1, col2 = st.columns(2)
            
            with col1:
                corr_matrix = numeric_df.corr()
                fig_corr = px.imshow(
                    corr_matrix,
                    title="Feature Correlation Matrix",
                    color_continuous_scale='RdBu_r',
                    aspect='auto'
                )
                st.plotly_chart(fig_corr, width='stretch')
            
            with col2:
                key_vars = [target_col, 'marketing_spend', 'price_index', 'unemployment_rate']
                available_vars = [var for var in key_vars if var in df.columns]
                
                if len(available_vars) >= 2:
                    fig_scatter_matrix = px.scatter_matrix(
                        df[available_vars].sample(min(1000, len(df))),
                        title="Key Variables Relationship Matrix"
                    )
                    st.plotly_chart(fig_scatter_matrix, width='stretch')
        
        with tab3:
            if 'date' in df.columns:
                monthly_data = df.copy()
                monthly_data['year_month'] = monthly_data['date'].dt.to_period('M')
                monthly_sales = monthly_data.groupby('year_month').agg({
                    target_col: ['sum', 'mean', 'count'],
                    'marketing_spend': 'sum'
                }).round(2)
                
                monthly_sales.columns = [f'Total_{target_col.title()}', f'Avg_{target_col.title()}', 'Transactions', 'Total_Marketing']
                monthly_sales = monthly_sales.reset_index()
                monthly_sales['year_month'] = monthly_sales['year_month'].astype(str)
                
                fig_time = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=(f'Total {target_col.title()}', f'Average {target_col.title()}', 'Transaction Count', 'Marketing Spend'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                fig_time.add_trace(
                    go.Scatter(x=monthly_sales['year_month'], y=monthly_sales[f'Total_{target_col.title()}'],
                              name=f'Total {target_col.title()}', line=dict(color='#667eea')),
                    row=1, col=1
                )
                
                fig_time.add_trace(
                    go.Scatter(x=monthly_sales['year_month'], y=monthly_sales[f'Avg_{target_col.title()}'],
                              name=f'Avg {target_col.title()}', line=dict(color='#764ba2')),
                    row=1, col=2
                )
                
                fig_time.add_trace(
                    go.Scatter(x=monthly_sales['year_month'], y=monthly_sales['Transactions'],
                              name='Transactions', line=dict(color='#f093fb')),
                    row=2, col=1
                )
                
                fig_time.add_trace(
                    go.Scatter(x=monthly_sales['year_month'], y=monthly_sales['Total_Marketing'],
                              name='Marketing', line=dict(color='#f5576c')),
                    row=2, col=2
                )
                
                fig_time.update_layout(height=600, title_text="Business Metrics Over Time")
                st.plotly_chart(fig_time, width='stretch')
        
        # Model training and results
        if run_analysis:
            st.header("🤖 Machine Learning Model Training")
            
            if target_column not in df.columns:
                st.error(f"❌ Target column '{target_column}' not found in dataset")
                st.info(f"Available columns: {', '.join(df.columns)}")
                st.stop()
            
            with st.spinner("🔄 Training model and generating insights..."):
                try:
                    results = predictor.train_and_evaluate(
                        df, target_column, model_name, test_size, enable_tuning
                    )
                    
                    st.session_state['model_trained'] = True
                    st.session_state['predictor'] = predictor
                    st.session_state['df'] = df
                    st.session_state['results'] = results
                    
                    logger.info(f"Performance metrics after training: {predictor.performance_metrics}")
                    
                    st.success("✅ Model training completed successfully!")
                    
                    st.subheader("📊 Model Performance Metrics")
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    metrics = results['test_metrics']
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>R² Score</h4>
                            <h2>{metrics['R²']:.3f}</h2>
                            <small>Variance Explained</small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>RMSE</h4>
                            <h2>${metrics['RMSE']:,.0f}</h2>
                            <small>Prediction Error</small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>MAE</h4>
                            <h2>${metrics['MAE']:,.0f}</h2>
                            <small>Average Error</small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>MAPE</h4>
                            <h2>{metrics['MAPE']:.1f}%</h2>
                            <small>Percentage Error</small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col5:
                        cv_mean = results['cv_scores'].mean()
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>CV Score</h4>
                            <h2>{cv_mean:.3f}</h2>
                            <small>Cross Validation</small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if results['best_params']:
                        st.subheader("⚙️ Optimized Parameters")
                        params_df = pd.DataFrame([results['best_params']]).T
                        params_df.columns = ['Best Value']
                        st.dataframe(params_df)
                    
                    create_advanced_visualizations(results, df)
                    
                    st.header("💡 AI-Generated Business Insights")
                    insights = predictor.generate_business_insights(results, df)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="insight-box">
                            <h4>🎯 Model Performance</h4>
                            <p>{insights.get('performance', 'Analysis complete')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="insight-box">
                            <h4>🔑 Key Driver</h4>
                            <p>{insights.get('top_driver', 'Feature analysis complete')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="insight-box">
                            <h4>📊 Data Quality</h4>
                            <p>{insights.get('data_quality', 'Data validation complete')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"❌ Model training failed: {str(e)}")
                    logger.error(f"Training error: {e}")
                    st.info("💡 Please check your data format and try again")
                    st.session_state['model_trained'] = False
                    st.stop()
        
        if st.session_state.get('model_trained', False):
            create_prediction_interface(
                st.session_state['predictor'],
                st.session_state['df'],
                target_column
            )
        
        if st.session_state.get('model_trained', False):
            create_scenario_analysis_dashboard(
                st.session_state['predictor'],
                st.session_state['df'],
                target_column
            )
            
            monte_carlo_simulation(
                st.session_state['predictor'],
                st.session_state['df'].iloc[0].to_dict(),  # Base data from first row
                n_simulations=500
            )
        
        if st.session_state.get('model_trained', False):
            st.header("📥 Export Results & Model")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Download Performance Report"):
                    if st.session_state['predictor'].performance_metrics:
                        metrics_df = pd.DataFrame([st.session_state['predictor'].performance_metrics])
                        csv = metrics_df.to_csv(index=False)
                        st.download_button(
                            "📄 Performance Report (CSV)",
                            data=csv,
                            file_name=f"model_performance_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error("❌ No performance metrics available. Please train the model first.")
                        logger.warning("Attempted to download performance report but metrics are empty")
            
            with col2:
                if st.button("🎯 Download Predictions"):
                    results = st.session_state.get('results', {})
                    if results:
                        predictions_df = pd.DataFrame({
                            'Actual': results['y_test'].reset_index(drop=True),
                            'Predicted': results['y_pred_test'],
                            'Residual': results['y_test'].reset_index(drop=True) - results['y_pred_test']
                        })
                        csv = predictions_df.to_csv(index=False)
                        st.download_button(
                            "🎯 Predictions (CSV)",
                            data=csv,
                            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv"
                        )
            
            with col3:
                if st.button("📋 Generate Model Summary"):
                    results = st.session_state.get('results', {})
                    summary = f"""
                    AI-Driven Business Intelligence Predictor
                    ========================================
                    
                    Model: {model_name}
                    Target: {target_column}
                    Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
                    
                    Performance Metrics:
                    - R² Score: {st.session_state['predictor'].performance_metrics.get('R²', 0):.3f}
                    - RMSE: ${st.session_state['predictor'].performance_metrics.get('RMSE', 0):,.2f}
                    - MAE: ${st.session_state['predictor'].performance_metrics.get('MAE', 0):,.2f}
                    - MAPE: {st.session_state['predictor'].performance_metrics.get('MAPE', 0):.1f}%
                    
                    Dataset Information:
                    - Records: {len(df):,}
                    - Features: {len(df.columns)-1}
                    - Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')} if 'date' in df.columns else 'N/A'
                    
                    Business Insights:
                    {chr(10).join([f"- {insight}" for insight in predictor.generate_business_insights(results, df).values()])}
                    """
                    
                    st.download_button(
                        "📋 Model Summary (TXT)",
                        data=summary,
                        file_name=f"model_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                        mime="text/plain"
                    )
    
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        logger.error(f"Application error: {e}")
        
        st.info("🔄 Switching to demo mode...")
        df = predictor.generate_sample_data()
        st.success("✅ Demo data loaded successfully!")
        st.session_state['df'] = df
        st.session_state['model_trained'] = False    
  
    

# Initialize session state
if __name__ == "__main__":
    if 'model_trained' not in st.session_state:
        st.session_state['model_trained'] = False
    
    main()