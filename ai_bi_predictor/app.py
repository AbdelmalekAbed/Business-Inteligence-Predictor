"""
AI-Driven Business Intelligence Predictor
=========================================
A comprehensive machine learning application for business forecasting and analytics.
Built for enterprise-level data science applications with focus on retail sales prediction.

Author: Data Science Portfolio Project
Target: PwC Technology Acceleration Center (TAC) Internship Application
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
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
    
    def __init__(self):
        self.models = {
            "Random Forest": {
                "model": RandomForestRegressor(random_state=42, n_jobs=-1),
                "params": {"n_estimators": [100, 200, 300], "max_depth": [10, 20, None]}
            },
            "Gradient Boosting": {
                "model": GradientBoostingRegressor(random_state=42),
                "params": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1, 0.15]}
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
            'date': date_range,  # Already datetime64[ns] from pd.date_range
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
        """Comprehensive data validation"""
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
        
        return len(issues) == 0, issues
    
    def add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced feature engineering for business intelligence"""
        df = df.copy()
        
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            
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
                                labels=['Winter', 'Spring', 'Summer', 'Fall'])
        
        # Business intelligence features
        if "marketing_spend" in df.columns and "sales" in df.columns:
            df["marketing_roi"] = df["sales"] / (df["marketing_spend"] + 1)
            df["marketing_efficiency"] = np.log1p(df["marketing_spend"])
        
        if "unemployment_rate" in df.columns:
            df["economic_health"] = np.where(df["unemployment_rate"] < 5, "Good", 
                                           np.where(df["unemployment_rate"] < 10, "Moderate", "Poor"))
        
        # Store performance metrics
        if "store_id" in df.columns and "sales" in df.columns:
            store_performance = df.groupby("store_id")["sales"].mean()
            df["store_avg_performance"] = df["store_id"].map(store_performance)
            df["store_performance_category"] = pd.qcut(df["store_avg_performance"], 
                                                     q=3, labels=["Low", "Medium", "High"])
        
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
        """Train model with comprehensive evaluation"""
        
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
        
        # Hyperparameter tuning (optional)
        if enable_tuning and model_config["params"]:
            param_grid = {f'model__{k}': v for k, v in model_config["params"].items()}
            grid_search = GridSearchCV(
                self.pipeline, param_grid, cv=3, scoring='r2', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            self.pipeline = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            self.pipeline.fit(X_train, y_train)
            best_params = None
        
        # Predictions
        y_pred_train = self.pipeline.predict(X_train)
        y_pred_test = self.pipeline.predict(X_test)
        
        # Comprehensive metrics
        train_metrics = self._calculate_metrics(y_train, y_pred_train)
        test_metrics = self._calculate_metrics(y_test, y_pred_test)
        
        # Cross-validation
        cv_scores = cross_val_score(self.pipeline, X_train, y_train, cv=5, scoring='r2')
        
        # Store feature names for later use
        self.feature_names = X.columns.tolist()
        self.training_X = X
        
        # Calculate feature importance using different methods for different models
        try:
            if hasattr(self.pipeline['model'], 'feature_importances_'):
                # For tree-based models like RandomForest and GradientBoosting
                importances = self.pipeline['model'].feature_importances_
                n_features = len(importances)
                feature_names_subset = self.processed_feature_names[:n_features] if len(self.processed_feature_names) >= n_features else self.processed_feature_names
                
                self.feature_importance = pd.DataFrame({
                    'Feature': feature_names_subset,
                    'Importance': importances[:len(feature_names_subset)]
                }).sort_values('Importance', ascending=False)
            else:
                # For linear models, use coefficient magnitude
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
        # Ensure inputs are numeric and handle any potential issues
        y_true = pd.to_numeric(y_true, errors='coerce').astype(float)
        y_pred = pd.to_numeric(y_pred, errors='coerce').astype(float)
        
        # Remove any NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        # Check if there's enough data to compute metrics
        if len(y_true_clean) < 2:
            logger.warning("Insufficient valid data for metrics calculation")
            return {
                'MAE': 0.0,
                'MSE': 0.0,
                'RMSE': 0.0,
                'R²': 0.0,
                'MAPE': 0.0
            }
        
        # Calculate MAPE safely
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
        
        # Model performance insight
        r2 = results['test_metrics']['R²']
        if r2 > 0.8:
            insights['performance'] = "🎯 Excellent model accuracy - High confidence in predictions"
        elif r2 > 0.6:
            insights['performance'] = "✅ Good model performance - Reliable for business decisions"
        elif r2 > 0.0:
            insights['performance'] = "⚠️ Moderate accuracy - Consider additional features or data"
        else:
            insights['performance'] = "❌ Poor model performance - Model requires retraining or more data"
        
        # Feature importance insights
        if self.feature_importance is not None and not self.feature_importance.empty:
            top_feature = self.feature_importance.iloc[0]['Feature']
            insights['top_driver'] = f"💡 Key business driver: {top_feature.replace('_', ' ').title()}"
        else:
            insights['top_driver'] = "💡 Feature analysis completed"
        
        # Data quality insights
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_pct < 5:
            insights['data_quality'] = "✨ Excellent data quality - Minimal missing values"
        else:
            insights['data_quality'] = f"⚠️ Data quality concern - {missing_pct:.1f}% missing values"
        
        return insights

def load_data_with_validation(path: str, uploaded_file=None) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """Load and validate data with comprehensive error handling and encoding detection"""
    try:
        if uploaded_file is not None:
            # Handle uploaded file with encoding detection
            try:
                # First, try UTF-8
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    # Reset file pointer and try with different encodings
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='latin-1')
                except UnicodeDecodeError:
                    try:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding='cp1252')
                    except UnicodeDecodeError:
                        try:
                            uploaded_file.seek(0)
                            df = pd.read_csv(uploaded_file, encoding='iso-8859-1')
                        except UnicodeDecodeError:
                            # Last resort - try with errors='ignore'
                            uploaded_file.seek(0)
                            df = pd.read_csv(uploaded_file, encoding='utf-8', errors='ignore')
            
        elif path.startswith('http'):
            predictor = BusinessIntelligencePredictor()
            df = predictor.generate_sample_data()
            return df, []
        
        else:
            # Handle file path with encoding detection
            if not Path(path).exists():
                return None, [f"File not found: {path}"]
            
            try:
                # First, try UTF-8
                df = pd.read_csv(path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(path, encoding='latin-1')
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(path, encoding='cp1252')
                    except UnicodeDecodeError:
                        try:
                            df = pd.read_csv(path, encoding='iso-8859-1')
                        except UnicodeDecodeError:
                            # Last resort
                            df = pd.read_csv(path, encoding='utf-8', errors='ignore')
        
        # Basic validation
        if df.empty:
            return None, ["Dataset is empty"]
        
        # Try to parse date column
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        if date_cols:
            try:
                df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors='coerce')
            except:
                return df, [f"Could not parse date column: {date_cols[0]}"]
        
        return df, []
    
    except Exception as e:
        return None, [f"Error loading data: {str(e)}"]

def create_advanced_visualizations(results: Dict, df: pd.DataFrame):
    """Create comprehensive visualization suite"""
    
    # Performance dashboard
    st.subheader("🎯 Model Performance Dashboard")
    
    # Create performance comparison
    metrics_df = pd.DataFrame([
        {'Dataset': 'Training', **results['train_metrics']},
        {'Dataset': 'Testing', **results['test_metrics']}
    ])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Metrics comparison - FIXED: Replace use_container_width with width
        fig_metrics = px.bar(
            metrics_df.melt(id_vars='Dataset', var_name='Metric', value_name='Value'),
            x='Metric', y='Value', color='Dataset',
            title="Training vs Testing Performance",
            barmode='group'
        )
        st.plotly_chart(fig_metrics, width='stretch')
    
    with col2:
        # Actual vs Predicted scatter with confidence bands - FIXED: Replace use_container_width with width
        fig_pred = px.scatter(
            x=results['y_test'], 
            y=results['y_pred_test'],
            title="Actual vs Predicted Sales",
            labels={'x': 'Actual Sales', 'y': 'Predicted Sales'}
        )
        
        # Add perfect prediction line
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
    
    # Time series analysis if date available
    if 'date' in df.columns:
        st.subheader("📈 Time Series Analysis")
        
        # Aggregate by date
        daily_sales = df.groupby('date')['sales'].sum().reset_index()
        
        fig_ts = px.line(
            daily_sales, x='date', y='sales',
            title="Sales Trend Over Time"
        )
        st.plotly_chart(fig_ts, width='stretch')
    
    # Feature importance visualization - FIXED
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
        
        # Categorize features
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_features = df.select_dtypes(include=['datetime64[ns]']).columns.tolist()
        
        # Remove target column from features
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
                        # Use try-except to handle non-numeric data
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
            # Apply scenario adjustments for numeric features
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
            
            # Ensure all required columns are present
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
            
            # Select and order features to match training
            pred_df_final = pred_df_processed[predictor.feature_names]
            
            # Clean prediction data
            for col in pred_df_final.columns:
                if pred_df_final[col].dtype in ['object', 'category']:
                    pred_df_final[col] = pred_df_final[col].astype(str).fillna('Unknown')
                else:
                    pred_df_final[col] = pd.to_numeric(pred_df_final[col], errors='coerce').fillna(0)
            
            # Make prediction
            prediction = predictor.pipeline.predict(pred_df_final)[0]
            
            # Calculate confidence interval
            rmse = predictor.performance_metrics.get('RMSE', 0)
            confidence_interval = 1.96 * rmse
            
            # Display results
            st.markdown(f"""
            <div class="prediction-result">
                <h2>🎯 Predicted {target_column.title()}</h2>
                <h1>${prediction:,.2f}</h1>
                <p><strong>Scenario:</strong> {scenario}</p>
                <p><strong>95% Confidence Interval:</strong> ${max(0, prediction - confidence_interval):,.2f} - ${prediction + confidence_interval:,.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional insights
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



def detect_target_column(df: pd.DataFrame) -> str:
    """Automatically detect the most likely target column"""
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Common target column names
    target_keywords = ['sales', 'revenue', 'price', 'amount', 'value', 'profit', 'target', 'y']
    
    # Look for columns with target keywords
    for keyword in target_keywords:
        matches = [col for col in df.columns if keyword.lower() in col.lower()]
        if matches:
            return matches[0]
    
    # If no keyword matches, return the first numeric column
    if numeric_columns:
        return numeric_columns[0]
    
    # Last resort - return the last column
    return df.columns[-1]

def display_column_info(df: pd.DataFrame):
    """Display column information to help user select target"""
    st.subheader("Dataset Column Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Numeric Columns (Possible Targets):**")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            col_stats = df[col].describe()
            st.write(f"- **{col}**: Mean={col_stats['mean']:.2f}, Range=[{col_stats['min']:.2f}, {col_stats['max']:.2f}]")
    
    with col2:
        st.write("**Categorical Columns:**")
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in cat_cols:
            unique_count = df[col].nunique()
            st.write(f"- **{col}**: {unique_count} unique values")

# Updated main function section for better error handling
def handle_dataset_loading_and_validation(data_source: str, uploaded_file) -> tuple[pd.DataFrame, list, str]:
    """Load and validate dataset, cleaning currency symbols and malformed numeric columns"""
    df = None
    errors = []
    suggested_target = None
    
    try:
        predictor = BusinessIntelligencePredictor()
        if data_source == "Retail Sales Dataset":
            df = predictor.generate_sample_data()
        elif uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        
        if df is None:
            errors.append("No dataset selected or uploaded")
            return None, errors, suggested_target
        
        # Clean numeric columns
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column contains currency symbols or can be numeric
                sample_values = df[col].dropna().head(5)
                if sample_values.empty:
                    continue
                # Try to convert strings to numeric after cleaning currency symbols
                try:
                    cleaned_series = df[col].str.replace(r'[₹$,]', '', regex=True).str.strip()
                    cleaned_series = cleaned_series.replace('', np.nan)  # Replace empty strings with NaN
                    numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
                    if numeric_series.notna().sum() > len(df) * 0.5:  # At least 50% convertible
                        df[col] = numeric_series
                        logger.info(f"Cleaned column {col} to numeric")
                    else:
                        logger.warning(f"Column {col} could not be fully converted to numeric")
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

# Updated sidebar configuration section
def create_sidebar_config(df, suggested_target="sales"):
    """Create sidebar configuration with dynamic target column detection"""
    
    with st.sidebar:
        st.header("⚙️ Configuration Panel")
        
        # Data source
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
        
        # Model configuration
        st.subheader("🤖 Model Settings")
        
        # Dynamic target column selection
        if df is not None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_columns:
                st.error("❌ No numeric columns found for prediction target")
                target_column = None
            else:
                # Set default target
                default_idx = 0
                if suggested_target in numeric_columns:
                    default_idx = numeric_columns.index(suggested_target)
                
                target_column = st.selectbox(
                    "Target Column:",
                    options=numeric_columns,
                    index=default_idx,
                    help="Select the column you want to predict"
                )
                
                # Show target column statistics
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
        
        # Training button
        run_analysis = st.button("🚀 Run Complete Analysis", type="primary")
        
        return data_source, uploaded_file, model_name, target_column, test_size, enable_tuning, run_analysis

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
    data_source, uploaded_file, model_name, target_column, test_size, enable_tuning, run_analysis = create_sidebar_config(
        st.session_state.get('df'), 
        suggested_target=suggested_target
    )
    
    # Main content area
    try:
        # Load and validate dataset
        df, errors, suggested_target = handle_dataset_loading_and_validation(data_source, uploaded_file)
        
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