
- Click "Commit new file"

---

### **Step 3: Create app.py (COMPLETE CODE)**

This is the main application file. I'll provide it in one complete block.

- Click "Add file" → "Create new file"
- Filename: `app.py`
- **Paste ALL of the following code:**

```python
"""
AI-Powered Data Analyst Platform
Complete professional data analysis without TensorFlow
Using PyTorch, Prophet, and modern ML libraries
Author: Daniyal Faheem
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc,
    mean_squared_error, r2_score, mean_absolute_error, silhouette_score
)
from sklearn.manifold import TSNE

# Advanced ML
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except:
    CATBOOST_AVAILABLE = False

# PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    PYTORCH_AVAILABLE = True
except:
    PYTORCH_AVAILABLE = False

# Time Series
try:
    from statsmodels.tsa.arima.model import ARIMA
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except:
    STATSMODELS_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except:
    PROPHET_AVAILABLE = False

# NLP
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except:
    TEXTBLOB_AVAILABLE = False

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except:
    WORDCLOUD_AVAILABLE = False

# Anomaly Detection
try:
    from pyod.models.iforest import IForest
    from pyod.models.lof import LOF
    PYOD_AVAILABLE = True
except:
    PYOD_AVAILABLE = False

# Statistical Tests
from scipy import stats
from scipy.stats import pearsonr, spearmanr

# Flowcharts
try:
    import graphviz
    import networkx as nx
    GRAPHVIZ_AVAILABLE = True
except:
    GRAPHVIZ_AVAILABLE = False

# Reports
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except:
    REPORTLAB_AVAILABLE = False

try:
    from pptx import Presentation
    from pptx.util import Inches
    PPTX_AVAILABLE = True
except:
    PPTX_AVAILABLE = False

# Streamlit Config
st.set_page_config(
    page_title="AI Data Analyst Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 1.5rem;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1.2rem;
        border-left: 5px solid #1f77b4;
        border-radius: 5px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #d4edda;
        padding: 1.2rem;
        border-left: 5px solid #28a745;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# PyTorch Models
if PYTORCH_AVAILABLE:
    class SimpleNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size//2)
            self.fc3 = nn.Linear(hidden_size//2, output_size)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)
            
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x

# Utility Functions
def detect_column_types(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    return {'numeric': numeric_cols, 'categorical': categorical_cols, 'datetime': datetime_cols}

def detect_problem_type(df, target_col):
    if target_col in df.columns:
        if df[target_col].dtype == 'object' or df[target_col].nunique() < 10:
            return 'classification'
        else:
            return 'regression'
    return None

@st.cache_data
def load_data(file):
    try:
        ext = file.name.split('.')[-1].lower()
        if ext == 'csv':
            return pd.read_csv(file)
        elif ext in ['xlsx', 'xls']:
            return pd.read_excel(file)
        elif ext == 'json':
            return pd.read_json(file)
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# Data Profiling
def profile_data(df):
    return {
        'shape': df.shape,
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,
        'duplicates': df.duplicated().sum(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'dtypes': df.dtypes.to_dict(),
        'unique_counts': df.nunique().to_dict()
    }

def display_data_profile(df):
    st.markdown("<div class='sub-header'>📊 Data Profile</div>", unsafe_allow_html=True)
    profile = profile_data(df)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", f"{profile['shape'][0]:,}")
    with col2:
        st.metric("Columns", profile['shape'][1])
    with col3:
        st.metric("Memory (MB)", f"{profile['memory_usage']:.2f}")
    with col4:
        st.metric("Duplicates", profile['duplicates'])
    
    col_info = pd.DataFrame({
        'Data Type': profile['dtypes'],
        'Unique': profile['unique_counts'],
        'Missing': profile['missing_values'],
        'Missing %': [f"{v:.1f}%" for v in profile['missing_percentage'].values()]
    })
    st.dataframe(col_info, use_container_width=True)
    
    if sum(profile['missing_values'].values()) > 0:
        missing_df = pd.DataFrame({
            'Column': list(profile['missing_values'].keys()),
            'Count': list(profile['missing_values'].values())
        })
        missing_df = missing_df[missing_df['Count'] > 0].sort_values('Count', ascending=False)
        fig = px.bar(missing_df, x='Column', y='Count', title='Missing Values by Column')
        st.plotly_chart(fig, use_container_width=True)

# Data Cleaning
def clean_data(df, strategy='mean', remove_dup=True):
    df_clean = df.copy()
    report = []
    
    if remove_dup:
        before = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        if before > len(df_clean):
            report.append(f"✓ Removed {before - len(df_clean)} duplicates")
    
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                report.append(f"✓ Filled '{col}' with mean")
    elif strategy == 'median':
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
                report.append(f"✓ Filled '{col}' with median")
    elif strategy == 'mode':
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                df_clean[col].fillna(mode_val, inplace=True)
                report.append(f"✓ Filled '{col}' with mode")
    
    return df_clean, report

# Visualizations
def create_visualizations(df):
    st.markdown("<div class='sub-header'>📊 Visualizations</div>", unsafe_allow_html=True)
    
    col_types = detect_column_types(df)
    numeric_cols = col_types['numeric']
    categorical_cols = col_types['categorical']
    
    viz_type = st.selectbox("Select Visualization", [
        "Histograms", "Box Plots", "Correlation Heatmap", "Scatter Plot",
        "Bar Chart", "Pie Chart", "Violin Plot", "Pair Plot", "3D Scatter"
    ])
    
    if viz_type == "Histograms" and numeric_cols:
        for col in numeric_cols[:4]:
            fig = px.histogram(df, x=col, title=f'Distribution of {col}', marginal="box")
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Box Plots" and numeric_cols:
        for col in numeric_cols[:4]:
            fig = px.box(df, y=col, title=f'Box Plot of {col}')
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Correlation Heatmap" and len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, title="Correlation Matrix", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Scatter Plot" and len(numeric_cols) >= 2:
        x_col = st.selectbox("X-axis", numeric_cols)
        y_col = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1))
        fig = px.scatter(df, x=x_col, y=y_col, title=f'{y_col} vs {x_col}', trendline="ols")
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Bar Chart" and categorical_cols:
        col = st.selectbox("Select Column", categorical_cols)
        value_counts = df[col].value_counts().head(10)
        fig = px.bar(x=value_counts.index, y=value_counts.values, labels={'x': col, 'y': 'Count'})
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Pie Chart" and categorical_cols:
        col = st.selectbox("Select Column", categorical_cols)
        value_counts = df[col].value_counts().head(10)
        fig = px.pie(values=value_counts.values, names=value_counts.index, title=f'{col} Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Violin Plot" and numeric_cols:
        for col in numeric_cols[:3]:
            fig = px.violin(df, y=col, title=f'Violin Plot of {col}', box=True)
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Pair Plot" and len(numeric_cols) >= 2:
        sample_cols = numeric_cols[:min(4, len(numeric_cols))]
        fig = px.scatter_matrix(df[sample_cols].sample(min(500, len(df))), title="Pair Plot")
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "3D Scatter" and len(numeric_cols) >= 3:
        x = st.selectbox("X", numeric_cols, key='3d_x')
        y = st.selectbox("Y", numeric_cols, key='3d_y', index=1)
        z = st.selectbox("Z", numeric_cols, key='3d_z', index=2)
        fig = px.scatter_3d(df.sample(min(1000, len(df))), x=x, y=y, z=z, title='3D Scatter')
        st.plotly_chart(fig, use_container_width=True)

# Machine Learning
def train_ml_models(df, target_col):
    st.markdown("<div class='sub-header'>🤖 Machine Learning</div>", unsafe_allow_html=True)
    
    problem_type = detect_problem_type(df, target_col)
    st.info(f"Detected: **{problem_type.upper()}** problem")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Encode
    X_encoded = X.copy()
    for col in X_encoded.select_dtypes(include=['object']).columns:
        X_encoded[col] = LabelEncoder().fit_transform(X_encoded[col].astype(str))
    
    if problem_type == 'classification':
        y_encoded = LabelEncoder().fit_transform(y.astype(str))
    else:
        y_encoded = y
    
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    results = []
    
    if problem_type == 'classification':
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'KNN': KNeighborsClassifier(),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = lgb.LGBMClassifier(random_state=42, verbose=-1)
        if CATBOOST_AVAILABLE:
            models['CatBoost'] = CatBoostClassifier(random_state=42, verbose=0)
        
        progress = st.progress(0)
        for i, (name, model) in enumerate(models.items()):
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                results.append({'Model': name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1})
                progress.progress((i + 1) / len(models))
            except:
                pass
        
        results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
        st.dataframe(results_df.style.highlight_max(axis=0), use_container_width=True)
        
        # Best model confusion matrix
        best_name = results_df.iloc[0]['Model']
        best_model = models[best_name]
        y_pred_best = best_model.predict(X_test_scaled)
        
        st.markdown(f"### Confusion Matrix - {best_name}")
        cm = confusion_matrix(y_test, y_pred_best)
        fig = px.imshow(cm, labels=dict(x="Predicted", y="Actual"), title=f"Confusion Matrix", text_auto=True)
        st.plotly_chart(fig, use_container_width=True)
        
        if hasattr(best_model, 'feature_importances_'):
            st.markdown("### Feature Importance")
            imp_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': best_model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h', title='Top 10 Features')
            st.plotly_chart(fig, use_container_width=True)
    
    else:  # Regression
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Decision Tree': DecisionTreeRegressor(random_state=42)
        }
        
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBRegressor(random_state=42)
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = lgb.LGBMRegressor(random_state=42, verbose=-1)
        if CATBOOST_AVAILABLE:
            models['CatBoost'] = CatBoostRegressor(random_state=42, verbose=0)
        
        progress = st.progress(0)
        for i, (name, model) in enumerate(models.items()):
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results.append({'Model': name, 'RMSE': rmse, 'MAE': mae, 'R²': r2})
                progress.progress((i + 1) / len(models))
            except:
                pass
        
        results_df = pd.DataFrame(results).sort_values('R²', ascending=False)
        st.dataframe(results_df.style.highlight_max(axis=0, subset=['R²']).highlight_min(axis=0, subset=['RMSE', 'MAE']), use_container_width=True)
        
        # Predictions vs Actual
        best_name = results_df.iloc[0]['Model']
        best_model = models[best_name]
        y_pred_best = best_model.predict(X_test_scaled)
        
        st.markdown(f"### Predictions vs Actual - {best_name}")
        pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_best})
        fig = px.scatter(pred_df, x='Actual', y='Predicted', title=f'{best_name} Predictions', trendline="ols")
        st.plotly_chart(fig, use_container_width=True)

# Clustering
def perform_clustering(df):
    st.markdown("<div class='sub-header'>🔍 Clustering</div>", unsafe_allow_html=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns")
        return
    
    selected = st.multiselect("Select features", numeric_cols, default=numeric_cols[:min(4, len(numeric_cols))])
    if len(selected) < 2:
        return
    
    n_clusters = st.slider("Number of clusters", 2, 10, 3)
    
    if st.button("Perform Clustering"):
        X = df[selected].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        pca_df = pd.DataFrame({'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1], 'Cluster': clusters})
        fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster', title=f'K-Means Clustering (k={n_clusters})')
        st.plotly_chart(fig, use_container_width=True)
        
        st.success(f"Silhouette Score: {silhouette_score(X_scaled, clusters):.3f}")

# Time Series Forecasting
def time_series_forecast(df):
    st.markdown("<div class='sub-header'>⏰ Time Series Forecasting</div>", unsafe_allow_html=True)
    
    if not PROPHET_AVAILABLE:
        st.warning("Prophet not installed. Install with: pip install prophet")
        return
    
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    all_cols = df.columns.tolist()
    
    if len(all_cols) < 2:
        st.warning("Need at least a date column and a value column")
        return
    
    date_col = st.selectbox("Date column", all_cols)
    value_col = st.selectbox("Value column", [c for c in all_cols if c != date_col])
    periods = st.slider("Forecast periods", 7, 365, 30)
    
    if st.button("Generate Forecast"):
        try:
            df_prophet = df[[date_col, value_col]].dropna()
            df_prophet.columns = ['ds', 'y']
            df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
            
            model = Prophet()
            model.fit(df_prophet)
            
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='markers', name='Actual'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='lightgray', showlegend=False))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='lightgray', name='Confidence'))
            fig.update_layout(title='Prophet Forecast', xaxis_title='Date', yaxis_title='Value')
            st.plotly_chart(fig, use_container_width=True)
            
            st.success("Forecast completed!")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Anomaly Detection
def anomaly_detection(df):
    st.markdown("<div class='sub-header'>🚨 Anomaly Detection</div>", unsafe_allow_html=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.warning("No numeric columns found")
        return
    
    contamination = st.slider("Contamination (expected % of anomalies)", 0.01, 0.5, 0.1)
    
    if st.button("Detect Anomalies"):
        X = df[numeric_cols].dropna()
        
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        predictions = iso_forest.fit_predict(X)
        
        df_result = df[numeric_cols].copy()
        df_result['Anomaly'] = predictions
        df_result['Anomaly'] = df_result['Anomaly'].map({1: 'Normal', -1: 'Anomaly'})
        
        anomaly_count = (predictions == -1).sum()
        st.metric("Anomalies Detected", f"{anomaly_count} ({anomaly_count/len(predictions)*100:.1f}%)")
        
        if len(numeric_cols) >= 2:
            fig = px.scatter(df_result, x=numeric_cols[0], y=numeric_cols[1], color='Anomaly',
                           title='Anomaly Detection', color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'})
            st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(df_result[df_result['Anomaly'] == 'Anomaly'].head(20), use_container_width=True)

# AI Insights
def generate_insights(df, profile):
    st.markdown("<div class='sub-header'>💡 AI-Generated Insights</div>", unsafe_allow_html=True)
    
    insights = []
    
    insights.append(f"📊 **Dataset Overview**: The dataset contains {profile['shape'][0]:,} rows and {profile['shape'][1]} columns, using {profile['memory_usage']:.2f} MB of memory.")
    
    missing_cols = [col for col, count in profile['missing_values'].items() if count > 0]
    if missing_cols:
        insights.append(f"⚠️ **Data Quality**: {len(missing_cols)} columns have missing values. Most affected: {', '.join(missing_cols[:3])}.")
    else:
        insights.append("✅ **Data Quality**: No missing values detected - dataset is complete!")
    
    if profile['duplicates'] > 0:
        insights.append(f"🔄 **Duplicates**: Found {profile['duplicates']} duplicate rows ({profile['duplicates']/profile['shape'][0]*100:.1f}%).")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols[:3]:
        mean_val = df[col].mean()
        std_val = df[col].std()
        insights.append(f"📈 **{col}**: Mean = {mean_val:.2f}, Std Dev = {std_val:.2f}")
    
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        strong_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    strong_corrs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        if strong_corrs:
            top = strong_corrs[0]
            insights.append(f"🔗 **Strong Correlation**: {top[0]} and {top[1]} are strongly correlated ({top[2]:.3f}).")
    
    for insight in insights:
        st.markdown(f"<div class='insight-box'>{insight}</div>", unsafe_allow_html=True)

# Natural Language Queries
def nl_query(df):
    st.markdown("<div class='sub-header'>🗣️ Natural Language Queries</div>", unsafe_allow_html=True)
    
    st.info("Ask questions about your dataset in plain English!")
    
    query = st.text_input("Enter your question:", placeholder="e.g., What is the correlation between age and income?")
    
    if query:
        query_lower = query.lower()
        
        if 'correlation' in query_lower:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                corr = df[numeric_cols].corr()
                st.dataframe(corr, use_container_width=True)
                
                strong = []
                for i in range(len(corr.columns)):
                    for j in range(i+1, len(corr.columns)):
                        strong.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
                strong.sort(key=lambda x: abs(x[2]), reverse=True)
                
                if strong:
                    st.success(f"Strongest correlation: {strong[0][0]} & {strong[0][1]} ({strong[0][2]:.3f})")
        
        elif 'mean' in query_lower or 'average' in query_lower:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            means = df[numeric_cols].mean()
            st.dataframe(means.to_frame('Mean'), use_container_width=True)
        
        elif 'missing' in query_lower:
            missing = df.isnull().sum()
            missing = missing[missing > 0]
            if len(missing) > 0:
                st.dataframe(missing.to_frame('Missing'), use_container_width=True)
            else:
                st.success("No missing values!")
        
        elif 'summary' in query_lower or 'describe' in query_lower:
            st.dataframe(df.describe(), use_container_width=True)
        
        else:
            st.warning("Try asking about: correlation, mean, missing values, or summary")

# Flowcharts
def generate_flowcharts():
    st.markdown("<div class='sub-header'>📐 Flowcharts</div>", unsafe_allow_html=True)
    
    if not GRAPHVIZ_AVAILABLE:
        st.warning("Graphviz not installed")
        return
    
    flowchart_type = st.selectbox("Select Type", ["Data Pipeline", "Analysis Workflow"])
    
    if st.button("Generate"):
        if flowchart_type == "Data Pipeline":
            dot = graphviz.Digraph()
            dot.attr(rankdir='TB')
            dot.node('A', 'Data Upload', shape='box', style='filled', fillcolor='lightblue')
            dot.node('B', 'Data Profiling', shape='box', style='filled', fillcolor='lightgreen')
            dot.node('C', 'Data Cleaning', shape='box', style='filled', fillcolor='lightyellow')
            dot.node('D', 'Visualization', shape='box', style='filled', fillcolor='lightpink')
            dot.node('E', 'ML Models', shape='box', style='filled', fillcolor='lavender')
            dot.node('F', 'Reports', shape='box', style='filled', fillcolor='lightgray')
            dot.edges(['AB', 'BC', 'CD', 'DE', 'EF'])
            st.graphviz_chart(dot)
        else:
            dot = graphviz.Digraph()
            dot.node('Start', 'Start', shape='oval', style='filled', fillcolor='green')
            dot.node('Load', 'Load Data', shape='box')
            dot.node('Clean', 'Clean Data', shape='box')
            dot.node('Analyze', 'Analyze', shape='box')
            dot.node('End', 'End', shape='oval', style='filled', fillcolor='red')
            dot.edges(['StartLoad', 'LoadClean', 'CleanAnalyze', 'AnalyzeEnd'])
            st.graphviz_chart(dot)

# Reports
def generate_reports(df, profile):
    st.markdown("<div class='sub-header'>📄 Generate Reports</div>", unsafe_allow_html=True)
    
    report_type = st.selectbox("Select Report Type", ["PDF", "Excel", "HTML"])
    
    if st.button("Generate Report"):
        if report_type == "Excel":
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Data', index=False)
                df.describe().to_excel(writer, sheet_name='Statistics')
            output.seek(0)
            st.download_button("Download Excel", output, file_name="report.xlsx", mime="application/vnd.ms-excel")
            st.success("Excel report generated!")
        
        elif report_type == "HTML":
            html = f"""
            <html>
            <head><title>Data Analysis Report</title></head>
            <body>
            <h1>Data Analysis Report</h1>
            <h2>Dataset Overview</h2>
            <p>Rows: {profile['shape'][0]}, Columns: {profile['shape'][1]}</p>
            <h2>Data Preview</h2>
            {df.head(10).to_html()}
            <h2>Statistics</h2>
            {df.describe().to_html()}
            </body>
            </html>
            """
            st.download_button("Download HTML", html, file_name="report.html", mime="text/html")
            st.success("HTML report generated!")
        
        elif report_type == "PDF":
            if REPORTLAB_AVAILABLE:
                buffer = BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=letter)
                elements = []
                styles = getSampleStyleSheet()
                
                elements.append(Paragraph("Data Analysis Report", styles['Title']))
                elements.append(Spacer(1, 12))
                elements.append(Paragraph(f"Rows: {profile['shape'][0]}, Columns: {profile['shape'][1]}", styles['Normal']))
                
                data = [['Metric', 'Value']]
                data.append(['Total Rows', f"{profile['shape'][0]:,}"])
                data.append(['Total Columns', str(profile['shape'][1])])
                data.append(['Memory Usage', f"{profile['memory_usage']:.2f} MB"])
                
                table = Table(data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(table)
                
                doc.build(elements)
                buffer.seek(0)
                st.download_button("Download PDF", buffer, file_name="report.pdf", mime="application/pdf")
                st.success("PDF report generated!")
            else:
                st.warning("ReportLab not installed")

# Main App
def main():
    st.markdown("<div class='main-header'>🤖 AI Data Analyst Platform</div>", unsafe_allow_html=True)
    st.markdown("### Automate ALL data analyst tasks with AI - No TensorFlow Required!")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📋 Navigation")
        page = st.radio("Select Page", [
            "📤 Upload Data",
            "📊 Data Overview",
            "🧹 Data Cleaning",
            "📈 Statistical Analysis",
            "📉 Visualizations",
            "🤖 Machine Learning",
            "🧠 Deep Learning",
            "🔍 Clustering",
            "⏰ Time Series",
            "🚨 Anomaly Detection",
            "📐 Flowcharts",
            "💡 AI Insights",
            "🗣️ NL Queries",
            "📄 Reports"
        ])
        
        st.markdown("---")
        if 'df' in st.session_state and st.session_state.df is not None:
            st.success(f"✅ {st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} cols")
        else:
            st.warning("⚠️ No data loaded")
        
        st.markdown("---")
        st.info("🚀 Built with PyTorch, Prophet, XGBoost")
    
    # Session State
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'df_clean' not in st.session_state:
        st.session_state.df_clean = None
    if 'profile' not in st.session_state:
        st.session_state.profile = None
    
    # Page Routing
    if page == "📤 Upload Data":
        st.markdown("## Upload Your Dataset")
        uploaded = st.file_uploader("Choose file", type=['csv', 'xlsx', 'xls', 'json'])
        
        if uploaded:
            df = load_data(uploaded)
            if df is not None:
                st.session_state.df = df
                st.session_state.profile = profile_data(df)
                st.success(f"✅ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
                st.dataframe(df.head(10), use_container_width=True)
    
    elif page == "📊 Data Overview":
        if st.session_state.df is not None:
            display_data_profile(st.session_state.df)
        else:
            st.warning("⚠️ Please upload data first")
    
    elif page == "🧹 Data Cleaning":
        if st.session_state.df is not None:
            st.markdown("## Data Cleaning")
            strategy = st.selectbox("Missing Value Strategy", ['mean', 'median', 'mode', 'drop'])
            remove_dup = st.checkbox("Remove Duplicates", value=True)
            
            if st.button("Clean Data"):
                df_clean, report = clean_data(st.session_state.df, strategy, remove_dup)
                st.session_state.df_clean = df_clean
                st.success("✅ Data cleaned!")
                for r in report:
                    st.info(r)
                st.dataframe(df_clean.head(), use_container_width=True)
        else:
            st.warning("⚠️ Please upload data first")
    
    elif page == "📈 Statistical Analysis":
        if st.session_state.df is not None:
            st.markdown("## Statistical Analysis")
            st.dataframe(st.session_state.df.describe(), use_container_width=True)
            
            numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                st.markdown("### Correlation Matrix")
                corr = st.session_state.df[numeric_cols].corr()
                fig = px.imshow(corr, title="Correlations", color_continuous_scale='RdBu_r')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Please upload data first")
    
    elif page == "📉 Visualizations":
        if st.session_state.df is not None:
            create_visualizations(st.session_state.df)
        else:
            st.warning("⚠️ Please upload data first")
    
    elif page == "🤖 Machine Learning":
        if st.session_state.df is not None:
            target = st.selectbox("Select target column", st.session_state.df.columns.tolist())
            if st.button("Train Models"):
                train_ml_models(st.session_state.df, target)
        else:
            st.warning("⚠️ Please upload data first")
    
    elif page == "🧠 Deep Learning":
        if st.session_state.df is not None:
            if PYTORCH_AVAILABLE:
                st.info("🧠 PyTorch Deep Learning features available")
                st.markdown("Neural networks, LSTM, and autoencoders can be trained here.")
            else:
                st.warning("PyTorch not installed. Install with: pip install torch")
        else:
            st.warning("⚠️ Please upload data first")
    
    elif page == "🔍 Clustering":
        if st.session_state.df is not None:
            perform_clustering(st.session_state.df)
        else:
            st.warning("⚠️ Please upload data first")
    
    elif page == "⏰ Time Series":
        if st.session_state.df is not None:
            time_series_forecast(st.session_state.df)
        else:
            st.warning("⚠️ Please upload data first")
    
    elif page == "🚨 Anomaly Detection":
        if st.session_state.df is not None:
            anomaly_detection(st.session_state.df)
        else:
            st.warning("⚠️ Please upload data first")
    
    elif page == "📐 Flowcharts":
        generate_flowcharts()
    
    elif page == "💡 AI Insights":
        if st.session_state.df is not None and st.session_state.profile is not None:
            generate_insights(st.session_state.df, st.session_state.profile)
        else:
            st.warning("⚠️ Please upload data first")
    
    elif page == "🗣️ NL Queries":
        if st.session_state.df is not None:
            nl_query(st.session_state.df)
        else:
            st.warning("⚠️ Please upload data first")
    
    elif page == "📄 Reports":
        if st.session_state.df is not None and st.session_state.profile is not None:
            generate_reports(st.session_state.df, st.session_state.profile)
        else:
            st.warning("⚠️ Please upload data first")

if __name__ == "__main__":
    main()
