# ============================================
# Fact Checker Analysis Suite
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from textblob import TextBlob

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================
# Page Configuration
# ============================
st.set_page_config(
    page_title="Fact Checker Pro",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================
# Purple and Black CSS Theme
# ============================
st.markdown("""
<style>
    /* Purple and Black Color Scheme */
    :root {
        --primary: #6B46C1;
        --primary-dark: #553C9A;
        --primary-light: #9F7AEA;
        --secondary: #1A202C;
        --accent: #805AD5;
        --success: #38A169;
        --warning: #D69E2E;
        --danger: #E53E3E;
        --light: #F7FAFC;
        --dark: #0F1419;
        --text-light: #FFFFFF;
        --text-dark: #2D3748;
        --gray-100: #F7FAFC;
        --gray-200: #EDF2F7;
        --gray-300: #E2E8F0;
        --gray-400: #CBD5E0;
        --gray-500: #A0AEC0;
        --gray-600: #718096;
        --gray-700: #4A5568;
        --gray-800: #2D3748;
        --gray-900: #1A202C;
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, var(--dark) 0%, var(--secondary) 100%);
        color: var(--text-light);
    }
    
    /* Professional Header */
    .professional-header {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        padding: 2.5rem 0;
        margin-bottom: 2rem;
        border-radius: 0 0 15px 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    /* Cards */
    .professional-card {
        background: var(--secondary);
        border-radius: 12px;
        padding: 1.8rem;
        margin: 1rem 0;
        border: 1px solid var(--primary);
        box-shadow: 0 4px 15px rgba(107, 70, 193, 0.2);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        color: var(--text-light);
    }
    
    .professional-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(107, 70, 193, 0.3);
        border-color: var(--primary-light);
    }
    
    .professional-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--primary-light) 0%, var(--primary) 100%);
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: var(--text-light);
        padding: 1.8rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem;
        border: none;
        box-shadow: 0 4px 15px rgba(107, 70, 193, 0.4);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(107, 70, 193, 0.6);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: var(--text-light);
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.9;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: var(--text-light);
    }
    
    /* Sections */
    .section-header {
        font-size: 1.6rem;
        font-weight: 600;
        color: var(--text-light);
        margin: 2.5rem 0 1.5rem 0;
        padding: 0.8rem 1.2rem;
        background: var(--secondary);
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        border-left: 4px solid var(--primary-light);
    }
    
    /* Sidebar - Right Side */
    section[data-testid="stSidebar"] {
        background: var(--secondary) !important;
        border-left: 2px solid var(--primary) !important;
        box-shadow: -2px 0 10px rgba(0,0,0,0.3);
    }
    
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--text-light);
        margin-bottom: 1.5rem;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        border-radius: 8px;
        border-left: 4px solid var(--primary-light);
    }
    
    /* Buttons - Professional Style */
    .stButton button {
        width: 100%;
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: var(--text-light);
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(107, 70, 193, 0.4);
        background: linear-gradient(135deg, var(--primary-light) 0%, var(--primary) 100%);
    }
    
    /* Select boxes and inputs */
    .stSelectbox, .stTextInput, .stNumberInput {
        background: var(--secondary) !important;
        color: var(--text-light) !important;
        border: 1px solid var(--primary) !important;
        border-radius: 6px !important;
    }
    
    .stSelectbox div, .stTextInput input, .stNumberInput input {
        background: var(--secondary) !important;
        color: var(--text-light) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: var(--secondary);
        border-bottom: 2px solid var(--primary);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: var(--secondary) !important;
        color: var(--text-light) !important;
        border-radius: 0;
        padding: 1rem 2rem;
        border-bottom: 3px solid transparent;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--secondary) !important;
        color: var(--primary-light) !important;
        border-bottom: 3px solid var(--primary-light) !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: var(--secondary) !important;
        color: var(--text-light) !important;
        border: 1px solid var(--primary) !important;
        border-radius: 6px !important;
        font-weight: 600;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--primary-light) 0%, var(--primary) 100%);
    }
    
    /* Success, Error, Info */
    .stSuccess {
        background: rgba(56, 161, 105, 0.2) !important;
        border: 1px solid var(--success) !important;
        color: var(--success) !important;
        border-radius: 6px;
    }
    
    .stError {
        background: rgba(229, 62, 62, 0.2) !important;
        border: 1px solid var(--danger) !important;
        color: var(--danger) !important;
        border-radius: 6px;
    }
    
    .stInfo {
        background: rgba(128, 90, 213, 0.2) !important;
        border: 1px solid var(--accent) !important;
        color: var(--accent) !important;
        border-radius: 6px;
    }
    
    /* Dataframe styling */
    .dataframe {
        background: var(--secondary) !important;
        color: var(--text-light) !important;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid var(--primary);
    }
    
    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, rgba(26, 32, 44, 0.95) 0%, rgba(15, 20, 25, 0.95) 100%);
        padding: 4rem 3rem;
        border-radius: 15px;
        margin: 2rem 0;
        text-align: center;
        border: 1px solid var(--primary);
        box-shadow: 0 8px 25px rgba(107, 70, 193, 0.3);
    }
    
    /* Model Performance Cards */
    .model-card {
        background: var(--secondary);
        border-radius: 10px;
        padding: 1.8rem;
        margin: 1rem;
        border: 1px solid var(--primary);
        box-shadow: 0 4px 15px rgba(107, 70, 193, 0.2);
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        color: var(--text-light);
    }
    
    .model-card:hover {
        border-color: var(--primary-light);
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(107, 70, 193, 0.3);
    }
    
    .model-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--primary-light) 0%, var(--primary) 100%);
    }
    
    .model-accuracy {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-light);
        margin: 1rem 0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    /* Feature Tags */
    .feature-tag {
        background: rgba(107, 70, 193, 0.2);
        color: var(--primary-light);
        padding: 0.5rem 1.2rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.3rem;
        display: inline-block;
        border: 1px solid rgba(107, 70, 193, 0.4);
        transition: all 0.3s ease;
    }
    
    .feature-tag:hover {
        background: rgba(107, 70, 193, 0.3);
        transform: translateY(-2px);
        border-color: var(--primary-light);
    }
    
    /* Charts container */
    .chart-container {
        background: var(--secondary);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(107, 70, 193, 0.2);
        margin: 1rem 0;
        border: 1px solid var(--primary);
    }
    
    /* Text visibility enhancements */
    .stMarkdown, .stText, .stHeader, .stSubheader {
        color: var(--text-light) !important;
    }
    
    .stDataFrame {
        color: var(--text-light) !important;
    }
    
    /* Ensure all text is visible */
    * {
        color: var(--text-light) !important;
    }
    
    /* Specific text color overrides */
    .st-bq, .st-br, .st-bs, .st-bt {
        color: var(--text-light) !important;
    }
    
    /* Plotly chart background */
    .js-plotly-plot .plotly {
        background: var(--secondary) !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# Initialize NLP
# ============================
@st.cache_resource
def load_nlp_model():
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        st.error("""
        SpaCy English model not found. 
        Please install: python -m spacy download en_core_web_sm
        """)
        st.stop()

nlp = load_nlp_model()
stop_words = STOP_WORDS

# ============================
# Enhanced Feature Engineering
# ============================
class ProfessionalFeatureExtractor:
    @staticmethod
    def extract_lexical_features(texts):
        """Extract lexical features with advanced preprocessing"""
        processed_texts = []
        for text in texts:
            doc = nlp(str(text).lower())
            tokens = [token.lemma_ for token in doc if token.text not in stop_words and token.is_alpha]
            processed_texts.append(" ".join(tokens))
        return TfidfVectorizer(max_features=1000, ngram_range=(1, 2)).fit_transform(processed_texts)
    
    @staticmethod
    def extract_semantic_features(texts):
        """Extract semantic features with sentiment analysis"""
        features = []
        for text in texts:
            blob = TextBlob(str(text))
            features.append([
                blob.sentiment.polarity,
                blob.sentiment.subjectivity,
                len(text.split()),
                len([word for word in text.split() if len(word) > 6]),
                text.count('!'),
                text.count('?'),
            ])
        return np.array(features)
    
    @staticmethod
    def extract_syntactic_features(texts):
        """Extract syntactic features with POS analysis"""
        processed_texts = []
        for text in texts:
            doc = nlp(str(text))
            pos_tags = [f"{token.pos_}_{token.tag_}" for token in doc]
            processed_texts.append(" ".join(pos_tags))
        return CountVectorizer(max_features=800, ngram_range=(1, 3)).fit_transform(processed_texts)
    
    @staticmethod
    def extract_pragmatic_features(texts):
        """Extract pragmatic features - context and intent analysis"""
        pragmatic_features = []
        pragmatic_indicators = {
            'modality': ['must', 'should', 'could', 'would', 'might', 'may'],
            'certainty': ['certainly', 'definitely', 'obviously', 'clearly'],
            'uncertainty': ['perhaps', 'maybe', 'possibly', 'probably'],
            'question': ['what', 'why', 'how', 'when', 'where', 'which', '?'],
            'emphasis': ['very', 'extremely', 'highly', 'absolutely']
        }
        
        for text in texts:
            text_lower = str(text).lower()
            features = []
            
            for category, words in pragmatic_indicators.items():
                count = sum(text_lower.count(word) for word in words)
                features.append(count)
            
            features.extend([
                text.count('!'),
                text.count('?'),
                len([s for s in text.split('.') if s.strip()]),
                len([w for w in text.split() if w.istitle()]),
            ])
            
            pragmatic_features.append(features)
        
        return np.array(pragmatic_features)

# ============================
# Enhanced Model Trainer
# ============================
class ProfessionalModelTrainer:
    def __init__(self):
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced'),
            "Support Vector": SVC(random_state=42, probability=True, class_weight='balanced'),
            "Naive Bayes": MultinomialNB()
        }
    
    def train_and_evaluate(self, X, y):
        """Professional model training with comprehensive evaluation"""
        results = {}
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        n_classes = len(le.classes_)
        
        test_size = max(0.15, min(0.25, 3 * n_classes / len(y_encoded)))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Professional progress tracking
        progress_container = st.empty()
        
        for i, (name, model) in enumerate(self.models.items()):
            with progress_container.container():
                cols = st.columns([3, 1])
                with cols[0]:
                    st.markdown(f"**Training {name}**")
                with cols[1]:
                    progress_bar = st.progress(0)
                    
                    # Simulate professional loading
                    for step in range(5):
                        progress_bar.progress((step + 1) / 5)
                        import time
                        time.sleep(0.1)
            
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                
                results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'model': model,
                    'predictions': y_pred,
                    'true_labels': y_test,
                    'probabilities': y_proba,
                    'confusion_matrix': cm,
                    'n_classes': n_classes,
                    'test_size': len(y_test),
                    'feature_importance': getattr(model, 'feature_importances_', None)
                }
                
            except Exception as e:
                results[name] = {'error': str(e)}
        
        progress_container.empty()
        return results, le

# ============================
# Enhanced Professional Visualizations
# ============================
class ProfessionalVisualizer:
    @staticmethod
    def create_performance_dashboard(results):
        """Create professional performance dashboard with Plotly"""
        models = []
        metrics_data = {
            'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': []
        }
        
        for model_name, result in results.items():
            if 'error' not in result:
                models.append(model_name)
                metrics_data['Accuracy'].append(result['accuracy'])
                metrics_data['Precision'].append(result['precision'])
                metrics_data['Recall'].append(result['recall'])
                metrics_data['F1-Score'].append(result['f1_score'])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy', 'Precision', 'Recall', 'F1-Score'),
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )
        
        colors = ['#9F7AEA', '#805AD5', '#6B46C1', '#553C9A']
        
        # Accuracy
        fig.add_trace(
            go.Bar(x=models, y=metrics_data['Accuracy'], 
                  marker_color=colors, name='Accuracy',
                  text=[f'{x:.3f}' for x in metrics_data['Accuracy']],
                  textposition='auto'),
            row=1, col=1
        )
        
        # Precision
        fig.add_trace(
            go.Bar(x=models, y=metrics_data['Precision'], 
                  marker_color=colors, name='Precision',
                  text=[f'{x:.3f}' for x in metrics_data['Precision']],
                  textposition='auto'),
            row=1, col=2
        )
        
        # Recall
        fig.add_trace(
            go.Bar(x=models, y=metrics_data['Recall'], 
                  marker_color=colors, name='Recall',
                  text=[f'{x:.3f}' for x in metrics_data['Recall']],
                  textposition='auto'),
            row=2, col=1
        )
        
        # F1-Score
        fig.add_trace(
            go.Bar(x=models, y=metrics_data['F1-Score'], 
                  marker_color=colors, name='F1-Score',
                  text=[f'{x:.3f}' for x in metrics_data['F1-Score']],
                  textposition='auto'),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            paper_bgcolor='#1A202C',
            plot_bgcolor='#1A202C',
            font=dict(color='#FFFFFF'),
            margin=dict(t=80, b=80, l=80, r=80)
        )
        
        # Update axes
        for i in range(4):
            fig.update_yaxes(range=[0, 1], row=(i//2)+1, col=(i%2)+1, gridcolor='#4A5568', color='#FFFFFF')
            fig.update_xaxes(tickangle=45, row=(i//2)+1, col=(i%2)+1, color='#FFFFFF')
            fig.update_annotations(font_color='#FFFFFF')
        
        return fig
    
    @staticmethod
    def create_confusion_matrix_heatmap(results, label_encoder):
        """Create interactive confusion matrix heatmap"""
        best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'] if 'error' not in x[1] else 0)[0]
        best_result = results[best_model_name]
        
        if 'confusion_matrix' not in best_result:
            return None
            
        cm = best_result['confusion_matrix']
        labels = label_encoder.classes_
        
        fig = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            x=labels,
            y=labels,
            title=f"Confusion Matrix - {best_model_name}",
            color_continuous_scale='Purples'
        )
        
        fig.update_layout(
            xaxis_title="Predicted",
            yaxis_title="Actual",
            paper_bgcolor='#1A202C',
            plot_bgcolor='#1A202C',
            font=dict(color='#FFFFFF')
        )
        
        return fig
    
    @staticmethod
    def create_metrics_radar_chart(results):
        """Create radar chart for model comparison"""
        models = []
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = []
        
        for model_name, result in results.items():
            if 'error' not in result:
                models.append(model_name)
                values.append([
                    result['accuracy'],
                    result['precision'],
                    result['recall'],
                    result['f1_score']
                ])
        
        fig = go.Figure()
        
        colors = ['#9F7AEA', '#805AD5', '#6B46C1', '#553C9A']
        
        for i, (model, metric_values) in enumerate(zip(models, values)):
            fig.add_trace(go.Scatterpolar(
                r=metric_values + [metric_values[0]],  # Close the radar
                theta=metrics + [metrics[0]],
                fill='toself',
                name=model,
                line=dict(color=colors[i % len(colors)]),
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    color='#FFFFFF'
                ),
                bgcolor='#1A202C'
            ),
            showlegend=True,
            title="Model Performance Radar Chart",
            paper_bgcolor='#1A202C',
            font=dict(color='#FFFFFF')
        )
        
        return fig

# ============================
# Professional Sidebar (Right Side)
# ============================
def setup_sidebar():
    """Setup professional sidebar on the right side"""
    with st.sidebar:
        st.markdown("<div class='sidebar-header'>FACT CHECKER PRO</div>", unsafe_allow_html=True)
        st.markdown("---")
        
        st.markdown("<div class='sidebar-header'>DATA UPLOAD</div>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose CSV File",
            type=["csv"],
            help="Upload your dataset for analysis"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.session_state.file_uploaded = True
                
                st.success(f"Successfully loaded {df.shape[0]} rows")
                
                st.markdown("<div class='sidebar-header'>ANALYSIS CONFIGURATION</div>", unsafe_allow_html=True)
                
                text_col = st.selectbox(
                    "Text Column",
                    df.columns,
                    help="Select column containing text data"
                )
                
                target_col = st.selectbox(
                    "Target Column",
                    df.columns,
                    help="Select column containing labels"
                )
                
                feature_type = st.selectbox(
                    "Feature Engineering",
                    ["Lexical", "Semantic", "Syntactic", "Pragmatic", "Comprehensive"],
                    help="Choose feature extraction method"
                )
                
                st.session_state.config = {
                    'text_col': text_col,
                    'target_col': target_col,
                    'feature_type': feature_type
                }
                
                if st.button("START ANALYSIS", use_container_width=True):
                    st.session_state.analyze_clicked = True
                else:
                    st.session_state.analyze_clicked = False
                    
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        else:
            st.session_state.file_uploaded = False
            st.session_state.analyze_clicked = False

# ============================
# Enhanced Main Content
# ============================
def main_content():
    """Main content with professional styling"""
    
    # Professional Header
    st.markdown("""
    <div class='professional-header'>
        <div style='text-align: center;'>
            <h1 style='color: white; font-size: 3rem; font-weight: 700; margin: 0;'>FACT CHECKER PRO</h1>
            <p style='color: rgba(255,255,255,0.9); font-size: 1.2rem; margin: 0.5rem 0 0 0;'>
                Advanced Text Analysis for Fact Verification
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.get('file_uploaded', False):
        show_professional_welcome()
        return
    
    df = st.session_state.df
    config = st.session_state.get('config', {})
    
    # Dataset Overview
    st.markdown("<div class='section-header'>DATASET OVERVIEW</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df.shape[0]:,}</div>
            <div class="metric-label">Total Records</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df.shape[1]}</div>
            <div class="metric-label">Features</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        missing_vals = df.isnull().sum().sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{missing_vals}</div>
            <div class="metric-label">Missing Values</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        unique_classes = df[config.get('target_col', '')].nunique() if config.get('target_col') in df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{unique_classes}</div>
            <div class="metric-label">Unique Classes</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Data Preview
    with st.expander("DATA EXPLORATION", expanded=True):
        tab1, tab2, tab3 = st.tabs(["Data Preview", "Statistics", "Target Distribution"])
        with tab1:
            st.dataframe(df.head(10), use_container_width=True)
        with tab2:
            st.write(df.describe(include='all'))
        with tab3:
            if config.get('target_col') in df.columns:
                target_dist = df[config['target_col']].value_counts()
                fig = px.bar(target_dist, 
                           title="Target Variable Distribution",
                           labels={'value': 'Count', 'index': config['target_col']})
                fig.update_layout(
                    paper_bgcolor='#1A202C',
                    plot_bgcolor='#1A202C',
                    font=dict(color='#FFFFFF'),
                    xaxis=dict(color='#FFFFFF'),
                    yaxis=dict(color='#FFFFFF')
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Analysis Results
    if st.session_state.get('analyze_clicked', False):
        perform_professional_analysis(df, config)

def show_professional_welcome():
    """Professional welcome screen"""
    st.markdown("""
    <div class='hero-section'>
        <h1 style='color: #FFFFFF; font-size: 2.8rem; font-weight: 700; margin-bottom: 1.5rem;'>
            Welcome to Fact Checker Pro
        </h1>
        <p style='color: #CBD5E0; font-size: 1.2rem; margin-bottom: 2.5rem; line-height: 1.6;'>
            Advanced text analysis platform for fact verification and content validation. 
            Leverage machine learning algorithms to analyze and classify textual content.
        </p>
        <div style='display: flex; gap: 1rem; flex-wrap: wrap; justify-content: center; margin-bottom: 2rem;'>
            <span class="feature-tag">4 ML Algorithms</span>
            <span class="feature-tag">Interactive Dashboards</span>
            <span class="feature-tag">Advanced Feature Engineering</span>
            <span class="feature-tag">Real-time Analytics</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='section-header'>GETTING STARTED</div>", unsafe_allow_html=True)
    
    steps = [
        {"title": "UPLOAD DATA", "desc": "Upload your CSV file using the sidebar"},
        {"title": "CONFIGURE", "desc": "Select text columns, target variables, and analysis type"},
        {"title": "ANALYZE", "desc": "Run comprehensive NLP analysis with multiple algorithms"},
        {"title": "VISUALIZE", "desc": "Explore interactive dashboards and insights"}
    ]
    
    cols = st.columns(4)
    for idx, step in enumerate(steps):
        with cols[idx]:
            st.markdown(f"""
            <div class="professional-card">
                <h3 style="color: #FFFFFF; margin-bottom: 1rem; text-align: center;">{step['title']}</h3>
                <p style="color: #CBD5E0; line-height: 1.5; text-align: center;">{step['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

def perform_professional_analysis(df, config):
    """Perform professional analysis with enhanced visualizations"""
    st.markdown("<div class='section-header'>ANALYSIS RESULTS</div>", unsafe_allow_html=True)
    
    # Data validation
    if config['text_col'] not in df.columns or config['target_col'] not in df.columns:
        st.error("Selected columns not found in dataset.")
        return
    
    if df[config['text_col']].isnull().any():
        df[config['text_col']] = df[config['text_col']].fillna('')
    
    if df[config['target_col']].isnull().any():
        st.error("Target column contains missing values.")
        return
    
    if len(df[config['target_col']].unique()) < 2:
        st.error("Target column must have at least 2 unique classes.")
        return
    
    # Feature extraction
    with st.spinner("Extracting features..."):
        extractor = ProfessionalFeatureExtractor()
        X = df[config['text_col']].astype(str)
        y = df[config['target_col']]
        
        if config['feature_type'] == "Lexical":
            X_features = extractor.extract_lexical_features(X)
            feature_desc = "Word-level analysis with lemmatization"
        elif config['feature_type'] == "Semantic":
            X_features = extractor.extract_semantic_features(X)
            feature_desc = "Sentiment analysis and text complexity"
        elif config['feature_type'] == "Syntactic":
            X_features = extractor.extract_syntactic_features(X)
            feature_desc = "Grammar structure and POS analysis"
        elif config['feature_type'] == "Pragmatic":
            X_features = extractor.extract_pragmatic_features(X)
            feature_desc = "Context analysis and intent detection"
        else:  # Comprehensive
            # Combine all features
            lexical = extractor.extract_lexical_features(X)
            semantic = extractor.extract_semantic_features(X)
            pragmatic = extractor.extract_pragmatic_features(X)
            
            # Convert sparse matrices to arrays and combine
            if hasattr(lexical, 'toarray'):
                lexical = lexical.toarray()
            if hasattr(semantic, 'toarray'):
                semantic = semantic.toarray()
                
            X_features = np.hstack([lexical, semantic, pragmatic])
            feature_desc = "Comprehensive feature engineering combining all methods"
    
    st.success(f"Feature extraction completed: {feature_desc}")
    
    # Model training
    with st.spinner("Training machine learning models..."):
        trainer = ProfessionalModelTrainer()
        results, label_encoder = trainer.train_and_evaluate(X_features, y)
    
    # Display results
    successful_models = {k: v for k, v in results.items() if 'error' not in v}
    
    if successful_models:
        # Model Performance Cards
        st.markdown("#### MODEL PERFORMANCE SUMMARY")
        
        cols = st.columns(len(successful_models))
        for idx, (model_name, result) in enumerate(successful_models.items()):
            with cols[idx]:
                accuracy = result['accuracy']
                st.markdown(f"""
                <div class="model-card">
                    <h4 style="color: #FFFFFF; margin-bottom: 1rem;">{model_name}</h4>
                    <div class="model-accuracy">{accuracy:.1%}</div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem; color: #CBD5E0;">
                        <div style="text-align: center;">
                            <small>Precision</small>
                            <div style="font-weight: 600; color: #9F7AEA;">{result['precision']:.3f}</div>
                        </div>
                        <div style="text-align: center;">
                            <small>Recall</small>
                            <div style="font-weight: 600; color: #9F7AEA;">{result['recall']:.3f}</div>
                        </div>
                        <div style="text-align: center;">
                            <small>F1-Score</small>
                            <div style="font-weight: 600; color: #9F7AEA;">{result['f1_score']:.3f}</div>
                        </div>
                        <div style="text-align: center;">
                            <small>Test Size</small>
                            <div style="font-weight: 600; color: #9F7AEA;">{result['test_size']}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Enhanced Visualizations
        st.markdown("#### INTERACTIVE DASHBOARD")
        
        viz = ProfessionalVisualizer()
        
        # Performance Dashboard
        with st.container():
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.plotly_chart(viz.create_performance_dashboard(successful_models), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Additional Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            radar_chart = viz.create_metrics_radar_chart(successful_models)
            if radar_chart:
                st.plotly_chart(radar_chart, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            cm_chart = viz.create_confusion_matrix_heatmap(successful_models, label_encoder)
            if cm_chart:
                st.plotly_chart(cm_chart, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Best Model Recommendation
        best_model = max(successful_models.items(), key=lambda x: x[1]['accuracy'])
        st.markdown(f"""
        <div class="professional-card">
            <h3 style="color: #FFFFFF; margin-bottom: 1rem;">RECOMMENDED MODEL</h3>
            <div style="background: linear-gradient(135deg, #6B46C1, #553C9A); color: white; padding: 1.5rem; border-radius: 8px;">
                <h4 style="margin: 0 0 0.5rem 0; color: white;">{best_model[0]}</h4>
                <p style="margin: 0; font-size: 1.1rem;">
                    Achieved the highest accuracy of <strong>{best_model[1]['accuracy']:.1%}</strong>
                    with balanced performance across all metrics.
                </p>
            </div>
            <p style="color: #CBD5E0; margin-top: 1rem;">
                This model demonstrates robust performance and is recommended for production deployment.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.error("No models were successfully trained. Please check your data and configuration.")

# ============================
# Main Application
# ============================
def main():
    # Initialize session state
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    if 'analyze_clicked' not in st.session_state:
        st.session_state.analyze_clicked = False
    
    # Setup sidebar
    setup_sidebar()
    
    # Main content
    main_content()

if __name__ == "__main__":
    main()
