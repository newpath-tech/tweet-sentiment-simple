"""
TWEET SENTIMENT ANALYTICS DASHBOARD
Complete Phases 1-5: Basic App → TextBlob → VADER → pandas/numpy → Visualizations
Deployment-ready for Streamlit Cloud
FIXED: Data type errors and navigation issues
"""

import streamlit as st
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # For Streamlit compatibility
import time
from datetime import datetime
import numpy as np
import warnings
import json
import io
import sys
import traceback
from typing import Dict, Tuple, Any

# ============================================
# CONFIGURATION & SETUP
# ============================================
warnings.filterwarnings('ignore')

# Set page configuration - PHASE 1
st.set_page_config(
    page_title="Tweet Sentiment Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/newpath-tech/tweet-sentiment-simple',
        'Report a bug': 'https://github.com/newpath-tech/tweet-sentiment-simple/issues',
        'About': "# Tweet Sentiment Analytics Dashboard\n\nComplete sentiment analysis tool for tweets."
    }
)

# ============================================
# CUSTOM STYLING - Clean UI
# ============================================
st.markdown("""
<style>
    /* Main headers */
    .main-title {
        font-size: 2.8rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
    }
    
    .section-header {
        font-size: 1.8rem;
        color: #4F46E5;
        border-bottom: 3px solid #4F46E5;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
        font-weight: 700;
    }
    
    /* Cards and metrics */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
        border: 1px solid #E5E7EB;
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    
    .positive { 
        color: #10B981; 
        background: linear-gradient(135deg, #10B98122 0%, #10B98111 100%);
        padding: 4px 12px;
        border-radius: 20px;
        display: inline-block;
        font-weight: 600;
    }
    
    .negative { 
        color: #EF4444; 
        background: linear-gradient(135deg, #EF444422 0%, #EF444411 100%);
        padding: 4px 12px;
        border-radius: 20px;
        display: inline-block;
        font-weight: 600;
    }
    
    .neutral { 
        color: #3B82F6; 
        background: linear-gradient(135deg, #3B82F622 0%, #3B82F611 100%);
        padding: 4px 12px;
        border-radius: 20px;
        display: inline-block;
        font-weight: 600;
    }
    
    /* Buttons */
    .stButton button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.3s ease !important;
        border: 2px solid transparent !important;
    }
    
    .stButton button:hover {
        transform: scale(1.05) !important;
    }
    
    /* Dataframes */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 100%);
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #4F46E5;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem;
        }
        .metric-value {
            font-size: 1.8rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# INITIALIZATION FUNCTIONS
# ============================================

@st.cache_resource
def initialize_nltk() -> Tuple[Any, bool]:
    """
    Initialize NLTK and VADER sentiment analyzer with robust error handling
    PHASE 3: NLTK/VADER Integration
    """
    try:
        # Create progress indicator
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        status_text.text("🔄 Initializing NLTK...")
        progress_bar.progress(25)
        
        # Download required NLTK data
        nltk.download('vader_lexicon', quiet=True)
        progress_bar.progress(50)
        
        nltk.download('punkt', quiet=True)
        progress_bar.progress(75)
        
        # Initialize VADER sentiment analyzer
        sia = SentimentIntensityAnalyzer()
        
        # Test initialization
        test_result = sia.polarity_scores("I love this!")
        
        progress_bar.progress(100)
        status_text.text("✅ NLTK initialized successfully!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        return sia, True
        
    except Exception as e:
        st.sidebar.error(f"""
        ⚠️ **NLTK Initialization Error**
        
        Error: {str(e)}
        
        **Troubleshooting steps:**
        1. Check your internet connection
        2. Try running the app again
        3. TextBlob analysis will still work
        4. Some features will be limited
        
        Full error details have been logged.
        """)
        
        # Log detailed error
        st.sidebar.code(f"Error details: {traceback.format_exc()}")
        
        return None, False

# Initialize NLTK/VADER
sia, nltk_available = initialize_nltk()

# ============================================
# SESSION STATE MANAGEMENT - FIXED
# ============================================

def initialize_session_state():
    """Initialize all session state variables"""
    # Initialize analysis history
    if 'analysis_history' not in st.session_state:
        # PHASE 4: pandas DataFrame structure
        st.session_state.analysis_history = pd.DataFrame(columns=[
            'id', 'timestamp', 'tweet', 'tweet_short', 
            'textblob_score', 'textblob_sentiment', 'textblob_subjectivity',
            'vader_score', 'vader_sentiment', 'vader_positive', 'vader_negative', 'vader_neutral',
            'word_count', 'char_count', 'analysis_time'
        ])
    
    # Initialize analysis count
    if 'analysis_count' not in st.session_state:
        st.session_state.analysis_count = 0
    
    # Initialize current view - FIXED
    if 'current_view' not in st.session_state:
        st.session_state.current_view = "Live Analysis"
    
    # Initialize chart theme
    if 'chart_theme' not in st.session_state:
        st.session_state.chart_theme = "default"
    
    # Initialize example text - FIXED
    if 'example_text' not in st.session_state:
        st.session_state.example_text = ""
    
    # Initialize chart style
    if 'chart_style' not in st.session_state:
        st.session_state.chart_style = "Bar Chart"

initialize_session_state()

# ============================================
# UTILITY FUNCTIONS
# ============================================

def analyze_sentiment_textblob(text: str) -> Dict[str, Any]:
    """
    Analyze sentiment using TextBlob
    PHASE 2: TextBlob Integration
    """
    try:
        blob = TextBlob(text)
        score = float(blob.sentiment.polarity)
        subjectivity = float(blob.sentiment.subjectivity)
        
        # Determine sentiment with thresholds
        if score > 0.2:
            sentiment = 'positive'
            emoji = '😊'
            intensity = 'strong positive' if score > 0.5 else 'positive'
        elif score > 0.1:
            sentiment = 'positive'
            emoji = '🙂'
            intensity = 'mild positive'
        elif score < -0.2:
            sentiment = 'negative'
            emoji = '😠'
            intensity = 'strong negative' if score < -0.5 else 'negative'
        elif score < -0.1:
            sentiment = 'negative'
            emoji = '😕'
            intensity = 'mild negative'
        else:
            sentiment = 'neutral'
            emoji = '😐'
            intensity = 'neutral'
        
        return {
            'score': score,
            'sentiment': sentiment,
            'subjectivity': subjectivity,
            'emoji': emoji,
            'intensity': intensity,
            'confidence': 1 - subjectivity  # Higher objectivity = higher confidence
        }
        
    except Exception as e:
        st.error(f"TextBlob analysis error: {str(e)}")
        return {
            'score': 0.0,
            'sentiment': 'error',
            'subjectivity': 0.0,
            'emoji': '❌',
            'intensity': 'error',
            'confidence': 0.0
        }

def analyze_sentiment_vader(text: str) -> Dict[str, Any]:
    """
    Analyze sentiment using VADER
    PHASE 3: VADER Integration
    """
    if not nltk_available or sia is None:
        return {
            'score': 0.0,
            'sentiment': 'N/A',
            'positive': 0.0,
            'negative': 0.0,
            'neutral': 0.0,
            'emoji': '⚠️',
            'intensity': 'N/A',
            'confidence': 0.0
        }
    
    try:
        scores = sia.polarity_scores(text)
        compound = float(scores['compound'])
        
        # Determine sentiment with VADER thresholds
        if compound >= 0.5:
            sentiment = 'positive'
            emoji = '😊'
            intensity = 'very positive'
        elif compound >= 0.1:
            sentiment = 'positive'
            emoji = '🙂'
            intensity = 'positive'
        elif compound <= -0.5:
            sentiment = 'negative'
            emoji = '😠'
            intensity = 'very negative'
        elif compound <= -0.1:
            sentiment = 'negative'
            emoji = '😕'
            intensity = 'negative'
        else:
            sentiment = 'neutral'
            emoji = '😐'
            intensity = 'neutral'
        
        # Calculate confidence based on score extremity
        confidence = min(abs(compound) * 2, 1.0)  # Normalize to 0-1
        
        return {
            'score': compound,
            'sentiment': sentiment,
            'positive': float(scores['pos']),
            'negative': float(scores['neg']),
            'neutral': float(scores['neu']),
            'emoji': emoji,
            'intensity': intensity,
            'confidence': confidence
        }
        
    except Exception as e:
        st.error(f"VADER analysis error: {str(e)}")
        return {
            'score': 0.0,
            'sentiment': 'error',
            'positive': 0.0,
            'negative': 0.0,
            'neutral': 0.0,
            'emoji': '❌',
            'intensity': 'error',
            'confidence': 0.0
        }

def calculate_text_stats(text: str) -> Dict[str, int]:
    """Calculate text statistics"""
    words = text.split()
    chars = len(text)
    sentences = len([s for s in text.split('.') if s.strip()])
    
    return {
        'word_count': len(words),
        'char_count': chars,
        'sentence_count': sentences,
        'avg_word_length': chars / len(words) if words else 0
    }

def get_sentiment_style(sentiment: str) -> str:
    """Get CSS class for sentiment"""
    styles = {
        'positive': 'positive',
        'negative': 'negative',
        'neutral': 'neutral',
        'error': 'neutral',
        'N/A': 'neutral'
    }
    return styles.get(sentiment, 'neutral')

# ============================================
# DATA ANALYSIS FUNCTIONS - PHASE 4
# ============================================

def convert_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric columns to proper numeric types"""
    df = df.copy()
    
    # Define numeric columns
    numeric_cols = [
        'textblob_score', 'textblob_subjectivity',
        'vader_score', 'vader_positive', 'vader_negative', 'vader_neutral',
        'word_count', 'char_count', 'analysis_time'
    ]
    
    # Convert each numeric column
    for col in numeric_cols:
        if col in df.columns:
            # Try to convert to numeric, coerce errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Ensure ID is integer
    if 'id' in df.columns:
        df['id'] = pd.to_numeric(df['id'], errors='coerce').fillna(0).astype(int)
    
    return df

def calculate_advanced_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate advanced analytics using pandas and numpy
    PHASE 4: pandas/numpy Integration
    """
    if df.empty:
        return {}
    
    # Convert to numeric first
    df_numeric = convert_to_numeric(df)
    
    metrics = {}
    
    # Basic counts
    metrics['total_analyses'] = len(df_numeric)
    
    # TextBlob metrics
    if 'textblob_score' in df_numeric.columns:
        # Remove NaN values
        scores = df_numeric['textblob_score'].dropna().astype(float).values
        
        if len(scores) > 0:
            metrics.update({
                'tb_mean': float(np.mean(scores)),
                'tb_median': float(np.median(scores)),
                'tb_std': float(np.std(scores)),
                'tb_min': float(np.min(scores)),
                'tb_max': float(np.max(scores)),
                'tb_range': float(np.ptp(scores)),
                'tb_q25': float(np.percentile(scores, 25)),
                'tb_q75': float(np.percentile(scores, 75)),
                'tb_iqr': float(np.percentile(scores, 75) - np.percentile(scores, 25)),
                'tb_skew': float(pd.Series(scores).skew()),
            })
        else:
            metrics.update({
                'tb_mean': 0.0,
                'tb_median': 0.0,
                'tb_std': 0.0,
                'tb_min': 0.0,
                'tb_max': 0.0,
                'tb_range': 0.0,
                'tb_q25': 0.0,
                'tb_q75': 0.0,
                'tb_iqr': 0.0,
                'tb_skew': 0.0,
            })
    
    # VADER metrics
    if 'vader_score' in df_numeric.columns:
        vader_scores = df_numeric['vader_score'].dropna().astype(float).values
        
        if len(vader_scores) > 0:
            metrics.update({
                'vader_mean': float(np.mean(vader_scores)),
                'vader_std': float(np.std(vader_scores)),
                'vader_min': float(np.min(vader_scores)),
                'vader_max': float(np.max(vader_scores)),
            })
        else:
            metrics.update({
                'vader_mean': 0.0,
                'vader_std': 0.0,
                'vader_min': 0.0,
                'vader_max': 0.0,
            })
    
    # Sentiment distribution
    if 'textblob_sentiment' in df.columns:
        sentiment_counts = df['textblob_sentiment'].value_counts()
        for sentiment in ['positive', 'negative', 'neutral']:
            count = sentiment_counts.get(sentiment, 0)
            metrics[f'tb_{sentiment}_count'] = int(count)
            metrics[f'tb_{sentiment}_percent'] = float((count / len(df)) * 100) if len(df) > 0 else 0.0
    
    # Text statistics
    if 'word_count' in df_numeric.columns:
        word_counts = df_numeric['word_count'].dropna().astype(float).values
        
        if len(word_counts) > 0:
            metrics.update({
                'avg_word_count': float(np.mean(word_counts)),
                'total_words': int(np.sum(word_counts)),
                'max_words': int(np.max(word_counts)),
                'min_words': int(np.min(word_counts)),
            })
        else:
            metrics.update({
                'avg_word_count': 0.0,
                'total_words': 0,
                'max_words': 0,
                'min_words': 0,
            })
    
    # Correlation between TextBlob and VADER
    if 'textblob_score' in df_numeric.columns and 'vader_score' in df_numeric.columns:
        tb_scores = df_numeric['textblob_score'].dropna().astype(float).values
        vader_scores = df_numeric['vader_score'].dropna().astype(float).values
        
        if len(tb_scores) > 1 and len(vader_scores) > 1:
            # Ensure same length
            min_len = min(len(tb_scores), len(vader_scores))
            if min_len > 1:
                correlation = np.corrcoef(tb_scores[:min_len], vader_scores[:min_len])[0, 1]
                metrics['tb_vader_correlation'] = float(correlation)
    
    # Agreement rate
    if 'textblob_sentiment' in df.columns and 'vader_sentiment' in df.columns:
        # Filter out N/A values
        valid_mask = (df['textblob_sentiment'] != 'N/A') & (df['vader_sentiment'] != 'N/A')
        if valid_mask.any():
            valid_df = df[valid_mask]
            agreement = (valid_df['textblob_sentiment'] == valid_df['vader_sentiment']).mean()
            metrics['agreement_rate'] = float(agreement * 100)
    
    return metrics

def generate_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Generate correlation matrix for numeric columns"""
    # Convert to numeric first
    df_numeric = convert_to_numeric(df)
    
    numeric_cols = ['textblob_score', 'vader_score', 'word_count', 'textblob_subjectivity']
    available_cols = [col for col in numeric_cols if col in df_numeric.columns]
    
    if len(available_cols) >= 2:
        # Filter out NaN values
        clean_df = df_numeric[available_cols].dropna()
        if len(clean_df) >= 2:
            return clean_df.corr()
    return pd.DataFrame()

# ============================================
# VISUALIZATION FUNCTIONS - PHASE 5 (FIXED)
# ============================================

def create_sentiment_distribution_chart(df: pd.DataFrame, theme: str = "default") -> plt.Figure:
    """Create sentiment distribution bar chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set style based on theme
    if theme == "dark":
        plt.style.use('dark_background')
        bar_colors = ['#10B981', '#EF4444', '#3B82F6']
        bg_color = '#1F2937'
        text_color = 'white'
    elif theme == "light":
        plt.style.use('default')
        bar_colors = ['#34D399', '#F87171', '#60A5FA']
        bg_color = 'white'
        text_color = 'black'
    else:  # vibrant/default
        bar_colors = ['#10B981', '#EF4444', '#3B82F6']
        bg_color = '#F9FAFB'
        text_color = 'black'
    
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    if 'textblob_sentiment' in df.columns:
        sentiment_counts = df['textblob_sentiment'].value_counts()
        sentiments = sentiment_counts.index.tolist()
        counts = sentiment_counts.values.tolist()
        
        # Create bars
        bars = ax.bar(sentiments, counts, color=bar_colors[:len(sentiments)], 
                     edgecolor='white', linewidth=2, alpha=0.9)
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                   str(count), ha='center', va='bottom', 
                   fontweight='bold', fontsize=12, color=text_color)
        
        # Add percentage labels
        total = sum(counts)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            percentage = (count / total) * 100 if total > 0 else 0
            ax.text(bar.get_x() + bar.get_width()/2, height/2,
                   f'{percentage:.1f}%', ha='center', va='center',
                   fontweight='bold', fontsize=14, color='white')
    
    ax.set_title('Sentiment Distribution', fontsize=16, fontweight='bold', color=text_color, pad=20)
    ax.set_xlabel('Sentiment', fontsize=12, fontweight='bold', color=text_color)
    ax.set_ylabel('Count', fontsize=12, fontweight='bold', color=text_color)
    ax.grid(axis='y', alpha=0.3, color=text_color)
    ax.tick_params(colors=text_color)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig

def create_score_trend_chart(df: pd.DataFrame, theme: str = "default") -> plt.Figure:
    """Create sentiment score trend chart - FIXED"""
    if len(df) < 2:
        return None
    
    # Convert to numeric
    df_numeric = convert_to_numeric(df)
    
    # Check if we have valid numeric data
    if 'textblob_score' not in df_numeric.columns or df_numeric['textblob_score'].isna().all():
        return None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set style
    if theme == "dark":
        plt.style.use('dark_background')
        line_color_tb = '#3B82F6'
        line_color_vader = '#10B981'
        bg_color = '#1F2937'
        text_color = 'white'
    else:
        plt.style.use('default')
        line_color_tb = '#2563EB'
        line_color_vader = '#059669'
        bg_color = 'white'
        text_color = 'black'
    
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    # Ensure id is numeric and sort
    df_numeric = df_numeric.sort_values('id')
    
    # Plot TextBlob scores
    valid_tb_mask = df_numeric['textblob_score'].notna()
    if valid_tb_mask.any():
        ax.plot(df_numeric.loc[valid_tb_mask, 'id'], 
                df_numeric.loc[valid_tb_mask, 'textblob_score'], 
                label='TextBlob Score', marker='o', linewidth=2.5,
                color=line_color_tb, markersize=8, alpha=0.8)
    
    # Plot VADER scores if available
    if nltk_available and 'vader_score' in df_numeric.columns:
        valid_vader_mask = df_numeric['vader_score'].notna()
        if valid_vader_mask.any():
            ax.plot(df_numeric.loc[valid_vader_mask, 'id'], 
                    df_numeric.loc[valid_vader_mask, 'vader_score'], 
                    label='VADER Score', marker='s', linewidth=2.5,
                    color=line_color_vader, markersize=8, alpha=0.8)
    
    # Add zero line
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add confidence bands for TextBlob - FIXED
    if 'textblob_score' in df_numeric.columns and valid_tb_mask.any():
        mean_score = df_numeric.loc[valid_tb_mask, 'textblob_score'].mean()
        std_score = df_numeric.loc[valid_tb_mask, 'textblob_score'].std()
        
        # Only add confidence band if we have valid data
        if not np.isnan(mean_score) and not np.isnan(std_score):
            # Use numeric indexing for x-axis
            x_vals = df_numeric.loc[valid_tb_mask, 'id'].values
            if len(x_vals) > 0:
                ax.fill_between(x_vals,
                              mean_score - std_score, 
                              mean_score + std_score, 
                              alpha=0.2, color=line_color_tb,
                              label='±1 Std Dev')
    
    ax.set_title('Sentiment Score Trend Over Time', fontsize=16, fontweight='bold', color=text_color, pad=20)
    ax.set_xlabel('Analysis ID', fontsize=12, fontweight='bold', color=text_color)
    ax.set_ylabel('Sentiment Score', fontsize=12, fontweight='bold', color=text_color)
    
    # Only add legend if we have data
    if ax.has_data():
        ax.legend(facecolor=bg_color, edgecolor=text_color)
    
    ax.grid(True, alpha=0.3, color=text_color)
    ax.tick_params(colors=text_color)
    
    # Set y-axis limits
    ax.set_ylim([-1.1, 1.1])
    
    # Add annotations for extremes - FIXED
    if 'textblob_score' in df_numeric.columns and valid_tb_mask.any():
        tb_scores = df_numeric.loc[valid_tb_mask, 'textblob_score']
        if len(tb_scores) > 0:
            max_idx = tb_scores.idxmax()
            min_idx = tb_scores.idxmin()
            
            if pd.notna(max_idx):
                ax.annotate(f"Max: {tb_scores.loc[max_idx]:.2f}",
                           xy=(df_numeric.loc[max_idx, 'id'], tb_scores.loc[max_idx]),
                           xytext=(0, 20), textcoords='offset points',
                           ha='center', fontweight='bold',
                           arrowprops=dict(arrowstyle='->', color=text_color))
            
            if pd.notna(min_idx):
                ax.annotate(f"Min: {tb_scores.loc[min_idx]:.2f}",
                           xy=(df_numeric.loc[min_idx, 'id'], tb_scores.loc[min_idx]),
                           xytext=(0, -25), textcoords='offset points',
                           ha='center', fontweight='bold',
                           arrowprops=dict(arrowstyle='->', color=text_color))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig

def create_comparison_scatter(df: pd.DataFrame, theme: str = "default") -> plt.Figure:
    """Create TextBlob vs VADER comparison scatter plot - FIXED"""
    if len(df) < 2 or 'vader_score' not in df.columns:
        return None
    
    # Convert to numeric
    df_numeric = convert_to_numeric(df)
    
    # Filter out NaN values
    valid_mask = df_numeric['textblob_score'].notna() & df_numeric['vader_score'].notna()
    valid_df = df_numeric[valid_mask]
    
    if len(valid_df) < 2:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set style
    if theme == "dark":
        plt.style.use('dark_background')
        point_color = '#8B5CF6'
        line_color = '#EF4444'
        bg_color = '#1F2937'
        text_color = 'white'
    else:
        point_color = '#7C3AED'
        line_color = '#DC2626'
        bg_color = 'white'
        text_color = 'black'
    
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    # Create scatter plot with valid numeric data
    scatter = ax.scatter(valid_df['textblob_score'].astype(float), 
                        valid_df['vader_score'].astype(float),
                        c=valid_df['textblob_score'].astype(float), 
                        cmap='RdYlGn',
                        alpha=0.7, s=150, edgecolors='white', linewidth=1)
    
    # Add perfect agreement line
    ax.plot([-1, 1], [-1, 1], '--', color=line_color, alpha=0.7, linewidth=2,
           label='Perfect Agreement')
    
    # Add quadrant lines
    ax.axhline(y=0, color='gray', alpha=0.3, linewidth=1)
    ax.axvline(x=0, color='gray', alpha=0.3, linewidth=1)
    
    # Add quadrant labels
    ax.text(0.5, 0.5, 'Positive\nBoth', ha='center', va='center',
           transform=ax.transAxes, fontweight='bold', fontsize=12,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='#10B98122', edgecolor='#10B981'))
    
    ax.text(0.5, 0.1, 'TextBlob+\nVADER-', ha='center', va='center',
           transform=ax.transAxes, fontweight='bold', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='#F59E0B22', edgecolor='#F59E0B'))
    
    ax.text(0.1, 0.5, 'TextBlob-\nVADER+', ha='center', va='center',
           transform=ax.transAxes, fontweight='bold', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='#F59E0B22', edgecolor='#F59E0B'))
    
    ax.text(0.1, 0.1, 'Negative\nBoth', ha='center', va='center',
           transform=ax.transAxes, fontweight='bold', fontsize=12,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='#EF444422', edgecolor='#EF4444'))
    
    ax.set_title('TextBlob vs VADER Score Comparison', fontsize=16, fontweight='bold', color=text_color, pad=20)
    ax.set_xlabel('TextBlob Score', fontsize=12, fontweight='bold', color=text_color)
    ax.set_ylabel('VADER Score', fontsize=12, fontweight='bold', color=text_color)
    ax.legend(facecolor=bg_color, edgecolor=text_color)
    ax.grid(True, alpha=0.3, color=text_color)
    ax.tick_params(colors=text_color)
    
    # Set limits
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('TextBlob Score', fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig

# ============================================
# FIXED SIDEBAR - Navigation Working
# ============================================

def render_sidebar():
    """Render the sidebar controls - FIXED"""
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h2 style='color: #4F46E5;'>⚙️ Dashboard Controls</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # View Mode Selection - FIXED: Use session state
        view_options = {
            "🏠 Live Analysis": "Live Analysis",
            "📊 History Dashboard": "History Dashboard", 
            "📈 Advanced Analytics": "Advanced Analytics",
            "🤝 Method Comparison": "Method Comparison",
            "💾 Data Export": "Data Export"
        }
        
        # Display as radio buttons
        selected_view_label = st.radio(
            "**Navigation:**",
            list(view_options.keys()),
            index=list(view_options.keys()).index(
                f"🏠 {st.session_state.current_view}" if st.session_state.current_view == "Live Analysis" else
                f"📊 {st.session_state.current_view}" if st.session_state.current_view == "History Dashboard" else
                f"📈 {st.session_state.current_view}" if st.session_state.current_view == "Advanced Analytics" else
                f"🤝 {st.session_state.current_view}" if st.session_state.current_view == "Method Comparison" else
                f"💾 {st.session_state.current_view}"
            ),
            key="view_mode_radio"
        )
        
        # Update session state
        st.session_state.current_view = view_options[selected_view_label]
        
        st.markdown("---")
        
        # Visualization Settings
        st.markdown("### 🎨 Visualization Settings")
        
        # Chart theme - FIXED: Store in session state
        theme_options = ["default", "light", "dark", "vibrant"]
        st.session_state.chart_theme = st.selectbox(
            "Color Theme:",
            theme_options,
            index=theme_options.index(st.session_state.chart_theme),
            help="Choose the color theme for charts"
        )
        
        # Chart style - FIXED: Store in session state
        chart_options = ["Bar Chart", "Line Chart", "Scatter Plot", "All Charts"]
        st.session_state.chart_style = st.selectbox(
            "Chart Style:",
            chart_options,
            index=chart_options.index(st.session_state.chart_style),
            help="Select the type of visualization to display"
        )
        
        st.markdown("---")
        
        # Data Management
        st.markdown("### 🔧 Data Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Refresh", use_container_width=True, 
                        help="Refresh the dashboard and clear cache"):
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear All", use_container_width=True, 
                        type="secondary", help="Clear all analysis history"):
                if st.session_state.analysis_history.empty:
                    st.warning("No data to clear!")
                else:
                    st.session_state.analysis_history = pd.DataFrame(columns=[
                        'id', 'timestamp', 'tweet', 'tweet_short', 
                        'textblob_score', 'textblob_sentiment', 'textblob_subjectivity',
                        'vader_score', 'vader_sentiment', 'vader_positive', 'vader_negative', 'vader_neutral',
                        'word_count', 'char_count', 'analysis_time'
                    ])
                    st.session_state.analysis_count = 0
                    st.success("All data cleared successfully!")
                    time.sleep(1)
                    st.rerun()
        
        st.markdown("---")
        
        # Statistics
        st.markdown("### 📊 Quick Stats")
        
        df = st.session_state.analysis_history
        if not df.empty:
            total = len(df)
            st.metric("Total Analyses", total)
            
            if 'textblob_sentiment' in df.columns:
                positive = (df['textblob_sentiment'] == 'positive').sum()
                negative = (df['textblob_sentiment'] == 'negative').sum()
                neutral = (df['textblob_sentiment'] == 'neutral').sum()
                
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    st.metric("😊", positive, delta=None)
                with col_s2:
                    st.metric("😠", negative, delta=None)
                with col_s3:
                    st.metric("😐", neutral, delta=None)
                
                if total > 0:
                    st.progress(positive / total, text=f"Positive: {positive/total*100:.1f}%")
        else:
            st.info("No analyses yet. Start analyzing!")
        
        st.markdown("---")
        
        # System Info
        st.markdown("### ℹ️ System Info")
        st.caption(f"TextBlob: ✅ Ready")
        st.caption(f"VADER: {'✅ Ready' if nltk_available else '⚠️ Limited'}")
        st.caption(f"Analyses: {st.session_state.analysis_count}")
        st.caption(f"Version: 1.0.0")
        
        return st.session_state.current_view, st.session_state.chart_style

# ============================================
# FIXED LIVE ANALYSIS - Quick Examples Working
# ============================================

def render_live_analysis():
    """Render the Live Analysis view - FIXED"""
    st.markdown('<h1 class="main-title">Tweet Sentiment Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">✍️ Live Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Tweet input area - FIXED: Check for example text
        st.markdown("### Enter Tweet Text")
        
        # Check for example text in session state
        default_text = st.session_state.example_text if st.session_state.example_text else \
            "The customer support team was incredibly helpful and resolved my issue quickly! This product has exceeded all my expectations. Absolutely amazing! 👍"
        
        # Create a unique key for the text area
        tweet_key = f"tweet_input_{st.session_state.get('example_counter', 0)}"
        
        tweet = st.text_area(
            "**Type or paste your tweet here:**",
            value=default_text,
            height=200,
            placeholder="Enter your tweet text here...",
            key=tweet_key,
            help="Enter any text to analyze its sentiment. Longer texts work better with TextBlob, while VADER excels with social media text."
        )
        
        # Analysis options
        st.markdown("### ⚙️ Analysis Options")
        
        col_opt1, col_opt2, col_opt3 = st.columns(3)
        
        with col_opt1:
            use_textblob = st.checkbox("TextBlob", value=True, 
                                      help="Use TextBlob for sentiment analysis")
        
        with col_opt2:
            use_vader = st.checkbox("VADER", value=nltk_available, 
                                   disabled=not nltk_available,
                                   help="Use VADER for social media sentiment analysis" + 
                                        ("" if nltk_available else " (Not available)"))
        
        with col_opt3:
            store_result = st.checkbox("Store Result", value=True,
                                      help="Store this analysis in history")
        
        # Analyze button
        if st.button("🚀 **Analyze Sentiment**", type="primary", use_container_width=True):
            if not tweet.strip():
                st.error("Please enter some text to analyze!")
            else:
                with st.spinner("🤖 Analyzing sentiment..."):
                    # Create progress bar
                    progress_bar = st.progress(0)
                    
                    # Step 1: Text statistics
                    progress_bar.progress(20)
                    time.sleep(0.1)
                    text_stats = calculate_text_stats(tweet)
                    
                    # Step 2: TextBlob analysis
                    progress_bar.progress(40)
                    time.sleep(0.1)
                    if use_textblob:
                        tb_result = analyze_sentiment_textblob(tweet)
                    else:
                        tb_result = {'score': 0.0, 'sentiment': 'N/A', 'subjectivity': 0.0}
                    
                    # Step 3: VADER analysis
                    progress_bar.progress(60)
                    time.sleep(0.1)
                    if use_vader and nltk_available:
                        vader_result = analyze_sentiment_vader(tweet)
                    else:
                        vader_result = {'score': 0.0, 'sentiment': 'N/A'}
                    
                    # Step 4: Prepare results
                    progress_bar.progress(80)
                    time.sleep(0.1)
                    
                    # Store in history if requested
                    if store_result:
                        st.session_state.analysis_count += 1
                        new_entry = pd.DataFrame([{
                            'id': st.session_state.analysis_count,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'tweet': tweet,
                            'tweet_short': (tweet[:60] + '...') if len(tweet) > 60 else tweet,
                            'textblob_score': float(tb_result['score']),
                            'textblob_sentiment': tb_result['sentiment'],
                            'textblob_subjectivity': float(tb_result.get('subjectivity', 0.0)),
                            'vader_score': float(vader_result['score']),
                            'vader_sentiment': vader_result['sentiment'],
                            'vader_positive': float(vader_result.get('positive', 0.0)),
                            'vader_negative': float(vader_result.get('negative', 0.0)),
                            'vader_neutral': float(vader_result.get('neutral', 0.0)),
                            'word_count': int(text_stats['word_count']),
                            'char_count': int(text_stats['char_count']),
                            'analysis_time': time.time()
                        }])
                        
                        st.session_state.analysis_history = pd.concat(
                            [st.session_state.analysis_history, new_entry], 
                            ignore_index=True
                        )
                    
                    progress_bar.progress(100)
                    time.sleep(0.2)
                    progress_bar.empty()
                    
                    # Display results
                    st.success(f"✅ Analysis complete! Results:")
                    
                    # Results in columns
                    st.markdown("### 📊 Analysis Results")
                    
                    res_col1, res_col2, res_col3 = st.columns(3)
                    
                    with res_col1:
                        if use_textblob:
                            st.markdown(f"""
                            <div class='metric-card'>
                                <h4>TextBlob</h4>
                                <div class='metric-value'>{tb_result['score']:.3f}</div>
                                <div class='{get_sentiment_style(tb_result["sentiment"])}'>
                                    {tb_result['emoji']} {tb_result['sentiment'].title()}
                                </div>
                                <p style='margin-top: 10px; font-size: 0.9em; color: #666;'>
                                    Subjectivity: {tb_result.get('subjectivity', 0):.2f}<br>
                                    {tb_result.get('intensity', '')}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with res_col2:
                        if use_vader and nltk_available:
                            st.markdown(f"""
                            <div class='metric-card'>
                                <h4>VADER</h4>
                                <div class='metric-value'>{vader_result['score']:.3f}</div>
                                <div class='{get_sentiment_style(vader_result["sentiment"])}'>
                                    {vader_result['emoji']} {vader_result['sentiment'].title()}
                                </div>
                                <p style='margin-top: 10px; font-size: 0.9em; color: #666;'>
                                    Confidence: {vader_result.get('confidence', 0):.2f}<br>
                                    {vader_result.get('intensity', '')}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with res_col3:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <h4>Text Stats</h4>
                            <div class='metric-value'>{text_stats['word_count']}</div>
                            <p style='margin-top: 10px; font-size: 0.9em; color: #666;'>
                                Words: {text_stats['word_count']}<br>
                                Characters: {text_stats['char_count']}<br>
                                Sentences: {text_stats.get('sentence_count', 0)}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Agreement indicator
                    if use_textblob and use_vader and nltk_available:
                        if tb_result['sentiment'] == vader_result['sentiment']:
                            st.info(f"✅ **Agreement**: Both methods agree on **{tb_result['sentiment']}** sentiment")
                        else:
                            st.warning(f"⚠️ **Disagreement**: TextBlob: {tb_result['sentiment']}, VADER: {vader_result['sentiment']}")
    
    with col2:
        st.markdown("### 📝 Quick Examples")
        st.markdown("Try these examples to see how sentiment analysis works:")
        
        examples = [
            ("😊 Positive", "I absolutely love this product! It's changed my life for the better. Five stars! ⭐⭐⭐⭐⭐", "positive"),
            ("😠 Negative", "Worst customer service ever. Waited for 2 hours and still didn't get help. Never buying again! 👎", "negative"),
            ("😐 Neutral", "The package arrived on time as expected. Nothing special but nothing bad either.", "neutral"),
            ("🔥 Strong Positive", "THIS IS INCREDIBLE! BEST PURCHASE I'VE EVER MADE!!! THANK YOU SO MUCH! 😍🎉", "positive"),
            ("🤔 Mixed Feelings", "The quality is good but it's way too expensive for what you get. Could be better.", "neutral"),
            ("😡 Angry", "I'm furious! This product broke after 2 days. Complete waste of money! 😡", "negative"),
            ("🎉 Excited", "Just got the new update and it's amazing! So many new features! Can't wait to explore! 🚀", "positive"),
            ("😞 Disappointed", "Expected so much more based on the reviews. Really disappointed with the performance.", "negative"),
            ("📱 Social Media", "LOL just saw the funniest meme ever! ROFL 😂 #funny #viral", "positive"),
            ("💼 Professional", "The quarterly report shows promising growth trends with improved customer satisfaction metrics.", "neutral")
        ]
        
        for emoji, text, sentiment in examples:
            # FIXED: Use form submission to update example text
            if st.button(f"{emoji} {text[:40]}...", 
                        key=f"ex_{text[:10].replace(' ', '_')}", 
                        use_container_width=True, 
                        help=f"Example: {sentiment.title()} sentiment"):
                # Store example text in session state
                st.session_state.example_text = text
                # Increment counter to force text area refresh
                if 'example_counter' not in st.session_state:
                    st.session_state.example_counter = 0
                st.session_state.example_counter += 1
                # Rerun to update the text area
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 💡 Tips for Better Analysis")
        st.info("""
        **TextBlob:**
        • Better for longer, formal text
        • Provides subjectivity scores
        • Good for articles, reviews
        
        **VADER:**
        • Optimized for social media
        • Handles emojis, slang, abbreviations
        • Better for tweets, comments
        
        **General Tips:**
        • Use complete sentences
        • Include context
        • Check both methods for comparison
        """)

# ============================================
# FIXED HISTORY DASHBOARD
# ============================================

def render_history_dashboard(chart_style: str):
    """Render the History Dashboard view"""
    st.markdown('<h1 class="main-title">Tweet Sentiment Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">📊 History Dashboard</h2>', unsafe_allow_html=True)
    
    df = st.session_state.analysis_history
    
    if df.empty:
        st.warning("""
        ## 📭 No Analysis History Yet!
        
        Your history dashboard is empty because you haven't analyzed any tweets yet.
        
        **To get started:**
        1. Switch to **'Live Analysis'** mode
        2. Enter or paste a tweet
        3. Click 'Analyze Sentiment'
        4. Results will appear here automatically!
        
        Try using the example tweets to see how it works! 🚀
        """)
        
        if st.button("🚀 Go to Live Analysis", type="primary", use_container_width=True):
            st.session_state.current_view = "Live Analysis"
            st.rerun()
        
        return
    
    # Convert to numeric for calculations
    df_numeric = convert_to_numeric(df)
    
    # PHASE 4: Advanced Metrics
    st.markdown("### 📈 Advanced Analytics Summary")
    
    metrics = calculate_advanced_metrics(df)
    
    # Display metrics in cards
    metric_cols = st.columns(4)
    
    with metric_cols[0]:
        st.markdown(f"""
        <div class='metric-card'>
            <h4>Total Analyses</h4>
            <div class='metric-value'>{metrics.get('total_analyses', 0)}</div>
            <p style='color: #666; font-size: 0.9em;'>
                📅 Since: {df['timestamp'].min()[:10] if not df.empty else 'N/A'}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_cols[1]:
        avg_score = metrics.get('tb_mean', 0)
        st.markdown(f"""
        <div class='metric-card'>
            <h4>Average Score</h4>
            <div class='metric-value'>{avg_score:.3f}</div>
            <p style='color: #666; font-size: 0.9em;'>
                📊 Range: {metrics.get('tb_min', 0):.2f} to {metrics.get('tb_max', 0):.2f}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_cols[2]:
        pos_percent = metrics.get('tb_positive_percent', 0)
        st.markdown(f"""
        <div class='metric-card'>
            <h4>Positive Rate</h4>
            <div class='metric-value'>{pos_percent:.1f}%</div>
            <p style='color: #666; font-size: 0.9em;'>
                😊 {metrics.get('tb_positive_count', 0)} positive
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_cols[3]:
        if nltk_available and 'vader_mean' in metrics:
            vader_avg = metrics.get('vader_mean', 0)
            st.markdown(f"""
            <div class='metric-card'>
                <h4>VADER Average</h4>
                <div class='metric-value'>{vader_avg:.3f}</div>
                <p style='color: #666; font-size: 0.9em;'>
                    🤝 Agreement: {metrics.get('agreement_rate', 0):.1f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            consistency = 100 - (metrics.get('tb_std', 0) * 100)
            st.markdown(f"""
            <div class='metric-card'>
                <h4>Data Consistency</h4>
                <div class='metric-value'>{consistency:.1f}%</div>
                <p style='color: #666; font-size: 0.9em;'>
                    📐 Std Dev: {metrics.get('tb_std', 0):.3f}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # PHASE 5: Visualizations
    st.markdown("### 📊 Visualizations")
    
    # Create tabs for different chart types
    if chart_style == "All Charts":
        tab1, tab2, tab3 = st.tabs(["📈 Distribution", "📉 Trends", "🔄 Comparison"])
        
        with tab1:
            fig1 = create_sentiment_distribution_chart(df, st.session_state.chart_theme)
            if fig1:
                st.pyplot(fig1)
                plt.close(fig1)
            else:
                st.info("No sentiment data available for distribution chart.")
        
        with tab2:
            fig2 = create_score_trend_chart(df_numeric, st.session_state.chart_theme)
            if fig2:
                st.pyplot(fig2)
                plt.close(fig2)
            else:
                st.info("Need at least 2 analyses with numeric scores for trend chart.")
        
        with tab3:
            fig3 = create_comparison_scatter(df_numeric, st.session_state.chart_theme)
            if fig3:
                st.pyplot(fig3)
                plt.close(fig3)
            else:
                st.info("Need at least 2 analyses with both TextBlob and VADER scores for comparison.")
    
    else:
        # Show single chart based on selection
        if chart_style == "Bar Chart":
            fig = create_sentiment_distribution_chart(df, st.session_state.chart_theme)
        elif chart_style == "Line Chart":
            fig = create_score_trend_chart(df_numeric, st.session_state.chart_theme)
        elif chart_style == "Scatter Plot":
            fig = create_comparison_scatter(df_numeric, st.session_state.chart_theme)
        else:
            fig = create_sentiment_distribution_chart(df, st.session_state.chart_theme)
        
        if fig:
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info(f"Not enough data for {chart_style}. Analyze more tweets first.")
    
    # Recent Analyses Table
    st.markdown("### 📋 Recent Analyses")
    
    with st.expander("View All Analyses", expanded=False):
        # Display with formatting
        display_df = df.copy()
        if not display_df.empty:
            # Format sentiment columns with emojis
            def sentiment_with_emoji(sentiment):
                emojis = {'positive': '😊', 'negative': '😠', 'neutral': '😐', 'N/A': '❓', 'error': '❌'}
                return f"{emojis.get(sentiment, '')} {sentiment.title()}"
            
            if 'textblob_sentiment' in display_df.columns:
                display_df['TextBlob'] = display_df['textblob_sentiment'].apply(sentiment_with_emoji)
            
            if 'vader_sentiment' in display_df.columns:
                display_df['VADER'] = display_df['vader_sentiment'].apply(sentiment_with_emoji)
            
            # Select columns to display
            display_cols = ['id', 'timestamp', 'tweet_short', 'TextBlob']
            if 'VADER' in display_df.columns:
                display_cols.append('VADER')
            display_cols.extend(['textblob_score', 'word_count'])
            
            # Show dataframe
            st.dataframe(
                display_df[display_cols].rename(columns={
                    'id': 'ID',
                    'timestamp': 'Timestamp',
                    'tweet_short': 'Tweet',
                    'textblob_score': 'Score',
                    'word_count': 'Words'
                }),
                use_container_width=True,
                height=400
            )

# ============================================
# FIXED ADVANCED ANALYTICS
# ============================================

def render_advanced_analytics():
    """Render the Advanced Analytics view"""
    st.markdown('<h1 class="main-title">Tweet Sentiment Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">📈 Advanced Analytics</h2>', unsafe_allow_html=True)
    
    df = st.session_state.analysis_history
    
    if df.empty or len(df) < 2:
        st.warning("""
        ## 📊 More Data Needed!
        
        Advanced analytics requires at least 2 analyses to provide meaningful insights.
        
        **Current Status:**
        - Analyses: {len(df)}
        - Minimum Required: 2
        
        **Next Steps:**
        1. Analyze more tweets in **Live Analysis** mode
        2. Try different types of text (positive, negative, neutral)
        3. Compare TextBlob vs VADER results
        
        The more data you analyze, the better insights you'll get! 📈
        """)
        
        if st.button("🚀 Analyze More Tweets", type="primary", use_container_width=True):
            st.session_state.current_view = "Live Analysis"
            st.rerun()
        
        return
    
    # Convert to numeric
    df_numeric = convert_to_numeric(df)
    
    # Calculate advanced metrics
    metrics = calculate_advanced_metrics(df)
    
    # Statistical Analysis Section
    st.markdown("### 📊 Statistical Analysis")
    
    stat_cols = st.columns(3)
    
    with stat_cols[0]:
        st.markdown("#### Central Tendency")
        st.metric("Mean Score", f"{metrics.get('tb_mean', 0):.3f}")
        st.metric("Median Score", f"{metrics.get('tb_median', 0):.3f}")
        
        # Calculate mode safely
        if 'textblob_score' in df_numeric.columns:
            mode_series = df_numeric['textblob_score'].dropna().mode()
            mode_value = mode_series.iloc[0] if not mode_series.empty else 0.0
            st.metric("Mode Score", f"{mode_value:.3f}")
    
    with stat_cols[1]:
        st.markdown("#### Dispersion")
        st.metric("Standard Deviation", f"{metrics.get('tb_std', 0):.3f}")
        st.metric("Variance", f"{metrics.get('tb_std', 0)**2:.3f}")
        st.metric("Range", f"{metrics.get('tb_range', 0):.3f}")
    
    with stat_cols[2]:
        st.markdown("#### Distribution")
        st.metric("Skewness", f"{metrics.get('tb_skew', 0):.3f}")
        st.metric("25th Percentile", f"{metrics.get('tb_q25', 0):.3f}")
        st.metric("75th Percentile", f"{metrics.get('tb_q75', 0):.3f}")
        st.metric("IQR", f"{metrics.get('tb_iqr', 0):.3f}")

# ============================================
# FIXED METHOD COMPARISON
# ============================================

def render_method_comparison():
    """Render the Method Comparison view"""
    st.markdown('<h1 class="main-title">Tweet Sentiment Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">🤝 Method Comparison</h2>', unsafe_allow_html=True)
    
    df = st.session_state.analysis_history
    
    if df.empty or 'vader_score' not in df.columns or not nltk_available:
        st.warning("""
        ## 🔄 Method Comparison Unavailable
        
        Method comparison requires:
        1. At least 1 analysis with **both** TextBlob and VADER
        2. VADER to be properly initialized
        
        **Current Status:**
        - Total Analyses: {len(df)}
        - VADER Available: {'Yes' if nltk_available else 'No'}
        - Analyses with VADER: {len(df[df['vader_sentiment'] != 'N/A']) if 'vader_sentiment' in df.columns else 0}
        
        **To enable comparison:**
        1. Make sure VADER is working (check sidebar)
        2. Analyze tweets with **both** TextBlob and VADER enabled
        3. Come back here to see the comparison!
        """)
        
        if st.button("🔧 Check VADER Status", use_container_width=True):
            st.rerun()
        
        return
    
    # Convert to numeric
    df_numeric = convert_to_numeric(df)
    
    # Filter only analyses with both methods
    comparison_df = df_numeric[(df['textblob_sentiment'] != 'N/A') & (df['vader_sentiment'] != 'N/A')]
    
    if comparison_df.empty:
        st.error("No analyses with both TextBlob and VADER results found!")
        return
    
    # Agreement Analysis
    st.markdown("### 📊 Agreement Analysis")
    
    # Calculate agreement metrics
    agreement_mask = comparison_df['textblob_sentiment'] == comparison_df['vader_sentiment']
    agreement_rate = agreement_mask.mean() * 100
    total_comparisons = len(comparison_df)
    agreements = agreement_mask.sum()
    disagreements = total_comparisons - agreements
    
    # Display agreement metrics
    agree_col1, agree_col2, agree_col3, agree_col4 = st.columns(4)
    
    with agree_col1:
        st.metric("Agreement Rate", f"{agreement_rate:.1f}%")
    
    with agree_col2:
        st.metric("Total Comparisons", total_comparisons)
    
    with agree_col3:
        st.metric("Agreements", agreements)
    
    with agree_col4:
        st.metric("Disagreements", disagreements)

# ============================================
# FIXED DATA EXPORT
# ============================================

def render_data_export():
    """Render the Data Export view"""
    st.markdown('<h1 class="main-title">Tweet Sentiment Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">💾 Data Export</h2>', unsafe_allow_html=True)
    
    df = st.session_state.analysis_history
    
    if df.empty:
        st.warning("""
        ## 📁 No Data to Export
        
        Your analysis history is currently empty.
        
        **To export data:**
        1. Analyze some tweets in **Live Analysis** mode
        2. Make sure to check "Store Result" when analyzing
        3. Come back here to export your data
        
        Data can be exported in multiple formats for further analysis! 💾
        """)
        
        if st.button("🚀 Start Analyzing", type="primary", use_container_width=True):
            st.session_state.current_view = "Live Analysis"
            st.rerun()
        
        return
    
    # Data Summary
    st.markdown("### 📊 Export Summary")
    
    summary_cols = st.columns(4)
    
    with summary_cols[0]:
        st.metric("Total Records", len(df))
    
    with summary_cols[1]:
        time_range = f"{df['timestamp'].min()[:10] if not df.empty else 'N/A'} to {df['timestamp'].max()[:10] if not df.empty else 'N/A'}"
        st.metric("Time Range", time_range)
    
    with summary_cols[2]:
        file_size = sys.getsizeof(df) / 1024
        st.metric("File Size (approx)", f"{file_size:.1f} KB")
    
    with summary_cols[3]:
        unique_tweets = df['tweet'].nunique() if 'tweet' in df.columns else 0
        st.metric("Unique Tweets", unique_tweets)
    
    # Data Preview
    st.markdown("### 👁️ Data Preview")
    
    with st.expander("Preview First 10 Records", expanded=True):
        preview_df = df.head(10).copy()
        
        # Format sentiment columns
        if 'textblob_sentiment' in preview_df.columns:
            preview_df['TextBlob'] = preview_df['textblob_sentiment'].apply(
                lambda x: f"😊 {x.title()}" if x == 'positive' else 
                         f"😠 {x.title()}" if x == 'negative' else 
                         f"😐 {x.title()}" if x == 'neutral' else x
            )
        
        if 'vader_sentiment' in preview_df.columns:
            preview_df['VADER'] = preview_df['vader_sentiment'].apply(
                lambda x: f"😊 {x.title()}" if x == 'positive' else 
                         f"😠 {x.title()}" if x == 'negative' else 
                         f"😐 {x.title()}" if x == 'neutral' else x
            )
        
        # Select columns for preview
        preview_cols = ['id', 'timestamp', 'tweet_short', 'TextBlob']
        if 'VADER' in preview_df.columns:
            preview_cols.append('VADER')
        preview_cols.extend(['textblob_score', 'word_count'])
        
        st.dataframe(
            preview_df[preview_cols].rename(columns={
                'id': 'ID',
                'timestamp': 'Timestamp',
                'tweet_short': 'Tweet Preview',
                'textblob_score': 'Score',
                'word_count': 'Words'
            }),
            use_container_width=True
        )
    
    # Export Options
    st.markdown("### ⚙️ Export Options")
    
    export_format = st.radio(
        "Select Export Format:",
        ["CSV (Comma Separated Values)", 
         "JSON (JavaScript Object Notation)", 
         "Excel (Microsoft Excel)"],
        horizontal=True
    )
    
    # Prepare data for export
    export_df = df.copy()
    
    # Export Buttons
    st.markdown("### 📥 Download Data")
    
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    
    with col_exp1:
        # CSV Export
        if "CSV" in export_format:
            csv_data = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                help="Download data as CSV file"
            )
    
    with col_exp2:
        # JSON Export
        if "JSON" in export_format:
            json_data = export_df.to_json(orient='records', indent=2, default_handler=str)
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
                help="Download data as JSON file"
            )
    
    with col_exp3:
        # Excel Export
        if "Excel" in export_format:
            try:
                import openpyxl
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    export_df.to_excel(writer, index=False, sheet_name='Sentiment Analysis')
                
                st.download_button(
                    label="📥 Download Excel",
                    data=buffer.getvalue(),
                    file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    help="Download data as Excel file"
                )
            except ImportError:
                st.error("Excel export requires openpyxl. Install with: `pip install openpyxl`")
    
    # Export Statistics
    st.markdown("### 📈 Export Statistics")
    
    stat_cols = st.columns(3)
    
    with stat_cols[0]:
        st.metric("Records to Export", len(export_df))
    
    with stat_cols[1]:
        if 'textblob_sentiment' in export_df.columns:
            sentiment_dist = export_df['textblob_sentiment'].value_counts()
            most_common = sentiment_dist.index[0] if not sentiment_dist.empty else "N/A"
            st.metric("Most Common Sentiment", most_common)
    
    with stat_cols[2]:
        if 'textblob_score' in export_df.columns:
            # Convert to numeric for calculation
            scores = pd.to_numeric(export_df['textblob_score'], errors='coerce')
            avg_score = scores.mean()
            st.metric("Average Score", f"{avg_score:.3f}")

# ============================================
# MAIN APP FUNCTION - FIXED
# ============================================

def main():
    """Main application function - FIXED"""
    
    # Render sidebar
    view_mode, chart_style = render_sidebar()
    
    # Render main content based on view mode
    try:
        if view_mode == "Live Analysis":
            render_live_analysis()
        elif view_mode == "History Dashboard":
            render_history_dashboard(chart_style)
        elif view_mode == "Advanced Analytics":
            render_advanced_analytics()
        elif view_mode == "Method Comparison":
            render_method_comparison()
        elif view_mode == "Data Export":
            render_data_export()
        else:
            render_live_analysis()  # Default view
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.code(traceback.format_exc())
    
    # Footer
    st.markdown("---")
    
    # Deployment Progress
    st.markdown("### 🚀 Deployment Progress")
    
    progress_cols = st.columns(6)
    
    with progress_cols[0]:
        st.success("""
        ✅ **Phase 1**  
        Basic App
        """)
    
    with progress_cols[1]:
        st.success("""
        ✅ **Phase 2**  
        TextBlob
        """)
    
    with progress_cols[2]:
        st.success("""
        ✅ **Phase 3**  
        NLTK/VADER
        """)
    
    with progress_cols[3]:
        st.success("""
        ✅ **Phase 4**  
        pandas/numpy
        """)
    
    with progress_cols[4]:
        st.success("""
        ✅ **Phase 5**  
        Visualizations
        """)
    
    with progress_cols[5]:
        st.info("""
        🔄 **Phase 6**  
        ML Model
        """)
    
    # Copyright and info
    st.markdown("""
    <div style='text-align: center; margin-top: 2rem; color: #666; font-size: 0.9em;'>
        <hr style='border: none; border-top: 1px solid #E5E7EB; margin: 1rem 0;'>
        <p>
            <strong>Tweet Sentiment Analytics Dashboard</strong> v1.0.0 • 
            Made with ❤️ using Streamlit • 
            Phases 1-5 Complete
        </p>
        <p style='font-size: 0.8em;'>
            TextBlob • NLTK VADER • pandas • numpy • matplotlib • Streamlit
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# APP ENTRY POINT
# ============================================

if __name__ == "__main__":
    main()
