"""
TWEET SENTIMENT ANALYTICS DASHBOARD
Complete Phases 1-5: Basic App → TextBlob → VADER → pandas/numpy → Visualizations
Deployment-ready for Streamlit Cloud
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
        'Get Help': 'https://github.com/yourusername/tweet-sentiment-analytics',
        'Report a bug': 'https://github.com/yourusername/tweet-sentiment-analytics/issues',
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
# SESSION STATE MANAGEMENT
# ============================================

def initialize_session_state():
    """Initialize all session state variables"""
    if 'analysis_history' not in st.session_state:
        # PHASE 4: pandas DataFrame structure
        st.session_state.analysis_history = pd.DataFrame(columns=[
            'id', 'timestamp', 'tweet', 'tweet_short', 
            'textblob_score', 'textblob_sentiment', 'textblob_subjectivity',
            'vader_score', 'vader_sentiment', 'vader_positive', 'vader_negative', 'vader_neutral',
            'word_count', 'char_count', 'analysis_time'
        ])
    
    if 'analysis_count' not in st.session_state:
        st.session_state.analysis_count = 0
    
    if 'current_view' not in st.session_state:
        st.session_state.current_view = "Live Analysis"
    
    if 'chart_theme' not in st.session_state:
        st.session_state.chart_theme = "default"

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

def calculate_advanced_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate advanced analytics using pandas and numpy
    PHASE 4: pandas/numpy Integration
    """
    if df.empty:
        return {}
    
    metrics = {}
    
    # Basic counts
    metrics['total_analyses'] = len(df)
    
    # TextBlob metrics
    if 'textblob_score' in df.columns:
        scores = df['textblob_score'].astype(float).values
        
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
    
    # VADER metrics
    if 'vader_score' in df.columns:
        vader_scores = df['vader_score'].astype(float).values
        
        metrics.update({
            'vader_mean': float(np.mean(vader_scores)),
            'vader_std': float(np.std(vader_scores)),
            'vader_min': float(np.min(vader_scores)),
            'vader_max': float(np.max(vader_scores)),
        })
    
    # Sentiment distribution
    if 'textblob_sentiment' in df.columns:
        sentiment_counts = df['textblob_sentiment'].value_counts()
        for sentiment in ['positive', 'negative', 'neutral']:
            count = sentiment_counts.get(sentiment, 0)
            metrics[f'tb_{sentiment}_count'] = int(count)
            metrics[f'tb_{sentiment}_percent'] = float((count / len(df)) * 100) if len(df) > 0 else 0.0
    
    # Text statistics
    if 'word_count' in df.columns:
        metrics.update({
            'avg_word_count': float(df['word_count'].mean()),
            'total_words': int(df['word_count'].sum()),
            'max_words': int(df['word_count'].max()),
            'min_words': int(df['word_count'].min()),
        })
    
    # Correlation between TextBlob and VADER
    if 'textblob_score' in df.columns and 'vader_score' in df.columns:
        if len(df) > 1:
            correlation = np.corrcoef(df['textblob_score'], df['vader_score'])[0, 1]
            metrics['tb_vader_correlation'] = float(correlation)
    
    # Agreement rate
    if 'textblob_sentiment' in df.columns and 'vader_sentiment' in df.columns:
        agreement = (df['textblob_sentiment'] == df['vader_sentiment']).mean()
        metrics['agreement_rate'] = float(agreement * 100)
    
    return metrics

def generate_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Generate correlation matrix for numeric columns"""
    numeric_cols = ['textblob_score', 'vader_score', 'word_count', 'textblob_subjectivity']
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    if len(available_cols) >= 2:
        return df[available_cols].corr()
    return pd.DataFrame()

# ============================================
# VISUALIZATION FUNCTIONS - PHASE 5
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
            percentage = (count / total) * 100
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
    """Create sentiment score trend chart"""
    if len(df) < 2:
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
    
    # Plot TextBlob scores
    ax.plot(df['id'], df['textblob_score'], 
            label='TextBlob Score', marker='o', linewidth=2.5,
            color=line_color_tb, markersize=8, alpha=0.8)
    
    # Plot VADER scores if available
    if nltk_available and 'vader_score' in df.columns:
        ax.plot(df['id'], df['vader_score'], 
                label='VADER Score', marker='s', linewidth=2.5,
                color=line_color_vader, markersize=8, alpha=0.8)
    
    # Add zero line
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add confidence bands for TextBlob
    if 'textblob_score' in df.columns:
        mean_score = df['textblob_score'].mean()
        std_score = df['textblob_score'].std()
        ax.fill_between(df['id'], 
                       mean_score - std_score, 
                       mean_score + std_score, 
                       alpha=0.2, color=line_color_tb,
                       label='±1 Std Dev')
    
    ax.set_title('Sentiment Score Trend Over Time', fontsize=16, fontweight='bold', color=text_color, pad=20)
    ax.set_xlabel('Analysis ID', fontsize=12, fontweight='bold', color=text_color)
    ax.set_ylabel('Sentiment Score', fontsize=12, fontweight='bold', color=text_color)
    ax.legend(facecolor=bg_color, edgecolor=text_color)
    ax.grid(True, alpha=0.3, color=text_color)
    ax.tick_params(colors=text_color)
    
    # Set y-axis limits
    ax.set_ylim([-1.1, 1.1])
    
    # Add annotations for extremes
    if 'textblob_score' in df.columns:
        max_idx = df['textblob_score'].idxmax()
        min_idx = df['textblob_score'].idxmin()
        
        if pd.notna(max_idx):
            ax.annotate(f"Max: {df.loc[max_idx, 'textblob_score']:.2f}",
                       xy=(df.loc[max_idx, 'id'], df.loc[max_idx, 'textblob_score']),
                       xytext=(0, 20), textcoords='offset points',
                       ha='center', fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color=text_color))
        
        if pd.notna(min_idx):
            ax.annotate(f"Min: {df.loc[min_idx, 'textblob_score']:.2f}",
                       xy=(df.loc[min_idx, 'id'], df.loc[min_idx, 'textblob_score']),
                       xytext=(0, -25), textcoords='offset points',
                       ha='center', fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color=text_color))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig

def create_comparison_scatter(df: pd.DataFrame, theme: str = "default") -> plt.Figure:
    """Create TextBlob vs VADER comparison scatter plot"""
    if len(df) < 2 or 'vader_score' not in df.columns:
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
    
    # Create scatter plot
    scatter = ax.scatter(df['textblob_score'], df['vader_score'],
                        c=df['textblob_score'], cmap='RdYlGn',
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

def create_advanced_metrics_chart(metrics: Dict[str, Any], theme: str = "default") -> plt.Figure:
    """Create advanced metrics radar/spider chart"""
    if not metrics:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # Set style
    if theme == "dark":
        plt.style.use('dark_background')
        fill_color = 'rgba(59, 130, 246, 0.3)'
        line_color = '#3B82F6'
        bg_color = '#1F2937'
        text_color = 'white'
    else:
        fill_color = 'rgba(37, 99, 235, 0.3)'
        line_color = '#2563EB'
        bg_color = 'white'
        text_color = 'black'
    
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    # Select key metrics for radar chart
    radar_metrics = {
        'Positivity': metrics.get('tb_positive_percent', 0),
        'Negativity': metrics.get('tb_negative_percent', 0),
        'Agreement': metrics.get('agreement_rate', 0),
        'Score Range': metrics.get('tb_range', 0) * 50,  # Scale
        'Consistency': 100 - (metrics.get('tb_std', 0) * 100),  # Invert std
        'Subjectivity': metrics.get('avg_subjectivity', 50) if 'avg_subjectivity' in metrics else 50
    }
    
    categories = list(radar_metrics.keys())
    values = list(radar_metrics.values())
    
    # Complete the circle
    values += values[:1]
    categories += categories[:1]
    
    # Create angles
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    # Plot
    ax.plot(angles, values, 'o-', linewidth=2, color=line_color, markersize=8)
    ax.fill(angles, values, alpha=0.25, color=fill_color)
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories[:-1], fontsize=11, fontweight='bold', color=text_color)
    
    # Set y-axis
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], color=text_color, fontsize=9)
    
    ax.set_title('Advanced Analytics Overview', fontsize=16, fontweight='bold', 
                color=text_color, pad=30)
    
    # Add grid
    ax.grid(True, alpha=0.3, color=text_color)
    
    plt.tight_layout()
    return fig

# ============================================
# MAIN APP LAYOUT
# ============================================

def render_sidebar():
    """Render the sidebar controls"""
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h2 style='color: #4F46E5;'>⚙️ Dashboard Controls</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # View Mode Selection
        view_mode = st.radio(
            "**Navigation:**",
            ["🏠 Live Analysis", "📊 History Dashboard", "📈 Advanced Analytics", 
             "🤝 Method Comparison", "💾 Data Export"],
            index=0,
            key="view_mode_radio"
        )
        
        # Extract mode name
        view_mode_name = view_mode.split(" ")[-1]
        st.session_state.current_view = view_mode_name
        
        st.markdown("---")
        
        # Visualization Settings
        st.markdown("### 🎨 Visualization Settings")
        
        st.session_state.chart_theme = st.selectbox(
            "Color Theme:",
            ["default", "light", "dark", "vibrant"],
            index=0,
            help="Choose the color theme for charts"
        )
        
        chart_style = st.selectbox(
            "Chart Style:",
            ["Bar Chart", "Line Chart", "Scatter Plot", "Radar Chart", "All Charts"],
            index=0,
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
                    st.session_state.analysis_history = pd.DataFrame(columns=st.session_state.analysis_history.columns)
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
        
        return view_mode_name, chart_style

def render_live_analysis():
    """Render the Live Analysis view"""
    st.markdown('<h1 class="main-title">Tweet Sentiment Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">✍️ Live Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Tweet input area
        st.markdown("### Enter Tweet Text")
        
        # Check for example text
        if 'example_text' in st.session_state:
            default_text = st.session_state.example_text
            del st.session_state.example_text
        else:
            default_text = "The customer support team was incredibly helpful and resolved my issue quickly! This product has exceeded all my expectations. Absolutely amazing! 👍"
        
        tweet = st.text_area(
            "**Type or paste your tweet here:**",
            value=default_text,
            height=200,
            placeholder="Enter your tweet text here...",
            key="tweet_input",
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
                            'textblob_score': tb_result['score'],
                            'textblob_sentiment': tb_result['sentiment'],
                            'textblob_subjectivity': tb_result.get('subjectivity', 0.0),
                            'vader_score': vader_result['score'],
                            'vader_sentiment': vader_result['sentiment'],
                            'vader_positive': vader_result.get('positive', 0.0),
                            'vader_negative': vader_result.get('negative', 0.0),
                            'vader_neutral': vader_result.get('neutral', 0.0),
                            'word_count': text_stats['word_count'],
                            'char_count': text_stats['char_count'],
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
            if st.button(f"{emoji} {text[:40]}...", key=f"ex_{text[:10]}", 
                        use_container_width=True, 
                        help=f"Example: {sentiment.title()} sentiment"):
                st.session_state.example_text = text
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
        st.markdown(f"""
        <div class='metric-card'>
            <h4>Average Score</h4>
            <div class='metric-value'>{metrics.get('tb_mean', 0):.3f}</div>
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
            st.markdown(f"""
            <div class='metric-card'>
                <h4>VADER Average</h4>
                <div class='metric-value'>{metrics.get('vader_mean', 0):.3f}</div>
                <p style='color: #666; font-size: 0.9em;'>
                    🤝 Agreement: {metrics.get('agreement_rate', 0):.1f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='metric-card'>
                <h4>Data Consistency</h4>
                <div class='metric-value'>{100 - metrics.get('tb_std', 0)*100:.1f}%</div>
                <p style='color: #666; font-size: 0.9em;'>
                    📐 Std Dev: {metrics.get('tb_std', 0):.3f}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # PHASE 5: Visualizations
    st.markdown("### 📊 Visualizations")
    
    # Create tabs for different chart types
    if chart_style == "All Charts":
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Distribution", "📉 Trends", "🔄 Comparison", "🎯 Advanced"])
        
        with tab1:
            fig1 = create_sentiment_distribution_chart(df, st.session_state.chart_theme)
            if fig1:
                st.pyplot(fig1)
                plt.close(fig1)
        
        with tab2:
            fig2 = create_score_trend_chart(df, st.session_state.chart_theme)
            if fig2:
                st.pyplot(fig2)
                plt.close(fig2)
        
        with tab3:
            fig3 = create_comparison_scatter(df, st.session_state.chart_theme)
            if fig3:
                st.pyplot(fig3)
                plt.close(fig3)
        
        with tab4:
            fig4 = create_advanced_metrics_chart(metrics, st.session_state.chart_theme)
            if fig4:
                st.pyplot(fig4)
                plt.close(fig4)
    
    else:
        # Show single chart based on selection
        if chart_style == "Bar Chart":
            fig = create_sentiment_distribution_chart(df, st.session_state.chart_theme)
        elif chart_style == "Line Chart":
            fig = create_score_trend_chart(df, st.session_state.chart_theme)
        elif chart_style == "Scatter Plot":
            fig = create_comparison_scatter(df, st.session_state.chart_theme)
        elif chart_style == "Radar Chart":
            fig = create_advanced_metrics_chart(metrics, st.session_state.chart_theme)
        else:
            fig = create_sentiment_distribution_chart(df, st.session_state.chart_theme)
        
        if fig:
            st.pyplot(fig)
            plt.close(fig)
    
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
            
            # Quick filters
            st.markdown("**Quick Filters:**")
            filter_cols = st.columns(5)
            
            with filter_cols[0]:
                if st.button("😊 Positive", use_container_width=True):
                    filtered = df[df['textblob_sentiment'] == 'positive']
                    st.write(f"Found: {len(filtered)} positive analyses")
            
            with filter_cols[1]:
                if st.button("😠 Negative", use_container_width=True):
                    filtered = df[df['textblob_sentiment'] == 'negative']
                    st.write(f"Found: {len(filtered)} negative analyses")
            
            with filter_cols[2]:
                if st.button("😐 Neutral", use_container_width=True):
                    filtered = df[df['textblob_sentiment'] == 'neutral']
                    st.write(f"Found: {len(filtered)} neutral analyses")
            
            with filter_cols[3]:
                if st.button("📈 High Scores", use_container_width=True):
                    filtered = df[df['textblob_score'] > 0.5]
                    st.write(f"Found: {len(filtered)} high score analyses")
            
            with filter_cols[4]:
                if st.button("📉 Low Scores", use_container_width=True):
                    filtered = df[df['textblob_score'] < -0.5]
                    st.write(f"Found: {len(filtered)} low score analyses")

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
    
    # Calculate advanced metrics
    metrics = calculate_advanced_metrics(df)
    
    # Statistical Analysis Section
    st.markdown("### 📊 Statistical Analysis")
    
    stat_cols = st.columns(3)
    
    with stat_cols[0]:
        st.markdown("#### Central Tendency")
        st.metric("Mean Score", f"{metrics.get('tb_mean', 0):.3f}")
        st.metric("Median Score", f"{metrics.get('tb_median', 0):.3f}")
        st.metric("Mode Score", f"{df['textblob_score'].mode().iloc[0] if not df['textblob_score'].mode().empty else 0:.3f}")
    
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
    
    # Distribution Analysis
    st.markdown("### 📈 Distribution Analysis")
    
    dist_col1, dist_col2 = st.columns(2)
    
    with dist_col1:
        # Histogram with KDE
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        
        if st.session_state.chart_theme == "dark":
            plt.style.use('dark_background')
            hist_color = '#3B82F6'
            kde_color = '#EF4444'
            bg_color = '#1F2937'
            text_color = 'white'
        else:
            hist_color = '#60A5FA'
            kde_color = '#DC2626'
            bg_color = 'white'
            text_color = 'black'
        
        fig1.patch.set_facecolor(bg_color)
        ax1.set_facecolor(bg_color)
        
        # Plot histogram
        scores = df['textblob_score'].values
        ax1.hist(scores, bins=20, alpha=0.7, color=hist_color, 
                edgecolor='white', density=True, label='Histogram')
        
        # Plot KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(scores)
        x_range = np.linspace(min(scores), max(scores), 1000)
        ax1.plot(x_range, kde(x_range), color=kde_color, linewidth=3, label='KDE')
        
        # Add normal distribution for comparison
        from scipy.stats import norm
        mu, sigma = np.mean(scores), np.std(scores)
        normal_pdf = norm.pdf(x_range, mu, sigma)
        ax1.plot(x_range, normal_pdf, '--', color='green', alpha=0.7, linewidth=2, label='Normal Dist')
        
        ax1.set_title('Score Distribution with KDE', fontsize=14, fontweight='bold', color=text_color)
        ax1.set_xlabel('Sentiment Score', fontsize=12, color=text_color)
        ax1.set_ylabel('Density', fontsize=12, color=text_color)
        ax1.legend(facecolor=bg_color, edgecolor=text_color)
        ax1.grid(alpha=0.3, color=text_color)
        ax1.tick_params(colors=text_color)
        
        st.pyplot(fig1)
        plt.close(fig1)
    
    with dist_col2:
        # Box plot
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        if st.session_state.chart_theme == "dark":
            box_color = '#8B5CF6'
            median_color = '#10B981'
            bg_color = '#1F2937'
            text_color = 'white'
        else:
            box_color = '#7C3AED'
            median_color = '#059669'
            bg_color = 'white'
            text_color = 'black'
        
        fig2.patch.set_facecolor(bg_color)
        ax2.set_facecolor(bg_color)
        
        # Create box plot
        box_data = [df['textblob_score'].values]
        if nltk_available and 'vader_score' in df.columns:
            box_data.append(df['vader_score'].values)
        
        bp = ax2.boxplot(box_data, patch_artist=True, 
                        labels=['TextBlob', 'VADER'][:len(box_data)])
        
        # Style the box plot
        for box in bp['boxes']:
            box.set_facecolor(box_color)
            box.set_alpha(0.7)
        
        for median in bp['medians']:
            median.set_color(median_color)
            median.set_linewidth(2)
        
        for whisker in bp['whiskers']:
            whisker.set_color(text_color)
            whisker.set_linewidth(1.5)
        
        for cap in bp['caps']:
            cap.set_color(text_color)
            cap.set_linewidth(1.5)
        
        ax2.set_title('Score Distribution (Box Plot)', fontsize=14, fontweight='bold', color=text_color)
        ax2.set_ylabel('Sentiment Score', fontsize=12, color=text_color)
        ax2.grid(alpha=0.3, color=text_color, axis='y')
        ax2.tick_params(colors=text_color)
        
        st.pyplot(fig2)
        plt.close(fig2)
    
    # Correlation Analysis
    st.markdown("### 🔗 Correlation Analysis")
    
    corr_matrix = generate_correlation_matrix(df)
    
    if not corr_matrix.empty:
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        
        if st.session_state.chart_theme == "dark":
            cmap = 'coolwarm'
            bg_color = '#1F2937'
            text_color = 'white'
        else:
            cmap = 'RdYlBu'
            bg_color = 'white'
            text_color = 'black'
        
        fig3.patch.set_facecolor(bg_color)
        ax3.set_facecolor(bg_color)
        
        # Create heatmap
        im = ax3.imshow(corr_matrix.values, cmap=cmap, vmin=-1, vmax=1)
        
        # Add annotations
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                value = corr_matrix.iloc[i, j]
                color = 'white' if abs(value) > 0.5 else 'black'
                ax3.text(j, i, f'{value:.2f}', 
                        ha='center', va='center', 
                        color=color, fontweight='bold', fontsize=11)
        
        # Set labels
        ax3.set_xticks(np.arange(len(corr_matrix.columns)))
        ax3.set_yticks(np.arange(len(corr_matrix.columns)))
        ax3.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', color=text_color)
        ax3.set_yticklabels(corr_matrix.columns, color=text_color)
        
        ax3.set_title('Correlation Matrix', fontsize=16, fontweight='bold', 
                     color=text_color, pad=20)
        
        # Add colorbar
        cbar = ax3.figure.colorbar(im, ax=ax3)
        cbar.ax.tick_params(colors=text_color)
        cbar.set_label('Correlation Coefficient', color=text_color, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)
        
        # Interpretation
        st.info("""
        **Correlation Interpretation:**
        - **±0.9 to ±1.0**: Very strong relationship
        - **±0.7 to ±0.9**: Strong relationship
        - **±0.5 to ±0.7**: Moderate relationship
        - **±0.3 to ±0.5**: Weak relationship
        - **±0.0 to ±0.3**: Little to no relationship
        """)
    
    # Time Series Analysis (if enough data)
    if len(df) >= 5 and 'timestamp' in df.columns:
        st.markdown("### ⏳ Time Series Analysis")
        
        try:
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('datetime')
            
            # Resample by time frequency
            df.set_index('datetime', inplace=True)
            daily_avg = df['textblob_score'].resample('D').mean()
            
            fig4, ax4 = plt.subplots(figsize=(12, 5))
            
            if st.session_state.chart_theme == "dark":
                line_color = '#10B981'
                fill_color = '#10B98122'
                bg_color = '#1F2937'
                text_color = 'white'
            else:
                line_color = '#059669'
                fill_color = '#05966922'
                bg_color = 'white'
                text_color = 'black'
            
            fig4.patch.set_facecolor(bg_color)
            ax4.set_facecolor(bg_color)
            
            # Plot time series
            ax4.plot(daily_avg.index, daily_avg.values, 
                    color=line_color, linewidth=2.5, marker='o', markersize=6)
            
            # Fill between
            ax4.fill_between(daily_avg.index, 
                           daily_avg.values, 
                           alpha=0.2, color=fill_color)
            
            # Add trend line
            if len(daily_avg) > 1:
                z = np.polyfit(range(len(daily_avg)), daily_avg.values, 1)
                p = np.poly1d(z)
                ax4.plot(daily_avg.index, p(range(len(daily_avg))), 
                        '--', color='red', alpha=0.7, linewidth=2,
                        label=f'Trend (slope: {z[0]:.4f})')
            
            ax4.set_title('Daily Average Sentiment Trend', fontsize=14, 
                         fontweight='bold', color=text_color)
            ax4.set_xlabel('Date', fontsize=12, color=text_color)
            ax4.set_ylabel('Average Score', fontsize=12, color=text_color)
            ax4.legend(facecolor=bg_color, edgecolor=text_color)
            ax4.grid(alpha=0.3, color=text_color)
            ax4.tick_params(colors=text_color, rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close(fig4)
            
        except Exception as e:
            st.warning(f"Time series analysis not available: {str(e)}")

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
    
    # Filter only analyses with both methods
    comparison_df = df[(df['textblob_sentiment'] != 'N/A') & (df['vader_sentiment'] != 'N/A')]
    
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
    
    # Agreement visualization
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Set theme
    if st.session_state.chart_theme == "dark":
        plt.style.use('dark_background')
        agree_color = '#10B981'
        disagree_color = '#EF4444'
        bar_colors = ['#3B82F6', '#8B5CF6']
        bg_color = '#1F2937'
        text_color = 'white'
    else:
        agree_color = '#059669'
        disagree_color = '#DC2626'
        bar_colors = ['#2563EB', '#7C3AED']
        bg_color = 'white'
        text_color = 'black'
    
    fig1.patch.set_facecolor(bg_color)
    ax1.set_facecolor(bg_color)
    ax2.set_facecolor(bg_color)
    
    # Pie chart - Agreement distribution
    sizes = [agreements, disagreements]
    labels = ['Agree', 'Disagree']
    colors = [agree_color, disagree_color]
    explode = (0.05, 0.05)
    
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct='%1.1f%%', startangle=90, shadow=True,
           textprops={'color': text_color, 'fontweight': 'bold'})
    ax1.set_title('Method Agreement Distribution', fontsize=14, 
                 fontweight='bold', color=text_color)
    
    # Bar chart - Agreement by sentiment
    sentiment_agreement = {}
    for sentiment in ['positive', 'negative', 'neutral']:
        mask = (comparison_df['textblob_sentiment'] == sentiment) & \
               (comparison_df['vader_sentiment'] == sentiment)
        sentiment_agreement[sentiment] = mask.sum()
    
    sentiments = list(sentiment_agreement.keys())
    counts = list(sentiment_agreement.values())
    
    bars = ax2.bar(sentiments, counts, color=bar_colors[:len(sentiments)], 
                  edgecolor='white', linewidth=2)
    
    # Add value labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                str(count), ha='center', va='bottom',
                fontweight='bold', fontsize=12, color=text_color)
    
    ax2.set_title('Agreements by Sentiment', fontsize=14, 
                 fontweight='bold', color=text_color)
    ax2.set_xlabel('Sentiment', fontsize=12, color=text_color)
    ax2.set_ylabel('Agreement Count', fontsize=12, color=text_color)
    ax2.grid(axis='y', alpha=0.3, color=text_color)
    ax2.tick_params(colors=text_color)
    
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)
    
    # Detailed Disagreement Analysis
    st.markdown("### 🔍 Detailed Disagreement Analysis")
    
    # Get disagreements
    disagreements_df = comparison_df[comparison_df['textblob_sentiment'] != comparison_df['vader_sentiment']]
    
    if not disagreements_df.empty:
        # Categorize disagreement types
        disagreement_types = {}
        for _, row in disagreements_df.iterrows():
            key = f"{row['textblob_sentiment']}→{row['vader_sentiment']}"
            disagreement_types[key] = disagreement_types.get(key, 0) + 1
        
        # Display disagreement types
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Disagreement Patterns:**")
            for pattern, count in disagreement_types.items():
                tb, vader = pattern.split('→')
                st.write(f"• TextBlob: {tb.title()} → VADER: {vader.title()}: {count} cases")
        
        with col2:
            # Most common disagreement
            if disagreement_types:
                most_common = max(disagreement_types.items(), key=lambda x: x[1])
                st.metric("Most Common Disagreement", 
                         f"{most_common[0]}", 
                         f"{most_common[1]} cases")
        
        # Show sample disagreements
        with st.expander("📋 View Sample Disagreements", expanded=False):
            sample_df = disagreements_df[['id', 'tweet_short', 
                                         'textblob_sentiment', 'vader_sentiment',
                                         'textblob_score', 'vader_score']].head(10)
            
            # Format for display
            def format_disagreement(row):
                return f"**ID {row['id']}:** {row['tweet_short']}<br>" \
                       f"TextBlob: <span class='{get_sentiment_style(row['textblob_sentiment'])}'>" \
                       f"{row['textblob_sentiment'].title()} ({row['textblob_score']:.3f})</span> | " \
                       f"VADER: <span class='{get_sentiment_style(row['vader_sentiment'])}'>" \
                       f"{row['vader_sentiment'].title()} ({row['vader_score']:.3f})</span>"
            
            for _, row in sample_df.iterrows():
                st.markdown(format_disagreement(row), unsafe_allow_html=True)
    
    # Method Performance Comparison
    st.markdown("### ⚡ Method Performance Comparison")
    
    perf_cols = st.columns(3)
    
    with perf_cols[0]:
        # Score range comparison
        tb_range = comparison_df['textblob_score'].max() - comparison_df['textblob_score'].min()
        vader_range = comparison_df['vader_score'].max() - comparison_df['vader_score'].min()
        st.metric("Score Range", 
                 f"TB: {tb_range:.3f}", 
                 f"VADER: {vader_range:.3f}")
    
    with perf_cols[1]:
        # Standard deviation comparison
        tb_std = comparison_df['textblob_score'].std()
        vader_std = comparison_df['vader_score'].std()
        st.metric("Score Variability", 
                 f"TB: {tb_std:.3f}", 
                 f"VADER: {vader_std:.3f}")
    
    with perf_cols[2]:
        # Correlation
        correlation = comparison_df['textblob_score'].corr(comparison_df['vader_score'])
        st.metric("Score Correlation", 
                 f"{correlation:.3f}",
                 "Perfect = 1.0")
    
    # Recommendations
    st.markdown("### 💡 Recommendations")
    
    rec_col1, rec_col2 = st.columns(2)
    
    with rec_col1:
        st.info("""
        **When to use TextBlob:**
        • Longer, formal text
        • Articles, reviews, essays
        • When subjectivity matters
        • General sentiment analysis
        
        **TextBlob Strengths:**
        ✅ Better for longer text
        ✅ Provides subjectivity score
        ✅ More nuanced for formal language
        """)
    
    with rec_col2:
        st.info("""
        **When to use VADER:**
        • Social media content
        • Short texts, tweets
        • Text with emojis, slang
        • Real-time sentiment
        
        **VADER Strengths:**
        ✅ Optimized for social media
        ✅ Handles emojis and slang
        ✅ Better for short texts
        ✅ Faster processing
        """)
    
    # Interactive comparison tool
    st.markdown("### 🎯 Interactive Comparison Tool")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        sample_text = st.text_area(
            "Enter text to compare methods:",
            "I love this product! It's amazing! 😍",
            height=100,
            key="compare_input"
        )
        
        if st.button("Compare Methods", use_container_width=True):
            with st.spinner("Comparing..."):
                tb_result = analyze_sentiment_textblob(sample_text)
                vader_result = analyze_sentiment_vader(sample_text)
                
                st.success("Comparison Complete!")
                
                comp_col1, comp_col2 = st.columns(2)
                
                with comp_col1:
                    st.markdown(f"""
                    **TextBlob:**
                    - Score: {tb_result['score']:.3f}
                    - Sentiment: {tb_result['sentiment'].title()}
                    - Subjectivity: {tb_result.get('subjectivity', 0):.2f}
                    - Intensity: {tb_result.get('intensity', 'N/A')}
                    """)
                
                with comp_col2:
                    st.markdown(f"""
                    **VADER:**
                    - Score: {vader_result['score']:.3f}
                    - Sentiment: {vader_result['sentiment'].title()}
                    - Confidence: {vader_result.get('confidence', 0):.2f}
                    - Intensity: {vader_result.get('intensity', 'N/A')}
                    """)
                
                if tb_result['sentiment'] == vader_result['sentiment']:
                    st.success(f"✅ Methods agree: **{tb_result['sentiment'].title()}**")
                else:
                    st.warning(f"⚠️ Methods disagree: TextBlob={tb_result['sentiment'].title()}, VADER={vader_result['sentiment'].title()}")

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
        st.metric("Time Range", 
                 f"{df['timestamp'].min()[:10] if not df.empty else 'N/A'} to "
                 f"{df['timestamp'].max()[:10] if not df.empty else 'N/A'}")
    
    with summary_cols[2]:
        st.metric("File Size (approx)", 
                 f"{sys.getsizeof(df) / 1024:.1f} KB")
    
    with summary_cols[3]:
        st.metric("Unique Tweets", 
                 df['tweet'].nunique() if 'tweet' in df.columns else 0)
    
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
         "Excel (Microsoft Excel)", 
         "TSV (Tab Separated Values)"],
        horizontal=True
    )
    
    # Advanced Options
    with st.expander("⚡ Advanced Export Settings", expanded=False):
        col_opt1, col_opt2 = st.columns(2)
        
        with col_opt1:
            include_full_text = st.checkbox("Include Full Tweet Text", value=True,
                                          help="Include the complete tweet text in export")
        
        with col_opt2:
            export_all_columns = st.checkbox("Export All Columns", value=True,
                                           help="Export all available columns")
        
        date_range = st.date_input(
            "Filter by Date Range:",
            value=[pd.to_datetime(df['timestamp'].min()).date() if not df.empty else datetime.now().date(),
                   pd.to_datetime(df['timestamp'].max()).date() if not df.empty else datetime.now().date()],
            key="export_date_range"
        )
        
        sentiment_filter = st.multiselect(
            "Filter by Sentiment:",
            options=['positive', 'negative', 'neutral'],
            default=['positive', 'negative', 'neutral'],
            help="Select which sentiments to include in export"
        )
    
    # Prepare data for export
    export_df = df.copy()
    
    # Apply filters
    if len(date_range) == 2:
        start_date, end_date = date_range
        export_df['timestamp_dt'] = pd.to_datetime(export_df['timestamp'])
        export_df = export_df[(export_df['timestamp_dt'].dt.date >= start_date) & 
                             (export_df['timestamp_dt'].dt.date <= end_date)]
        export_df = export_df.drop('timestamp_dt', axis=1)
    
    if sentiment_filter and 'textblob_sentiment' in export_df.columns:
        export_df = export_df[export_df['textblob_sentiment'].isin(sentiment_filter)]
    
    if not export_all_columns:
        # Select essential columns
        essential_cols = ['id', 'timestamp', 'tweet_short', 
                         'textblob_score', 'textblob_sentiment']
        if 'vader_score' in export_df.columns:
            essential_cols.extend(['vader_score', 'vader_sentiment'])
        if include_full_text and 'tweet' in export_df.columns:
            essential_cols.append('tweet')
        
        export_df = export_df[essential_cols]
    elif not include_full_text and 'tweet' in export_df.columns:
        export_df = export_df.drop('tweet', axis=1)
    
    # Export Buttons
    st.markdown("### 📥 Download Data")
    
    if export_df.empty:
        st.error("No data matches your filters. Please adjust your filter settings.")
    else:
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
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        export_df.to_excel(writer, index=False, sheet_name='Sentiment Analysis')
                    
                    st.download_button(
                        label="📥 Download Excel",
                        data=buffer.getvalue(),
                        file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        help="Download data as Excel file (requires openpyxl)"
                    )
                except ImportError:
                    st.error("Excel export requires openpyxl. Install with: `pip install openpyxl`")
        
        # TSV Export (if selected)
        if "TSV" in export_format:
            tsv_data = export_df.to_csv(index=False, sep='\t')
            st.download_button(
                label="📥 Download TSV",
                data=tsv_data,
                file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tsv",
                mime="text/tab-separated-values",
                use_container_width=True,
                help="Download data as Tab-Separated Values file"
            )
    
    # Export Statistics
    st.markdown("### 📈 Export Statistics")
    
    stat_cols = st.columns(3)
    
    with stat_cols[0]:
        st.metric("Records to Export", len(export_df))
    
    with stat_cols[1]:
        if 'textblob_sentiment' in export_df.columns:
            sentiment_dist = export_df['textblob_sentiment'].value_counts()
            st.metric("Most Common Sentiment", 
                     sentiment_dist.index[0] if not sentiment_dist.empty else "N/A")
    
    with stat_cols[2]:
        if 'textblob_score' in export_df.columns:
            avg_score = export_df['textblob_score'].mean()
            st.metric("Average Score", f"{avg_score:.3f}")
    
    # Data Cleaning Tools
    st.markdown("### 🧹 Data Cleaning Tools")
    
    with st.expander("Data Cleaning Options", expanded=False):
        clean_col1, clean_col2 = st.columns(2)
        
        with clean_col1:
            if st.button("Remove Duplicate Tweets", use_container_width=True):
                initial_count = len(df)
                cleaned_df = df.drop_duplicates(subset=['tweet'], keep='first')
                removed = initial_count - len(cleaned_df)
                if removed > 0:
                    st.session_state.analysis_history = cleaned_df
                    st.success(f"Removed {removed} duplicate tweets!")
                    st.rerun()
                else:
                    st.info("No duplicate tweets found.")
        
        with clean_col2:
            if st.button("Remove Failed Analyses", use_container_width=True):
                initial_count = len(df)
                cleaned_df = df[~df['textblob_sentiment'].isin(['N/A', 'error'])]
                removed = initial_count - len(cleaned_df)
                if removed > 0:
                    st.session_state.analysis_history = cleaned_df
                    st.success(f"Removed {removed} failed analyses!")
                    st.rerun()
                else:
                    st.info("No failed analyses found.")

# ============================================
# MAIN APP FUNCTION
# ============================================

def main():
    """Main application function"""
    
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
