# Tweet Sentiment Analytics Dashboard

A real-time sentiment analysis dashboard for tweets using Streamlit, TextBlob, and VADER.

![Dashboard Preview](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Processing-blue)

## 🚀 Live Demo
Run the dashboard locally with Streamlit.

## 📋 Features
- **Real-time Analysis**: Analyze tweets instantly
- **Multiple Models**: TextBlob and VADER sentiment analysis
- **Data Visualization**: Interactive charts and graphs
- **Data Export**: Export results as CSV/JSON
- **History Tracking**: Store and compare past analyses

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/newpath-tech/tweet-sentiment-simple.git
cd tweet-sentiment-simple
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Prepare TweetClaw Exports

Use `scripts/prepare_tweetclaw_text.py` to convert reviewed TweetClaw JSON,
JSONL, or CSV exports into a simple CSV with a `tweet` column for this
dashboard. It can also write a newline-separated text file for quick paste
tests.

```bash
python scripts/prepare_tweetclaw_text.py examples/tweetclaw_export.jsonl \
  --output data/tweets.csv \
  --text-output data/tweets.txt
```

Review exports before conversion and do not commit private account data,
credentials, or non-public tweets. TweetClaw is available from
https://github.com/Xquik-dev/tweetclaw.
