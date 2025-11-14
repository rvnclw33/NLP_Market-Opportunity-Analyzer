# src/sentiment_models.py
import pandas as pd
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

IN_DOCS = Path("outputs/docs_with_topics.csv")
OUT_SENT = Path("outputs/docs_with_sentiment.csv")

def main():
    print("Loading docs with topics:", IN_DOCS)
    df = pd.read_csv(IN_DOCS, low_memory=False, parse_dates=['published_at'])
    analyzer = SentimentIntensityAnalyzer()
    print("Computing VADER compound sentiment for each doc...")
    df['vader_compound'] = df['text_clean'].astype(str).map(lambda t: analyzer.polarity_scores(t)['compound'])
    df['sent_label'] = df['vader_compound'].apply(lambda x: 'neg' if x < -0.05 else ('pos' if x > 0.05 else 'neu'))
    df.to_csv(OUT_SENT, index=False)
    print("Saved", OUT_SENT)

if __name__ == "__main__":
    main()
