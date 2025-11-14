# src/scoring.py
import pandas as pd
from pathlib import Path
OUT = Path("outputs/opportunity_scores.csv")
IN = Path("outputs/docs_with_sentiment.csv")

def aggregate_topic_scores(df):
    # compute mentions, negative fraction, mean sentiment
    stats = df.groupby('topic_id').agg(
        mentions=('text_clean','count'),
        mean_sentiment=('vader_compound','mean'),
        neg_fraction=('sent_label', lambda s: (s=='neg').mean())
    ).reset_index()
    # simple opportunity: mentions * neg_fraction
    stats['raw_opportunity'] = stats['mentions'] * stats['neg_fraction']
    # normalize 0-100
    if stats['raw_opportunity'].max() > 0:
        stats['opportunity_score'] = (stats['raw_opportunity'] / stats['raw_opportunity'].max()) * 100
    else:
        stats['opportunity_score'] = 0
    return stats

def main():
    print("Loading docs with sentiment:", IN)
    df = pd.read_csv(IN, low_memory=False)
    stats = aggregate_topic_scores(df)
    # join top_terms and examples from topics_summary if available
    try:
        topics = pd.read_csv("outputs/topics_summary.csv", low_memory=False)
        topics = topics[['topic_id','top_terms','examples']]
        # top_terms/examples likely stored as strings: keep as-is
        stats = stats.merge(topics, on='topic_id', how='left')
    except:
        print("topics_summary.csv not found; continuing without top_terms")
    stats = stats.sort_values('opportunity_score', ascending=False)
    stats.to_csv(OUT, index=False)
    print("Saved", OUT)

if __name__ == "__main__":
    main()