import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path
import os
from openai import OpenAI

# ML Models
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from collections import Counter

# -----------------------------
# CONFIGURATION
# -----------------------------
st.set_page_config(page_title="Market Opportunity Analyzer (Demo)", page_icon="📊", layout="wide")

# Model Names
SBERT_MODEL = "all-MiniLM-L6-v2"

# -----------------------------
# OPENAI CLIENT INITIALIZATION
# -----------------------------
openai_api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
client = None
if openai_api_key:
    client = OpenAI(api_key=openai_api_key)
else:
    st.info("⚠️ No OPENAI_API_KEY found. Set it in Streamlit secrets to use the 'Ask Questions' feature.")


# -----------------------------
# MODEL CACHING (VERY IMPORTANT)
# -----------------------------

# --- THIS FUNCTION IS NOW FIXED ---
# The st.toast() line has been REMOVED.
@st.cache_resource
def load_sbert_model():
    """Loads the Sentence Transformer model once and caches it."""
    # st.toast("Loading SBERT model (this happens once)...") # <-- THIS WAS THE BUG
    return SentenceTransformer(SBERT_MODEL)
# --- END OF FIX ---

@st.cache_resource
def get_vader_analyzer():
    """Loads the VADER analyzer once."""
    return SentimentIntensityAnalyzer()

# -----------------------------
# HELPER FUNCTIONS (PURE)
# -----------------------------
def extract_text_column(df: pd.DataFrame) -> pd.DataFrame:
    """Finds the best text column and creates 'text_clean'."""
    text_cols = ["text", "review", "content", "reviewText", "comment", "TagLine", "description", "body", "verified_reviews"] 
    title_cols = ["title", "Summary", "ProductName"]
    
    text_col = next((col for col in text_cols if col in df.columns), None)
    title_col = next((col for col in title_cols if col in df.columns), None)
    
    if title_col and text_col:
        df["text_clean"] = (df[title_col].astype(str).fillna('') + ". " + df[text_col].astype(str).fillna('')).map(clean_text)
    elif text_col:
        df["text_clean"] = df[text_col].astype(str).map(clean_text)
    elif title_col:
        df["text_clean"] = df[title_col].astype(str).map(clean_text)
    else:
        text_data = df.select_dtypes(include="object").astype(str)
        df["text_clean"] = text_data.apply(lambda x: " ".join(x), axis=1).map(clean_text)
    
    df = df[df["text_clean"].str.len() > 10].reset_index(drop=True)
    return df


def clean_text(t):
    if pd.isna(t):
        return ""
    t = str(t)
    # remove urls, collapse whitespace
    t = re.sub(r"http\S+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# -----------------------------
# CORE ML PIPELINE (CACHEABLE & PURE)
# -----------------------------
@st.cache_data(show_spinner="Running Topic Modeling (SBERT, UMAP, HDBSCAN)...")
def run_topic_modeling(_df):
    sbert_model = load_sbert_model() # This call is now 100% pure
    docs = _df['text_clean'].tolist()
    
    if len(docs) < 10:
        return None, None # Return error signal

    # --- 1. SBERT ---
    embeddings = sbert_model.encode(docs, show_progress_bar=False, convert_to_numpy=True)
    
    # --- 2. UMAP ---
    n_neighbors = min(15, len(docs) - 1)
    n_components = min(5, len(docs) - 1)
    
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, metric='cosine', random_state=42)
    embs_reduced = reducer.fit_transform(embeddings)

    # --- 3. HDBSCAN ---
    min_cluster_size = max(5, len(docs) // 20) 
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', gen_min_span_tree=True)
    labels = clusterer.fit_predict(embs_reduced)
    
    _df['topic_id'] = labels
    
    # --- 4. Topic Summarization ---
    groups = []
    for c in sorted(_df.topic_id.unique()):
        group_df = _df[_df.topic_id == c]
        texts = group_df['text_clean'].tolist()
        
        try:
            vec = TfidfVectorizer(ngram_range=(1,2), max_features=2000, stop_words='english')
            X = vec.fit_transform(texts)
            sums = np.asarray(X.sum(axis=0)).ravel()
            idx = np.argsort(sums)[::-1][:8]
            top_terms = [vec.get_feature_names_out()[i] for i in idx]
        except Exception:
            top_terms = ["-"] 
        
        examples = group_df.sample(min(3, len(group_df)))['text_clean'].tolist()
        
        groups.append({
            'topic_id': int(c),
            'topic_name': ", ".join(top_terms) if c != -1 else "Noise / Other",
            'mentions': int(len(group_df)),
            'examples': examples
        })
        
    topics_df = pd.DataFrame(groups).sort_values('mentions', ascending=False)
    
    return _df, topics_df

@st.cache_data(show_spinner="Running Sentiment Analysis (VADER)...")
def run_sentiment_analysis(_df):
    analyzer = get_vader_analyzer()
    _df['vader_compound'] = _df['text_clean'].astype(str).map(lambda t: analyzer.polarity_scores(t)['compound'])
    _df['sent_label'] = _df['vader_compound'].apply(lambda x: 'neg' if x < -0.05 else ('pos' if x > 0.05 else 'neu'))
    return _df

@st.cache_data(show_spinner="Calculating Opportunity Scores...")
def run_scoring(_df, _topics_df):
    if _df.empty or 'topic_id' not in _df.columns:
        return _topics_df
    
    stats = _df.groupby('topic_id').agg(
        mean_sentiment=('vader_compound','mean'),
        neg_fraction=('sent_label', lambda s: (s=='neg').mean())
    ).reset_index()

    _topics_df = _topics_df.merge(stats, on='topic_id', how='left')
    _topics_df['raw_opportunity'] = _topics_df['mentions'] * _topics_df['neg_fraction']
    
    if not _topics_df.empty and _topics_df['raw_opportunity'].max() > 0:
        _topics_df['opportunity_score'] = (_topics_df['raw_opportunity'] / _topics_df['raw_opportunity'].max()) * 100
    else:
        _topics_df['opportunity_score'] = 0
        
    return _topics_df.sort_values('opportunity_score', ascending=False)

# -----------------------------
# NEW FEATURE: AI CHAT FUNCTION (PURE)
# -----------------------------
@st.cache_data(show_spinner="AI is thinking...")
def generate_answer(query, data_summary):
    """Generates an answer from the AI based on the data."""
    if not client:
        return "OpenAI client is not configured. Cannot ask questions."
    
    system_prompt = """
    You are a world-class market analyst. A user has just run an analysis on their customer reviews.
    This analysis has produced a data summary showing topics, mention counts, negative sentiment fractions, and opportunity scores.
    Your job is to answer the user's questions based *only* on the data summary provided.
    Be concise, insightful, and use a professional tone.
    If the answer isn't in the data, say so. Do not make up information.
    """

    user_message = f"""
    Here is the data summary of the analysis:
    {data_summary}

    Here is my question:
    {query}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.5,
    )
    return response.choices[0].message.content

# -----------------------------
# UI - MAIN PAGE
# -----------------------------
st.title("📊 Market Opportunity Analyzer (Live Demo)")
st.markdown("This app uses SBERT, UMAP, HDBSCAN, and VADER to find market opportunities from customer reviews.")

st.subheader("1. Upload Your Customer Review Data")
st.markdown("Upload a small CSV or TSV (e.g., 500-1000 rows) for a fast, live demo.")

uploaded_file = st.file_uploader("Choose a CSV or TSV file", type=["csv", "tsv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding='latin1')
    except pd.errors.ParserError as e:
        st.warning(f"CSV read failed. Trying to read as a Tab-Separated File (TSV)...")
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep='\t', encoding='latin1')
            st.success("Successfully loaded file as a TSV.")
        except Exception as e2:
            st.error(f"Error reading file as CSV or TSV: {e2}")
            st.stop()
    except Exception as e:
        st.error(f"An unknown error occurred while reading the file: {e}")
        st.stop()
    
    st.success(f"Loaded **{uploaded_file.name}** with **{len(df)}** rows.")
    
    # -----------------------------
    # 2. RUN THE FULL PIPELINE
    # -----------------------------
    df_clean = extract_text_column(df)
    
    if df_clean.empty:
        st.error("Could not find any usable text in the CSV. Please check the file.")
        st.stop()
    st.write(f"Found {len(df_clean)} usable reviews.")

    # This call is now 100% safe and will not error.
    df_with_topics, topics_df = run_topic_modeling(df_clean)
    
    if df_with_topics is None:
        st.error("Not enough data for topic modeling. Need at least 10 reviews.")
        st.stop()
    
    df_with_sentiment = run_sentiment_analysis(df_with_topics)
    final_scores = run_scoring(df_with_sentiment, topics_df)
    
    # -----------------------------
    # 3. DISPLAY RESULTS
    # -----------------------------
    st.subheader("💡 Opportunity Dashboard")
    st.markdown("These are the topics with the highest opportunity, based on mention count and negative sentiment.")
    
    st.dataframe(
        final_scores,
        column_config={
            "opportunity_score": st.column_config.ProgressColumn("Opportunity Score", format="%.1f"),
            "neg_fraction": st.column_config.ProgressColumn("Negative Fraction", format="%.2f"),
            "topic_name": st.column_config.TextColumn("Topic Keywords", width="medium"),
            "mentions": st.column_config.NumberColumn("Mentions", format="%d"),
            "examples": st.column_config.ListColumn("Examples", width="large"),
        },
        use_container_width=True
    )

    st.subheader("🔍 Explore All Documents")
    st.markdown("Inspect all the documents with their assigned topics and sentiment.")
    st.dataframe(df_with_sentiment[['text_clean', 'topic_id', 'sent_label', 'vader_compound']], use_container_width=True)

    # -----------------------------
    # 4. ASK QUESTIONS (NEW FEATURE)
    # -----------------------------
    st.subheader("💬 4. Ask Questions About Your Data")
    
    if client: 
        data_summary_string = final_scores.to_string()
        
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if query := st.chat_input("E.g., What's the biggest pain point? Which topic has the worst sentiment?"):
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                try:
                    response = generate_answer(query, data_summary_string)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error connecting to OpenAI: {e}")
            
    else:
        st.warning("Please set your `OPENAI_API_KEY` in Streamlit secrets to enable this feature.")

else:
    st.info("Please upload a CSV or TSV file to begin the analysis.")

st.caption("This interactive app runs ML pipeline (SBERT, UMAP, HDBSCAN, VADER) live on user-uploaded data.")