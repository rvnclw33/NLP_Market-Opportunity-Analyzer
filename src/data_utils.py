# src/data_utils.py
import os
import pandas as pd
import streamlit as st # Keep streamlit for caching

DATA_DIR = "data"

@st.cache_data
def list_domains():
    """Lists all CSV files in the data directory."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    return [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

@st.cache_data
def load_data(domain_file: str) -> pd.DataFrame:
    """Loads a CSV file from the data directory."""
    data_path = os.path.join(DATA_DIR, domain_file)
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return pd.DataFrame()