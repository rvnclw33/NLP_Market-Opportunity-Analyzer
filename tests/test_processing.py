# tests/test_preprocessing.py
import pandas as pd
import numpy as np
import pytest
from pandas.testing import assert_frame_equal

# Import the functions to be tested
# We need to make sure Python can find the 'src' directory
import sys
from pathlib import Path

# Add the project root (one level up from 'tests') to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from src.preprocessing import clean_text, normalize_columns
except ImportError:
    pytest.skip("Could not import from 'src'. Make sure 'src' is in the Python path.", allow_module_level=True)


# --- Tests for clean_text ---

def test_clean_text_removes_urls():
    text = "This is great http://t.co/12345"
    assert clean_text(text) == "this is great"

def test_clean_text_collapses_whitespace():
    text = "Lots   of \n extra   spaces"
    assert clean_text(text) == "lots of extra spaces"

def test_clean_text_handles_null():
    assert clean_text(None) == ""
    assert clean_text(np.nan) == ""

def test_clean_text_integration():
    text = "Check this out! <br> http://example.com \n\n It's AMAZING.  "
    # Note: Your current clean_text function does not remove HTML,
    # it only removes URLs and collapses whitespace.
    # If it *should* remove HTML, you'd add: text = re.sub(r"<br\s*/?>", " ", str(text))
    # Based on your src/preprocessing.py, it *doesn't* have that.
    # The clean_text in src/preprocessing.py ONLY removes URLs and collapses whitespace.
    
    # Let's assume the clean_text from your script:
    # def clean_text(t):
    #     if pd.isna(t): return ""
    #     t = str(t)
    #     t = re.sub(r"http\S+", "", t)
    #     t = re.sub(r"\s+", " ", t).strip()
    #     return t

    # Based on *that* function, this is the expected output:
    assert clean_text(text) == "check this out! <br> it's amazing."


# --- Fixtures for normalize_columns ---
# Fixtures are reusable setup functions for tests

@pytest.fixture
def amazon_style_data():
    """Returns a DataFrame with Amazon-style column names."""
    return pd.DataFrame({
        "Text": ["Good", "Bad"],
        "Summary": ["Review title", "Another title"],
        "Time": ["2024-01-01", "2024-01-02"]
    })

@pytest.fixture
def product_hunt_style_data():
    """Returns a DataFrame with Product Hunt-style column names."""
    return pd.DataFrame({
        "TagLine": ["A new app", "A social network"],
        "ProductName": ["AppA", "AppB"],
        "Date": ["2024-03-01", "2024-03-02"],
        "ShortUrl": ["http://url.a", "http://url.b"]
    })

@pytest.fixture
def generic_style_data():
    """Returns a DataFrame with generic fallback column names."""
    return pd.DataFrame({
        "review": ["nice", "terrible"],
        "date": ["2024-04-01", "2024-04-02"],
        "id": [1, 2]
    })

@pytest.fixture
def perfect_style_data():
    """Returns a DataFrame with already-correct column names."""
    return pd.DataFrame({
        "text": ["A", "B"],
        "title": ["T1", "T2"],
        "published_at": ["2024-05-01", "2024-05-02"],
        "url": ["http://c", "http://d"],
        "source": ["web", "mobile"]
    })


# --- Tests for normalize_columns ---

def test_normalize_columns_amazon(amazon_style_data):
    df = normalize_columns(amazon_style_data)
    expected_cols = ["source", "title", "text", "published_at", "url", "__source_file"]
    
    assert "text" in df.columns
    assert "title" in df.columns
    assert "published_at" in df.columns
    assert "Text" not in df.columns  # Original column should be gone
    
    # Check that all canonical columns exist
    assert all(col in df.columns for col in expected_cols)
    assert df.loc[0, "text"] == "Good"

def test_normalize_columns_product_hunt(product_hunt_style_data):
    df = normalize_columns(product_hunt_style_data)
    expected_cols = ["source", "title", "text", "published_at", "url", "__source_file"]
    
    assert "text" in df.columns
    assert "title" in df.columns
    assert "published_at" in df.columns
    assert "url" in df.columns
    assert "TagLine" not in df.columns # Original column should be gone
    
    # Check that all canonical columns exist
    assert all(col in df.columns for col in expected_cols)
    assert df.loc[0, "text"] == "A new app"

def test_normalize_columns_generic(generic_style_data):
    df = normalize_columns(generic_style_data)
    expected_cols = ["source", "title", "text", "published_at", "url", "__source_file"]

    assert "text" in df.columns
    assert "published_at" in df.columns
    assert "review" not in df.columns # Original column should be gone
    
    # Check that all canonical columns exist
    assert all(col in df.columns for col in expected_cols)
    assert df.loc[0, "text"] == "nice"

def test_normalize_columns_perfect(perfect_style_data):
    df_original = perfect_style_data.copy()
    df_normalized = normalize_columns(df_original)
    
    # Use pandas testing utility to assert frames are identical
    assert_frame_equal(df_original, df_normalized)

def test_normalize_columns_adds_missing_cols(amazon_style_data):
    df = normalize_columns(amazon_style_data)
    
    # It renamed 'Text', 'Summary', 'Time', but 'source' and 'url' were missing
    # The function should add them as empty columns.
    assert "source" in df.columns
    assert "url" in df.columns
    assert df["source"].iloc[0] == ""
    assert df["url"].iloc[0] == ""