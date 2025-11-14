# src/preprocessing.py
"""
Preprocessing pipeline (robust + chunked sentence splitting)

Stage A: read raw CSVs -> normalize -> clean -> dedupe -> save cleaned CSV (temp)
Stage B: read cleaned CSV in chunks -> spaCy nlp.pipe sentence-splitting -> append to final CSV

This avoids holding all spaCy outputs in memory for very large datasets.
"""
import os
import re
import glob
import hashlib
import sys
from pathlib import Path

import pandas as pd
import spacy

# -------------------- config --------------------
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CLEANED_PATH = PROCESSED_DIR / "processed_reviews_cleaned.csv"   # interim file
OUT_PATH = PROCESSED_DIR / "processed_reviews.csv"               # final file with sentences

# spaCy model
SPACY_MODEL = "en_core_web_sm"

# chunking / spaCy settings (tune for your machine)
CHUNK_ROWS = 25000          # rows per chunk when doing spaCy pass (reduce if memory is tight)
SPACY_BATCH_SIZE = 1000     # docs per spaCy batch in nlp.pipe
DEFAULT_N_PROCESS = 1       # default parallel processes for nlp.pipe (we override for macOS)

# sentence limit per doc
MAX_SENTENCES = 5

# -------------------- load spaCy --------------------
print("[info] Loading spaCy model (this may take a moment)...")
nlp = spacy.load(SPACY_MODEL, disable=["ner"])
print("[info] spaCy loaded:", SPACY_MODEL)

# -------------------- helper functions --------------------
def read_csvs(raw_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(raw_dir / "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSVs found in {raw_dir}. Put your Kaggle/raw files there.")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            df["__source_file"] = Path(f).name
            dfs.append(df)
            print(f"[load] {Path(f).name} → {len(df):,} rows")
        except Exception as e:
            print(f"[skip] {Path(f).name}: {e}")
    if not dfs:
        raise FileNotFoundError("No usable CSVs found in data/raw.")
    return pd.concat(dfs, ignore_index=True, sort=False)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {}

    # Amazon
    if "Text" in df.columns and "text" not in df.columns:
        colmap["Text"] = "text"
    if "Summary" in df.columns and "title" not in df.columns:
        colmap["Summary"] = "title"
    if "Time" in df.columns and "published_at" not in df.columns:
        colmap["Time"] = "published_at"

    # Product Hunt
    if "TagLine" in df.columns and "text" not in df.columns:
        colmap["TagLine"] = "text"
    if "ProductName" in df.columns and "title" not in df.columns:
        colmap["ProductName"] = "title"
    if "Date" in df.columns and "published_at" not in df.columns:
        colmap["Date"] = "published_at"
    if "ShortUrl" in df.columns and "url" not in df.columns:
        colmap["ShortUrl"] = "url"

    # Google Play
    if "App" in df.columns and "title" not in df.columns:
        colmap["App"] = "title"
    if "Last Updated" in df.columns and "published_at" not in df.columns:
        colmap["Last Updated"] = "published_at"

    # generic fallbacks
    if "review" in df.columns and "text" not in df.columns:
        colmap["review"] = "text"
    if "content" in df.columns and "text" not in df.columns:
        colmap["content"] = "text"
    if "date" in df.columns and "published_at" not in df.columns:
        colmap["date"] = "published_at"

    if colmap:
        df = df.rename(columns=colmap)

    for c in ["source", "title", "text", "published_at", "url"]:
        if c not in df.columns:
            df[c] = ""

    # keep only canonical columns + source filename
    keep = [c for c in ["source", "title", "text", "published_at", "url", "__source_file"] if c in df.columns]
    return df[keep]


def clean_text(t):
    if pd.isna(t):
        return ""
    t = str(t)
    # remove urls, collapse whitespace
    t = re.sub(r"http\S+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def dedupe_df(df: pd.DataFrame) -> pd.DataFrame:
    def row_hash(r):
        text = (str(r.get("title", "")) + " " + str(r.get("text", ""))).strip().lower()
        return hashlib.sha1(text.encode("utf-8")).hexdigest()
    df["__hash"] = df.apply(row_hash, axis=1)
    before = len(df)
    df = df.drop_duplicates("__hash").reset_index(drop=True)
    print(f"[dedupe] {before:,} → {len(df):,}")
    return df


# -------------------- Stage A: read, normalize, clean, dedupe, save cleaned CSV --------------------
def stage_a_build_cleaned():
    print("[stage A] Reading raw CSVs and building cleaned interim file...")
    df = read_csvs(RAW_DIR)
    df = normalize_columns(df)

    # combine candidate text columns dynamically (title preferred)
    candidates = [c for c in ["title", "review", "text", "body", "description", "content"] if c in df.columns]
    if not candidates:
        print("[warn] No text-like columns detected. Exiting stage A.")
        df = df.assign(text_clean="")
        df.to_csv(CLEANED_PATH, index=False)
        return

    # build text_clean, clean, filter short entries
    df["text_clean"] = df[candidates].fillna("").agg(". ".join, axis=1)
    df["text_clean"] = df["text_clean"].map(clean_text)
    # drop very short or empty texts
    before = len(df)
    df = df[df["text_clean"].str.len() > 10].copy()
    print(f"[clean] filtered short/empty: {before:,} -> {len(df):,}")

    # dedupe
    df = dedupe_df(df)

    # parse dates if available
    try:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    except Exception:
        df["published_at"] = None

    # Save interim cleaned file (no sentences yet)
    df.to_csv(CLEANED_PATH, index=False)
    print(f"[stage A] Cleaned interim file saved to: {CLEANED_PATH} ({len(df):,} rows)")


# -------------------- Stage B: chunked spaCy sentence splitting and build final CSV --------------------
def stage_b_add_sentences(chunk_rows=CHUNK_ROWS, batch_size=SPACY_BATCH_SIZE):
    if not CLEANED_PATH.exists():
        raise FileNotFoundError(f"Cleaned file not found: {CLEANED_PATH}. Run stage A first.")
    # choose n_process conservatively for macOS
    if sys.platform == "darwin":
        n_process = 1
    else:
        n_process = DEFAULT_N_PROCESS

    print(f"[stage B] Reading cleaned file in chunks (chunk_rows={chunk_rows})")
    reader = pd.read_csv(CLEANED_PATH, chunksize=chunk_rows, low_memory=False)
    first_chunk = True
    total_written = 0
    for ci, chunk in enumerate(reader):
        texts = chunk["text_clean"].astype(str).tolist()
        sent_lists = []
        # run spaCy nlp.pipe in streaming mode
        print(f"[stage B] chunk {ci+1}: processing {len(texts):,} docs (batch_size={batch_size}, n_process={n_process})")
        for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
            sent_lists.append([s.text.strip() for s in doc.sents][:MAX_SENTENCES])
        chunk["sentences"] = sent_lists

        # write chunk to final CSV (append after first)
        if first_chunk:
            chunk.to_csv(OUT_PATH, index=False)
            first_chunk = False
        else:
            chunk.to_csv(OUT_PATH, index=False, header=False, mode="a")
        total_written += len(chunk)
        print(f"[stage B] wrote chunk {ci+1} -> {OUT_PATH} (total written: {total_written:,})")
    print(f"[stage B] Completed. Final file: {OUT_PATH} ({total_written:,} rows)")

# -------------------- main --------------------
def main():
    # Stage A
    stage_a_build_cleaned()

    # fast exit if cleaned file empty
    try:
        tmp = pd.read_csv(CLEANED_PATH, nrows=2, low_memory=False)
        if tmp.shape[0] == 0:
            print("[info] Cleaned file contains 0 rows; skipping sentence splitting.")
            # still write final empty file
            tmp.to_csv(OUT_PATH, index=False)
            return
    except Exception:
        pass

    # Stage B
    stage_b_add_sentences()

if __name__ == "__main__":
    main()
