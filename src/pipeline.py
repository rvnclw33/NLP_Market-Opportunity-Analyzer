# src/pipeline.py
"""
Utility to run the pipeline end-to-end locally:
python src/pipeline.py
"""
from subprocess import run

def main():
    steps = [
        "python src/preprocessing.py",
        "python src/topic_modeling.py",
        "python src/sentiment_models.py",
        "python src/scoring.py"
    ]
    for s in steps:
        print("RUN:", s)
        r = run(s, shell=True)
        if r.returncode != 0:
            print("Step failed:", s)
            break

if __name__ == "__main__":
    main()