import pandas as pd
import numpy as np
import json
import os
import time
import glob
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
INPUT_DATA_DIR = "lesswrong_full_posts_intent_filtered"
OUTPUT_FILE = "classification_comparison_results.csv"
CHUNK_SIZE = 250
WORD_THRESHOLD = 400

def log_with_time(message):
    """Helper to log with timestamp"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

# ---
# STAGE 1: UNIFIED DATA LOADING AND SEGMENTATION
# ---
def load_and_segment_data():
    """Loads all post/comment data and segments it into a standard DataFrame."""
    log_with_time("--- Stage 1: Loading and Segmenting Data ---")
    all_files = glob.glob(os.path.join(INPUT_DATA_DIR, "*.json"))
    if not all_files:
        log_with_time(f"Error: No JSON files found in '{INPUT_DATA_DIR}'. Halting.")
        return None
    log_with_time(f"Found {len(all_files)} files to process.")

    processed_segments = []
    for file_path in all_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
            
            post_id = data.get("post", {}).get("id", os.path.basename(file_path).replace(".json", ""))
            post_content = data.get("post", {}).get("content", "")
            post_date = data.get("post", {}).get("postedAt", "")

            # Process post content
            if post_content:
                segments = [post_content] if len(post_content.split()) <= WORD_THRESHOLD else [' '.join(post_content.split()[i:i+CHUNK_SIZE]) for i in range(0, len(post_content.split()), CHUNK_SIZE)]
                for i, segment in enumerate(segments):
                    processed_segments.append({"segment_id": f"post_{post_id}_seg_{i}", "text": segment, "date": post_date, "source_id": post_id, "type": "post"})

            # Process comments
            for i, comment in enumerate(data.get("comments", [])):
                comment_content = comment.get("content", "")
                if comment_content:
                    processed_segments.append({"segment_id": f"comment_{post_id}_{i}", "text": comment_content, "date": comment.get("postedAt", ""), "source_id": comment.get("id", ""), "type": "comment"})
        except Exception as e:
            log_with_time(f"Error processing file {file_path}: {e}")
            
    df = pd.DataFrame(processed_segments)
    log_with_time(f"Total segments extracted: {len(df)}")
    return df

# ---
# STAGE 2: HEURISTIC KEYWORD CLASSIFIER
# ---
def run_heuristic_classifier(df):
    """Applies a keyword-based heuristic classifier to the DataFrame."""
    log_with_time("--- Stage 2: Running Heuristic Keyword Classifier ---")
    
    abstract_keywords = {'AGI', 'superintelligence', 'alignment', 'x-risk', 'p(doom)', 'instrumental convergence', 'orthogonality', 'yudkowsky', 'bostrom', 'pivotal act', 'paperclip'}
    tangible_keywords = {'LLM', 'GPT-4', 'ChatGPT', 'jailbreak', 'fine-tuning', 'prompt injection', 'misuse', 'bioweapon', 'cybersecurity', 'regulation', 'policy'}

    results = []
    for text in df['text']:
        text_lower = text.lower()
        abs_score = sum(1 for word in abstract_keywords if word in text_lower)
        tan_score = sum(1 for word in tangible_keywords if word in text_lower)
        
        if abs_score > tan_score: results.append([1, 0])
        elif tan_score > abs_score: results.append([0, 1])
        else: results.append([1, 1]) # Default to "both" if scores are tied
            
    result_df = pd.DataFrame(results, columns=['heuristic_abstract', 'heuristic_tangible'])
    df_combined = pd.concat([df, result_df], axis=1)
    log_with_time("Heuristic classification complete.")
    return df_combined

# ---
# STAGE 3: LOCAL POWERFUL MODEL CLASSIFIER (FLAN-T5)
# ---
def run_flan_classifier(df):
    """Applies the Flan-T5 based classifier to the DataFrame."""
    log_with_time("--- Stage 3: Running Local Powerful Model (FLAN-T5) Classifier ---")
    
    try:
        from transformers import pipeline
        log_with_time("Attempting to load 'google/flan-t5-large' model...")
        classifier = pipeline("text2text-generation", model="google/flan-t5-large", device=-1)
        log_with_time("✓ Flan-T5 model loaded successfully.")
    except Exception as e:
        log_with_time(f"Could not load Flan-T5 model: {e}. Skipping this stage.")
        df[['flan_abstract', 'flan_tangible', 'flan_confidence']] = -1, -1, -1.0
        return df

    def create_prompt(content):
        return f"""Task: Classify AI safety text as ABSTRACT or TANGIBLE risk.
ABSTRACT = theoretical, future, speculative (nanobots, paperclip maximizer, AGI timelines, superintelligence, alignment theory)
TANGIBLE = current, practical, immediate (ChatGPT misuse, regulation, China competition, terrorist use, jailbreaking)
Text: {content[:1500]}
Classification:"""

    results = []
    for i, text in enumerate(df['text']):
        prompt = create_prompt(text)
        try:
            response = classifier(prompt, max_new_tokens=10)[0]['generated_text'].replace(prompt, "").strip().lower()
            if "tangible" in response: results.append([0, 1, 0.8])
            elif "abstract" in response: results.append([1, 0, 0.8])
            else: results.append([1, 1, 0.5]) # Ambiguous response
        except Exception as e:
            log_with_time(f"  Error during Flan classification for segment {i}: {e}")
            results.append([-1, -1, -1.0])
        
        if (i + 1) % 50 == 0:
            log_with_time(f"  Processed {i+1}/{len(df)} segments with Flan-T5...")

    result_df = pd.DataFrame(results, columns=['flan_abstract', 'flan_tangible', 'flan_confidence'])
    df_combined = pd.concat([df, result_df], axis=1)
    log_with_time("Flan-T5 classification complete.")
    return df_combined

# ---
# STAGE 4: SUPERVISED ML MODEL (TRAINED ON HEURISTICS)
# ---
def run_supervised_ml_classifier(df):
    """Attempts to train and apply a simple supervised ML model."""
    log_with_time("--- Stage 4: Attempting Supervised ML Classifier (trained on heuristics) ---")

    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.linear_model import LogisticRegression
        
        log_with_time("Attempting to load SentenceTransformer model...")
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        log_with_time("✓ SentenceTransformer loaded.")

        log_with_time("Creating embeddings for all text segments...")
        all_embeddings = embedder.encode(df['text'].tolist(), show_progress_bar=True)
        log_with_time("✓ Embeddings created.")
        
        log_with_time("Training classifiers on heuristic labels...")
        clf_abstract = LogisticRegression(random_state=42, max_iter=1000).fit(all_embeddings, df['heuristic_abstract'])
        clf_tangible = LogisticRegression(random_state=42, max_iter=1000).fit(all_embeddings, df['heuristic_tangible'])
        log_with_time("✓ Classifiers trained.")

        df['ml_abstract'] = clf_abstract.predict(all_embeddings)
        df['ml_tangible'] = clf_tangible.predict(all_embeddings)
        log_with_time("✓ Supervised ML classification complete.")
    except Exception as e:
        log_with_time(f"Could not complete supervised ML stage: {e}")
        log_with_time("Skipping this stage. The final output will not contain 'ml_' columns.")
        df[['ml_abstract', 'ml_tangible']] = -1, -1
        
    return df

# ---
# MAIN ORCHESTRATOR
# ---
def main():
    # Step 1: Load and standardize data
    df = load_and_segment_data()
    if df is None: return

    # Step 2: Run Heuristic Classifier
    df = run_heuristic_classifier(df)
    
    # Step 3: Run Flan-T5 Classifier
    df = run_flan_classifier(df)

    # Step 4: Attempt Supervised ML Classifier
    df = run_supervised_ml_classifier(df)
    
    # Step 5: Final summary and save
    log_with_time("--- Final Stage: Summarizing and Saving Results ---")
    
    # Calculate agreement between methods if they ran successfully
    if 'flan_abstract' in df.columns and df['flan_abstract'].max() >= 0:
        flan_pred = df.apply(lambda row: 'abstract' if row['flan_abstract'] == 1 else 'tangible' if row['flan_tangible'] == 1 else 'both', axis=1)
        heuristic_pred = df.apply(lambda row: 'abstract' if row['heuristic_abstract'] == 1 else 'tangible' if row['heuristic_tangible'] == 1 else 'both', axis=1)
        agreement = np.mean(flan_pred == heuristic_pred)
        log_with_time(f"Agreement between Flan-T5 and Heuristic Keyword classifier: {agreement:.2%}")

    if 'ml_abstract' in df.columns and df['ml_abstract'].max() >= 0:
        ml_pred = df.apply(lambda row: 'abstract' if row['ml_abstract'] == 1 else 'tangible' if row['ml_tangible'] == 1 else 'both', axis=1)
        agreement_ml = np.mean(flan_pred == ml_pred)
        log_with_time(f"Agreement between Flan-T5 and Supervised ML classifier: {agreement_ml:.2%}")
        
    try:
        df.to_csv(OUTPUT_FILE, index=False)
        log_with_time(f"✓ Successfully saved all {len(df)} classified segments to '{OUTPUT_FILE}'")
    except Exception as e:
        log_with_time(f"Error saving final results: {e}")

    print("\n=== SCRIPT COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()