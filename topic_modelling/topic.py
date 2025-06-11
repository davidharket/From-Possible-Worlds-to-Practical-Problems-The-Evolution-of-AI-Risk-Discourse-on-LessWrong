import json
import os
import glob
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import re
from datetime import datetime

# --- Configuration ---
# **** MODIFICATION HERE ****
POSTS_DIR = "lesswrong_full_posts_intent_filtered" # Directory where your intent-filtered JSON files are stored
# **** END OF MODIFICATION ****

# Consider using a specific sentence transformer model
# Using a smaller model for quicker processing, but larger models might yield better results.
# Common choice: "all-MiniLM-L6-v2" or "all-mpnet-base-v2" (better but slower)
SENTENCE_MODEL_NAME = "all-MiniLM-L6-v2"
# BERTopic parameters (can be tuned)
MIN_TOPIC_SIZE = 15
NR_TOPICS = None # Let HDBSCAN determine the number of topics, or set to an int like 50
# For topics over time
DATETIME_FORMAT_STRPTIME = "%Y-%m-%dT%H:%M:%S.%fZ" # Format of 'postedAt'
# If your 'postedAt' sometimes doesn't have milliseconds:
# DATETIME_FORMAT_STRPTIME_NO_MS = "%Y-%m-%dT%H:%M:%SZ" # Not strictly needed if first try always includes .%f and second replaces it
TIME_RESOLUTION = "M" # "D" for day, "W" for week, "M" for month, "Q" for quarter, "Y" for year

# --- 1. Load and Preprocess Data ---
def load_and_preprocess_data(posts_dir):
    """
    Loads post and comment data from JSON files, preprocesses text, and extracts timestamps.
    """
    all_texts = []
    all_timestamps_dt = [] # Store as datetime objects
    doc_ids = [] # To keep track of where texts came from (post_id or comment_id)
    doc_types = [] # 'post' or 'comment'

    json_files = glob.glob(os.path.join(posts_dir, "*.json"))
    if not json_files:
        print(f"No JSON files found in directory: {posts_dir}")
        print("Please ensure you have run the full content fetching script and it saved files to this directory.")
        return [], [], [], [] # Return empty lists if no files are found
        
    print(f"Found {len(json_files)} JSON files to process from '{posts_dir}'.")

    successful_parses = 0
    failed_parses = 0

    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {file_path}. Skipping.")
            failed_parses +=1
            continue
        except Exception as e:
            print(f"Warning: Could not read file {file_path} due to {e}. Skipping.")
            failed_parses +=1
            continue

        # Process post
        post_data = data.get("post")
        if post_data and isinstance(post_data, dict) and \
           post_data.get("content") and isinstance(post_data.get("content"), str) and \
           post_data.get("postedAt") and isinstance(post_data.get("postedAt"), str):

            text = post_data["content"].lower()
            text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
            if len(text.split()) > 5: # Only include if it has some substance
                all_texts.append(text)
                timestamp_str = post_data["postedAt"]
                parsed_ts = None
                try:
                    parsed_ts = datetime.strptime(timestamp_str, DATETIME_FORMAT_STRPTIME)
                except ValueError:
                    try: # Try without milliseconds
                        parsed_ts = datetime.strptime(timestamp_str, DATETIME_FORMAT_STRPTIME.replace(".%f", ""))
                    except ValueError:
                        try: # Try with just Z if T is also there
                             parsed_ts = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ")
                        except ValueError:
                            print(f"Warning: Could not parse timestamp '{timestamp_str}' for post {post_data.get('id','N/A')}. Skipping this document's timestamp.")
                all_timestamps_dt.append(parsed_ts) # Append None if parsing failed
                doc_ids.append(f"post_{post_data.get('id', os.path.basename(file_path).replace('.json',''))}")
                doc_types.append("post")
        # else: # Optional: log if post_data is missing expected fields
            # print(f"Warning: Post data in {file_path} is missing content or postedAt, or is not structured as expected.")


        # Process comments
        comments_data = data.get("comments", [])
        if isinstance(comments_data, list):
            for comment in comments_data:
                if comment and isinstance(comment, dict) and \
                   comment.get("content") and isinstance(comment.get("content"), str) and \
                   comment.get("postedAt") and isinstance(comment.get("postedAt"), str):

                    text = comment["content"].lower()
                    text = re.sub(r'\s+', ' ', text).strip()
                    if len(text.split()) > 3: # Comments can be shorter
                        all_texts.append(text)
                        timestamp_str = comment["postedAt"]
                        parsed_ts = None
                        try:
                            parsed_ts = datetime.strptime(timestamp_str, DATETIME_FORMAT_STRPTIME)
                        except ValueError:
                            try: # Try without milliseconds
                                parsed_ts = datetime.strptime(timestamp_str, DATETIME_FORMAT_STRPTIME.replace(".%f", ""))
                            except ValueError:
                                try: # Try with just Z
                                    parsed_ts = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ")
                                except ValueError:
                                    print(f"Warning: Could not parse timestamp '{timestamp_str}' for comment {comment.get('id','N/A')}. Skipping this document's timestamp.")
                        all_timestamps_dt.append(parsed_ts) # Append None if parsing failed
                        doc_ids.append(f"comment_{comment.get('id', 'unknown_comment_id')}")
                        doc_types.append("comment")
                # else: # Optional: log if comment data is missing expected fields
                    # print(f"Warning: Comment data in {file_path} is missing content or postedAt, or is not structured as expected.")
        successful_parses +=1
    
    if failed_parses > 0:
        print(f"Warning: Failed to parse or read {failed_parses} JSON files.")
    if successful_parses == 0 and json_files:
        print("ERROR: No JSON files were successfully parsed. Check file contents and structure.")
        return [], [], [], []


    # Filter out documents where timestamp parsing failed and resulted in None
    filtered_texts = []
    filtered_timestamps_dt = []
    filtered_doc_ids = []
    filtered_doc_types = []

    invalid_timestamp_count = 0
    for text, ts, doc_id, doc_type in zip(all_texts, all_timestamps_dt, doc_ids, doc_types):
        if ts is not None:
            filtered_texts.append(text)
            filtered_timestamps_dt.append(ts)
            filtered_doc_ids.append(doc_id)
            filtered_doc_types.append(doc_type)
        else:
            # print(f"Document {doc_id} skipped due to missing/invalid timestamp.") # Can be verbose
            invalid_timestamp_count +=1
            
    if invalid_timestamp_count > 0:
        print(f"Skipped {invalid_timestamp_count} documents due to invalid or unparseable timestamps.")
    print(f"Loaded {len(filtered_texts)} documents (posts and comments) with valid timestamps for BERTopic analysis.")
    return filtered_texts, filtered_timestamps_dt, filtered_doc_ids, filtered_doc_types

# --- 2. Initialize BERTopic Model ---
def initialize_bertopic_model(sentence_model_name, min_topic_size, nr_topics):
    """
    Initializes the BERTopic model with a sentence transformer and vectorizer.
    """
    print(f"Initializing BERTopic with sentence model: {sentence_model_name}")
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1,2), min_df=5) 

    try:
        embedding_model = SentenceTransformer(sentence_model_name)
    except Exception as e:
        print(f"Error loading SentenceTransformer model '{sentence_model_name}': {e}")
        print("Please ensure the model name is correct and you have an internet connection if downloading.")
        print("Common models: 'all-MiniLM-L6-v2', 'all-mpnet-base-v2'")
        raise # Re-raise the exception to stop execution if model can't be loaded
    
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        language="english",
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,
        calculate_probabilities=False, # Set to True if needed, but False is faster
        verbose=True
    )
    return topic_model

# --- 3. Train Model and Get Topics Over Time ---
def train_and_analyze_topics(topic_model, texts, timestamps_dt, doc_ids, time_resolution):
    """
    Trains the BERTopic model and analyzes topics over time.
    """
    if not texts: # Added check for timestamps_dt as well
        print("No texts or (by extension) no valid timestamps to process. Aborting training.")
        return None, None, None
    if not timestamps_dt: # Explicitly check if timestamps list is empty after filtering
        print("All documents were filtered out due to invalid timestamps. Aborting training.")
        return None, None, None


    print(f"Training BERTopic model on {len(texts)} documents...")
    # Ensure texts and timestamps_dt are perfectly aligned and have the same length
    if len(texts) != len(timestamps_dt):
        print(f"Error: Mismatch between number of texts ({len(texts)}) and timestamps ({len(timestamps_dt)}). Aborting.")
        # This should ideally not happen if load_and_preprocess_data is correct
        return None, None, None

    # Pre-calculating embeddings can sometimes offer more control or reuse, but BERTopic handles it internally.
    # For simplicity, we'll let BERTopic's fit_transform handle embedding generation if not provided.
    # embeddings = topic_model.embedding_model.encode(texts, show_progress_bar=True) # if you want to precompute
    # topics, probabilities = topic_model.fit_transform(texts, embeddings=embeddings) # Pass precomputed
    
    topics, probabilities = topic_model.fit_transform(texts) # Timestamps are passed to topics_over_time

    print("BERTopic model training complete.")
    num_topics_found = len(topic_model.get_topic_info())
    if num_topics_found > 0 and topic_model.get_topic_info().iloc[0]["Topic"] == -1:
        num_topics_found -=1 # Adjust for outlier topic
    print(f"Found {num_topics_found} topics (excluding outlier topic -1 if present).")

    topic_info = topic_model.get_topic_info()
    print("\nSample Topic Info (Top Topics):")
    print(topic_info.head(11)) 

    if not os.path.exists("bertopic_outputs"):
        os.makedirs("bertopic_outputs")
    topic_info_path = os.path.join("bertopic_outputs", "bertopic_topic_info.csv")
    topic_info.to_csv(topic_info_path, index=False)
    print(f"\nSaved topic info to {topic_info_path}")

    # --- Topics Over Time ---
    print(f"\nGenerating topics over time with documents and resolution: {time_resolution}...")
    try:
        # Ensure timestamps_dt is not empty and matches the number of documents used for `topics`
        # `topics` is an array corresponding to the input `texts` to fit_transform
        if not timestamps_dt or len(timestamps_dt) != len(texts):
            print("Error: Timestamps list is empty or does not match the number of documents processed by fit_transform.")
            print("Skipping topics over time analysis.")
            topics_over_time_df = None
        else:
            topics_over_time_df = topic_model.topics_over_time(
                docs=texts, # The original documents
                topics=topics, # The topics assigned by fit_transform
                timestamps=timestamps_dt, # The original datetime objects
                global_tuning=True,
                evolution_tuning=True,
                nr_bins=None 
            )
            print("Topics over time generated.")
            tot_df_path = os.path.join("bertopic_outputs", "bertopic_topics_over_time.csv")
            topics_over_time_df.to_csv(tot_df_path, index=False)
            print(f"Saved topics over time data to {tot_df_path}")

            if topics_over_time_df is not None and not topics_over_time_df.empty:
                fig_topics_over_time = topic_model.visualize_topics_over_time(
                    topics_over_time_df,
                    top_n_topics=min(10, num_topics_found if num_topics_found > 0 else 1) # Visualize top N or fewer if less topics
                )
                fig_tot_path = os.path.join("bertopic_outputs", "bertopic_topics_over_time_visualization.html")
                fig_topics_over_time.write_html(fig_tot_path)
                print(f"Saved topics over time visualization to {fig_tot_path}")
            else:
                print("Topics over time DataFrame is empty. Skipping visualization.")

    except ValueError as ve: # Catches issues like "Timestamps is not sorted"
        print(f"ValueError during topics over time generation: {ve}")
        print("This can happen if timestamps are not sorted or there are issues with data alignment.")
        topics_over_time_df = None
    except Exception as e:
        print(f"Error generating or visualizing topics over time: {e}")
        print("This can happen if there's not enough data variability over time or few topics.")
        import traceback
        traceback.print_exc()
        topics_over_time_df = None

    return topic_model, topic_info, topics_over_time_df


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting BERTopic analysis for LessWrong data...")

    # 1. Load Data
    # This POSTS_DIR should point to where your fetch_full_content.py saved its output
    texts, timestamps_dt, doc_ids, doc_types = load_and_preprocess_data(POSTS_DIR)

    if not texts:
        print("No data loaded or all data was filtered out. Exiting BERTopic analysis.")
    else:
        # 2. Initialize Model
        bertopic_model = initialize_bertopic_model(
            sentence_model_name=SENTENCE_MODEL_NAME,
            min_topic_size=MIN_TOPIC_SIZE,
            nr_topics=NR_TOPICS
        )

        # 3. Train and Analyze
        trained_model, topic_info_df, topics_over_time_df = train_and_analyze_topics(
            bertopic_model,
            texts,
            timestamps_dt, # This list should be aligned with 'texts'
            doc_ids, # doc_ids is also aligned with 'texts'
            time_resolution=TIME_RESOLUTION
        )

        if trained_model and topic_info_df is not None and not topic_info_df.empty : # Check if topic_info_df is valid
            model_save_path = os.path.join("bertopic_outputs", "lesswrong_bertopic_model")
            try:
                # Saving with safetensors if embeddings are not precomputed and passed to fit_transform
                # If you precompute embeddings, you might need to save them separately or use a different method.
                trained_model.save(model_save_path, serialization="safetensors", save_embedding_model=True)
                print(f"\nBERTopic model saved to '{model_save_path}'")
            except Exception as e:
                print(f"Could not save BERTopic model: {e}")
                print("Make sure the path exists and you have write permissions.")


            # Get document info including assigned topic
            # Ensure trained_model.topics_ is available and matches length of texts
            if hasattr(trained_model, 'topics_') and len(trained_model.topics_) == len(texts):
                doc_info_list = []
                for i in range(len(texts)):
                    doc_info_list.append({
                        "doc_id": doc_ids[i],
                        "doc_type": doc_types[i],
                        "timestamp": timestamps_dt[i].isoformat() if timestamps_dt[i] else None,
                        "text_snippet": texts[i][:150] + "...", 
                        "assigned_topic_id": trained_model.topics_[i]
                    })
                
                doc_info_df = pd.DataFrame(doc_info_list)
                # Merge with topic_info_df carefully, especially if topic_info_df could be empty
                doc_info_df = pd.merge(doc_info_df, topic_info_df[["Topic", "Name", "Representation", "Representative_Docs"]], # Added Rep Docs
                                       left_on="assigned_topic_id", right_on="Topic", how="left")
                
                doc_assignments_path = os.path.join("bertopic_outputs", "bertopic_document_topic_assignments.csv")
                doc_info_df.to_csv(doc_assignments_path, index=False)
                print(f"Saved document topic assignments to {doc_assignments_path}")

                print("\nFurther analysis can be done using the trained model and generated CSV files.")
            else:
                print("Could not generate document topic assignments: 'topics_' attribute missing or mismatched length.")
        elif trained_model:
             print("BERTopic model was trained, but no topic info was generated (e.g., no topics found).")
        else:
            print("BERTopic model training failed or was aborted.")


    print("\nScript finished.")