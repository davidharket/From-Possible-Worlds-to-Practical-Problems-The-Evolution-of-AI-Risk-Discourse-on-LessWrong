import requests
import json
import time
import os
import re
from datetime import datetime, timedelta, timezone
from dateutil import parser
from bs4 import BeautifulSoup

# ---
# SHARED CONFIGURATION
# ---
FILTERED_METADATA_FILE = "lesswrong_intent_filtered_posts.json"
POSTS_CONTENT_DIR = "lesswrong_full_posts_content"
GRAPHQL_ENDPOINT = "https://www.lesswrong.com/graphql"

# ---
# STAGE 1: METADATA FETCHING LOGIC
# ---
HEADERS_STAGE_1 = {"Content-Type": "application/json", "User-Agent": "LessWrongContentFetcher/2.0"}

def send_graphql_query_metadata(query, variables=None):
    try:
        response = requests.post(GRAPHQL_ENDPOINT, json={"query": query, "variables": variables}, headers=HEADERS_STAGE_1, timeout=60)
        response.raise_for_status(); return response.json()
    except requests.exceptions.RequestException as e:
        print(f"    GraphQL request failed: {e}"); return None

def fetch_posts_for_window(start_iso, end_iso, batch_size=50):
    query = """
    query GetPostsWithTermsInput($termsInput: JSON) {
      posts(input: { terms: $termsInput }) { results { _id title slug pageUrl postedAt baseScore voteCount commentCount tags { name slug } user { username slug } } }
    }"""
    posts, offset = [], 0
    print(f"  Fetching posts for window: {start_iso} to {end_iso}")
    for page in range(1, 201):
        print(f"    Window Page {page}: offset {offset}...")
        terms = {"after": start_iso, "before": end_iso, "sortBy": "postedAt", "sortOrder": "asc", "limit": batch_size, "offset": offset}
        response = send_graphql_query_metadata(query, {"termsInput": terms})
        if not response or not response.get('data', {}).get('posts', {}).get('results'):
            print("    Received empty or invalid response. Ending this window."); break
        batch = response['data']['posts']['results']
        posts.extend(batch)
        if len(batch) < batch_size: break
        offset += batch_size; time.sleep(1.5)
    return posts

def run_stage_1_fetch_all_metadata():
    print("\n--- RUNNING STAGE 1: Fetch All Raw Metadata ---")
    start_date, end_date = datetime(2021, 11, 30, tzinfo=timezone.utc), datetime(2023, 11, 30, 23, 59, 59, tzinfo=timezone.utc)
    all_posts, seen_ids, window_end = [], set(), end_date
    while window_end > start_date:
        window_start = max(start_date, window_end - timedelta(days=90))
        posts = fetch_posts_for_window(window_start.isoformat().replace('+00:00', 'Z'), window_end.isoformat().replace('+00:00', 'Z'))
        for post in posts:
            if post['_id'] not in seen_ids: all_posts.append(post); seen_ids.add(post['_id'])
        window_end = window_start - timedelta(microseconds=1)
        if window_start == start_date: break
        time.sleep(2)
    print(f"\nFinished fetching. Total unique raw posts found: {len(all_posts)}")
    print("--- STAGE 1 COMPLETE ---")
    return all_posts

# ---
# STAGE 2: INTENTIONAL FILTERING LOGIC
# ---
def run_stage_2_intent_filter(posts_to_filter):
    print("\n--- RUNNING STAGE 2: Apply Intentional Tag Filter ---")
    if not posts_to_filter: print("No posts provided to filter. Skipping."); return None
    TAG_SEQUENCES, SEQ1, SEQ2 = ["ai risk", "alignment"], "regulation", "ai risk"
    filtered, added = [], set()
    print(f"Filtering {len(posts_to_filter)} posts for tags containing 'ai risk' OR ('regulation' AND 'ai risk')...")
    for post in posts_to_filter:
        if post['_id'] in added: continue
        for tag in post.get("tags", []) or []:
            if not isinstance(tag, dict): continue
            tag_text = (tag.get("name", "") + " " + tag.get("slug", "")).lower()
            if any(s in tag_text for s in TAG_SEQUENCES) or (SEQ1 in tag_text and SEQ2 in tag_text):
                filtered.append(post); added.add(post['_id']); break
    print(f"\nFound {len(filtered)} posts matching the intentional tag criteria.")
    filtered.sort(key=lambda x: x.get("postedAt", ""));
    with open(FILTERED_METADATA_FILE, "w", encoding="utf-8") as f: json.dump(filtered, f, indent=2)
    print(f"Saved {len(filtered)} filtered posts to '{FILTERED_METADATA_FILE}'")
    print("--- STAGE 2 COMPLETE ---")
    return FILTERED_METADATA_FILE

# ---
# STAGE 3: FULL CONTENT SCRAPING LOGIC
# ---
HEADERS_STAGE_3 = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}

def pure_text_extraction(html):
    if not html: return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]): tag.decompose()
    return re.sub(r'\n{3,}', '\n\n', soup.get_text(separator="\n", strip=True)).strip()

def fetch_apollo_state(post_url):
    try:
        response = requests.get(post_url, headers=HEADERS_STAGE_3, timeout=45)
        response.raise_for_status()
        match = re.search(r'window\.__APOLLO_STATE__\s*=\s*(\{.*?\});?\s*<\/script>', response.text, re.DOTALL)
        return json.loads(match.group(1)) if match else None
    except Exception as e:
        print(f"    Request or parsing error for {post_url}: {e}"); return None

# --- REVISED AND ROBUST extract_full_content FUNCTION ---
def extract_full_content(apollo_state, post_id):
    """Robustly extracts detailed post and comment data, handling null fields."""
    post_key = f"Post:{post_id}"
    post_data = apollo_state.get(post_key)
    if not post_data:
        for key, value in apollo_state.items():
            if key.startswith("Post:") and value.get("_id") == post_id: post_data = value; break
    if not post_data: return None, []

    # --- Robustly extract post content ---
    post_html = ""
    contents_obj = post_data.get("contents")
    if isinstance(contents_obj, dict):
        ref = contents_obj.get("__ref")
        if ref and ref in apollo_state: post_html = apollo_state.get(ref, {}).get("html", "")
    
    # --- Robustly extract post author ---
    author_username = "Unknown Author"
    author_obj = post_data.get("user")
    if isinstance(author_obj, dict):
        ref = author_obj.get("__ref")
        if ref and ref in apollo_state: author_username = apollo_state.get(ref, {}).get("username", "Unknown Author")
    
    structured_post = {
        "id": post_id, "title": post_data.get("title"), "postedAt": post_data.get("postedAt"),
        "author": author_username, "commentCount": post_data.get("commentCount"),
        "content": pure_text_extraction(post_html)
    }

    # --- Robustly extract comments and all their metadata ---
    comments = []
    for key, value in apollo_state.items():
        if key.startswith("Comment:") and value.get("postId") == post_id:
            comment_html = ""
            comment_author_username = "Unknown Commenter"
            
            # FIX 1: Robustly handle null 'contents' for comments
            comment_contents_obj = value.get("contents")
            if isinstance(comment_contents_obj, dict):
                ref = comment_contents_obj.get("__ref")
                if ref and ref in apollo_state: comment_html = apollo_state.get(ref, {}).get("html", "")
            
            # FIX 2: Robustly handle null 'user' for comments
            comment_author_obj = value.get("user")
            if isinstance(comment_author_obj, dict):
                ref = comment_author_obj.get("__ref")
                if ref and ref in apollo_state: comment_author_username = apollo_state.get(ref, {}).get("username", "Unknown Commenter")
            
            comments.append({
                "id": value.get("_id"), "author": comment_author_username,
                "postedAt": value.get("postedAt"), "parentCommentId": value.get("parentCommentId"),
                "baseScore": value.get("baseScore"), "voteCount": value.get("voteCount"),
                "content": pure_text_extraction(comment_html)
            })
    return structured_post, comments

def run_stage_3_scrape_full_content(input_filepath):
    """Main function for Stage 3, orchestrating the scraping."""
    print("\n--- RUNNING STAGE 3: Scrape Full Post Content and Metadata ---")
    if not input_filepath or not os.path.exists(input_filepath):
        print(f"Error: Filtered metadata file '{input_filepath}' not found. Cannot proceed."); return
        
    os.makedirs(POSTS_CONTENT_DIR, exist_ok=True)
    with open(input_filepath, "r", encoding="utf-8") as f: posts_metadata = json.load(f)
        
    print(f"Loaded {len(posts_metadata)} posts to scrape from '{input_filepath}'")
    
    for post_meta in posts_metadata:
        post_id = post_meta.get("_id")
        out_file = os.path.join(POSTS_CONTENT_DIR, f"{post_id}.json")
        if os.path.exists(out_file):
            print(f"  Skipping '{post_meta.get('title')}' (already exists)"); continue
            
        print(f"  Processing '{post_meta.get('title')}' ({post_id})")
        post_url = post_meta.get("pageUrl", f"https://www.lesswrong.com/posts/{post_id}")
        apollo_state = fetch_apollo_state(post_url)
        if not apollo_state: continue

        post, comments = extract_full_content(apollo_state, post_id)
        if not post: continue
        
        final_data = {"post": post, "comments": comments, "retrievedCommentsCount": len(comments),
                      "sourceMethod": "apollo_state_parsing", "originalMeta": post_meta}
        
        with open(out_file, "w", encoding="utf-8") as f: json.dump(final_data, f, indent=2)
        print(f"    Saved data for post {post_id} to '{out_file}'")
        time.sleep(2)
        
    print("--- STAGE 3 COMPLETE ---")

# ---
# MAIN ORCHESTRATOR
# ---
if __name__ == "__main__":
    print("--- Starting Full Data Collection Pipeline ---")
    
    if os.path.exists(FILTERED_METADATA_FILE):
        print(f"Found existing filtered metadata file: '{FILTERED_METADATA_FILE}'")
        print("Skipping Stage 1 (Metadata Fetch) and Stage 2 (Filtering).")
        filtered_file_to_use = FILTERED_METADATA_FILE
    else:
        print(f"'{FILTERED_METADATA_FILE}' not found. Running full pipeline from Stage 1.")
        raw_posts = run_stage_1_fetch_all_metadata()
        filtered_file_to_use = run_stage_2_intent_filter(raw_posts) if raw_posts else None

    if filtered_file_to_use:
        run_stage_3_scrape_full_content(filtered_file_to_use)
    else:
        print("Halting pipeline because no filtered metadata file is available.")

    print("\nData collection pipeline finished successfully.")