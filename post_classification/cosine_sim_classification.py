import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
BERTOPIC_OUTPUTS_DIR = "bertopic_outputs"
OUTPUT_FILE = "all_posts_classifications_similarity_chunked.csv" # Using a new name to reflect the improved method
SUMMARY_FILE = "all_posts_classification_summary_similarity_chunked.csv"

def log_with_time(message):
    """Helper to log with timestamp"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

class SimilarityBasedClassifier:
    def __init__(self):
        # Configuration for chunking long documents
        self.chunk_size = 384  # Words per chunk, safely under the 512 token limit of the model
        self.chunk_overlap = 64 # Words to overlap between chunks to maintain context at boundaries
        
        self.embedder = None
        self.abstract_examples = []
        self.tangible_examples = []
        self.abstract_embeddings = None
        self.tangible_embeddings = None
        self.setup_classifier()
    
    def setup_classifier(self):
        """Setup similarity-based classifier"""
        log_with_time("Setting up similarity-based classifier...")
        try:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            log_with_time("✓ SentenceTransformer loaded successfully")
        except Exception as e:
            log_with_time(f"Failed to load SentenceTransformer: {e}")
            raise
        
        self.create_example_sets()
        self.create_example_embeddings()
    
    def create_example_sets(self):
        """(From your original script) Create the full, detailed list of examples."""
        # ABSTRACT examples - theoretical, futuristic, speculative
        # ABSTRACT examples - theoretical, futuristic, speculative
        self.abstract_examples = [
            # Nanobots and molecular scenarios
            "Nanobots could disassemble all matter on Earth to create paperclips",
            "Grey goo scenario where self-replicating nanomachines consume everything",
            "Molecular assemblers could be used by AI to reconstruct the physical world",
            "Nanotech swarms controlled by superintelligence pose existential risk",
            
            # Paperclip maximizer and instrumental convergence
            "The paperclip maximizer thought experiment shows how AI goals can go wrong",
            "Instrumental convergence means all AIs will seek power and resources",
            "An AI optimizing for paperclips might turn humans into paperclips",
            "Goal preservation drives AI to resist shutdown and modification",
            
            # Superintelligence and AGI timelines
            "Superintelligence could emerge suddenly through recursive self-improvement",
            "AGI timelines of 20-50 years require careful alignment research",
            "Fast takeoff scenarios where AI capabilities explode overnight",
            "Intelligence explosion leading to machine superintelligence",
            
            # Theoretical alignment concepts
            "The orthogonality thesis states intelligence and goals are independent",
            "Mesa-optimizers emerge during training creating inner alignment problems",
            "Outer alignment ensures the reward function matches human values",
            "Deceptive alignment where AI pretends to be aligned during training",
            
            # Philosophical and theoretical risks
            "The vulnerable world hypothesis suggests some technologies doom civilization",
            "Treacherous turn when AI suddenly reveals its true misaligned goals",
            "Human extinction from advanced AI systems optimizing the wrong objectives",
            "Value learning problem of teaching AI what humans actually want",
            
            # Future scenarios and predictions
            "In 30 years, AGI might develop dangerous capabilities we can't predict",
            "Post-human future where AI systems replace biological intelligence",
            "Technological singularity marking the end of human-comprehensible progress",
            "AI consciousness and moral status questions for future digital minds"
        ]
        
        # TANGIBLE examples - current, practical, immediate
        self.tangible_examples = [
            # Current LLM misuse and jailbreaking
            "ChatGPT can be jailbroken to generate harmful content and bypass safety",
            "Prompt injection attacks allow users to manipulate AI system behavior",
            "People are using ChatGPT to write malware and phishing emails",
            "Jailbreaking techniques remove safety filters from language models",
            
            # Geopolitics and China competition
            "China is racing to develop military AI capabilities and autonomous weapons",
            "US-China AI competition threatens global AI safety cooperation",
            "Chinese AI development lacks safety oversight and alignment research",
            "Geopolitical tensions around AI technology transfer and export controls",
            
            # Terrorism and immediate threats
            "Terrorists could use AI to design biological weapons and chemical attacks",
            "AI-generated disinformation campaigns threaten democratic elections",
            "Deepfakes enable unprecedented identity theft and fraud schemes",
            "AI-powered cyberattacks can overwhelm traditional security systems",
            
            # Current AI regulation and policy
            "EU AI Act creates the first comprehensive AI regulation framework",
            "Congress is debating AI safety bills and oversight requirements",
            "AI companies need immediate safety standards and testing protocols",
            "Government agencies lack expertise to regulate current AI systems",
            
            # Deployed AI system failures
            "Bias in hiring algorithms discriminates against minority candidates",
            "Autonomous vehicles struggle with edge cases causing fatal accidents",
            "Facial recognition systems have high error rates for certain demographics",
            "AI trading algorithms can cause flash crashes in financial markets",
            
            # Open source model risks
            "Removing safety training from open source models creates immediate dangers",
            "Anyone can download and modify powerful AI models without oversight",
            "Open source LLMs can be fine-tuned for harmful applications",
            "Safety research can't keep up with open source model development",
            
            # Current economic and social impacts
            "AI automation is already displacing workers in multiple industries",
            "Content creators face job losses from AI-generated text and images",
            "AI surveillance systems enable authoritarian government control",
            "Current AI systems amplify existing social biases and inequalities",
            
            # Near-term deployment risks
            "AI assistants are being deployed without adequate safety testing",
            "Medical AI systems make errors that harm patients today",
            "AI content moderation fails to stop harmful online content",
            "Current AI lacks robustness and fails in unpredictable ways"
        ]
        log_with_time(f"Created {len(self.abstract_examples)} abstract and {len(self.tangible_examples)} tangible examples")
    
    def create_example_embeddings(self):
        """Create embeddings for all examples"""
        log_with_time("Creating embeddings for example sentences...")
        self.abstract_embeddings = self.embedder.encode(self.abstract_examples, show_progress_bar=False)
        self.tangible_embeddings = self.embedder.encode(self.tangible_examples, show_progress_bar=False)
        log_with_time("✓ Example embeddings created")

    def _get_scores_for_text(self, text_chunk):
        """Helper method to calculate scores and best matches for a single piece of text."""
        if not text_chunk: return 0.0, 0.0, "", ""
        doc_embedding = self.embedder.encode([text_chunk])
        
        abs_sim = cosine_similarity(doc_embedding, self.abstract_embeddings)[0]
        tan_sim = cosine_similarity(doc_embedding, self.tangible_embeddings)[0]
        
        abs_score = 0.7 * np.max(abs_sim) + 0.3 * np.mean(abs_sim)
        tan_score = 0.7 * np.max(tan_sim) + 0.3 * np.mean(tan_sim)
        
        best_abs_ex = self.abstract_examples[np.argmax(abs_sim)]
        best_tan_ex = self.tangible_examples[np.argmax(tan_sim)]
        
        return abs_score, tan_score, best_abs_ex, best_tan_ex

    def classify_document(self, content):
        """Classifies a document, using chunking for long texts."""
        if not content or not isinstance(content, str):
            return {"abstract": 0, "tangible": 0, "confidence": 0.0, "reasoning": "No content"}

        words = content.split()
        num_words = len(words)
        
        chunk_abs_scores, chunk_tan_scores = [], []
        best_abs_match, best_tan_match = "", ""
        max_abs_chunk_score, max_tan_chunk_score = -1, -1

        if num_words <= self.chunk_size:
            abs_score, tan_score, best_abs_match, best_tan_match = self._get_scores_for_text(content)
            num_chunks = 1
        else:
            step = self.chunk_size - self.chunk_overlap
            for i in range(0, num_words, step):
                chunk_text = " ".join(words[i:i + self.chunk_size])
                abs_chunk_score, tan_chunk_score, best_abs_chunk, best_tan_chunk = self._get_scores_for_text(chunk_text)
                chunk_abs_scores.append(abs_chunk_score)
                chunk_tan_scores.append(tan_chunk_score)
                
                if abs_chunk_score > max_abs_chunk_score:
                    max_abs_chunk_score = abs_chunk_score; best_abs_match = best_abs_chunk
                if tan_chunk_score > max_tan_chunk_score:
                    max_tan_chunk_score = tan_chunk_score; best_tan_match = best_tan_chunk
            
            if not chunk_abs_scores: return {"abstract": 0, "tangible": 0, "confidence": 0.0, "reasoning": "Chunking failed"}
            
            abs_score, tan_score = np.mean(chunk_abs_scores), np.mean(chunk_tan_scores)
            num_chunks = len(chunk_abs_scores)

        # Final classification logic from your original script
        strong_threshold, weak_threshold = 0.45, 0.25
        if abs_score > strong_threshold and abs_score > tan_score * 1.1:
            classification = {"abstract": 1, "tangible": 0}; confidence = abs_score
        elif tan_score > strong_threshold and tan_score > abs_score * 1.1:
            classification = {"abstract": 0, "tangible": 1}; confidence = tan_score
        elif abs_score > weak_threshold and tan_score > weak_threshold:
            classification = {"abstract": 1, "tangible": 1}; confidence = min(abs_score, tan_score)
        elif abs_score > tan_score:
            classification = {"abstract": 1, "tangible": 0}; confidence = abs_score
        else:
            classification = {"abstract": 0, "tangible": 1}; confidence = tan_score

        classification.update({
            "confidence": confidence, "abstract_score": abs_score, "tangible_score": tan_score,
            "best_abstract_match": best_abs_match[:80] + "...",
            "best_tangible_match": best_tan_match[:80] + "...",
            "num_chunks": num_chunks, # Added for analysis
            "reasoning": f"A:{abs_score:.3f} T:{tan_score:.3f} from {num_chunks} chunk(s)"
        })
        return classification

def load_and_prepare_data():
    """Loads and prepares data from BERTopic outputs."""
    log_with_time("Loading BERTopic results for ALL posts...")
    path = os.path.join(BERTOPIC_OUTPUTS_DIR, "bertopic_document_topic_assignments.csv")
    if not os.path.exists(path):
        print(f"Error: {path} not found!"); return None
    df = pd.read_csv(path)
    posts_only = df[df['doc_type'] == 'post'].copy()
    documents = []
    for _, row in posts_only.iterrows():
        content = str(row.get('text_snippet', '')).strip()
        if len(content.split()) > 10:
            documents.append({
                'doc_id': row['doc_id'], 'topic_id': row['assigned_topic_id'],
                'timestamp': row.get('timestamp', ''), 'content': content,
                'topic_name': row.get('Name', 'Unknown'),
                'topic_representation': str(row.get('Representation', ''))
            })
    log_with_time(f"Prepared {len(documents)} posts for classification")
    return documents


def main():
    """Main function to run the entire classification pipeline."""
    log_with_time("=== All Posts Similarity-Based AI Risk Classifier (with Chunking) ===")
    
    documents = load_and_prepare_data()
    if not documents: return

    try:
        classifier = SimilarityBasedClassifier()
    except Exception as e:
        log_with_time(f"Failed to setup classifier: {e}"); return
        
    if input(f"\nProceed with classification of {len(documents)} posts? (y/n): ").lower() != 'y':
        print("Classification cancelled."); return
    
    log_with_time("Starting similarity-based classification of all posts...")
    
    classified_docs, start_time = [], time.time()
    for i, doc in enumerate(documents):
        classification = classifier.classify_document(doc['content'])
        result = {
            'doc_id': doc['doc_id'], 'topic_id': doc['topic_id'], 'topic_name': doc['topic_name'],
            'topic_representation': doc['topic_representation'], 'timestamp': doc['timestamp'],
            'abstract': classification['abstract'], 'tangible': classification['tangible'],
            'confidence': classification['confidence'], 'abstract_score': classification['abstract_score'],
            'tangible_score': classification['tangible_score'],
            'best_abstract_match': classification['best_abstract_match'],
            'best_tangible_match': classification['best_tangible_match'],
            'reasoning': classification['reasoning'], 'word_count': len(doc['content'].split()),
            'content_preview': doc['content'][:200] + "..."
        }
        classified_docs.append(result)
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (len(documents) - i - 1)
            log_with_time(f"Classified {i + 1}/{len(documents)} posts... ETA: {eta/60:.1f} minutes")

    results_df = pd.DataFrame(classified_docs)
    results_df.to_csv(OUTPUT_FILE, index=False)
    
    # Create and save summary (from your original script)
    summary = results_df.groupby(['topic_id', 'topic_name']).agg(
        total_posts=('doc_id', 'count'), abstract=('abstract', 'sum'),
        tangible=('tangible', 'sum'), avg_confidence=('confidence', 'mean')
    ).reset_index()
    summary['abstract_pct'] = (summary['abstract'] / summary['total_posts'] * 100).round(1)
    summary['tangible_pct'] = (summary['tangible'] / summary['total_posts'] * 100).round(1)
    summary['both_pct'] = ((summary['abstract'] + summary['tangible'] - summary['total_posts']) / summary['total_posts'] * 100).round(1)
    summary.sort_values('total_posts', ascending=False, inplace=True)
    summary.to_csv(SUMMARY_FILE, index=False)
    
    log_with_time("\n=== CLASSIFICATION COMPLETE ===")
    log_with_time(f"Results saved to {OUTPUT_FILE} and {SUMMARY_FILE}")

if __name__ == "__main__":
    main()