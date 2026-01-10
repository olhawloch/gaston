import os
import pandas as pd
from gensim.models import Word2Vec
from datetime import datetime
from tqdm import tqdm

# --- Configuration ---
DATA_PATH_GRAPH_OUTPUT = "../GASTON/pretraining/graph_data"

# --- Input File ---
USER_COMMUNITY_EDGES_PATH = os.path.join(DATA_PATH_GRAPH_OUTPUT, "user_community_active_edges.parquet")

# --- Output Paths ---
# Where the final Word2Vec model and vectors will be saved
BASE_MODEL_SAVE_PATH = "../GASTON/subreddit2vec"
os.makedirs(BASE_MODEL_SAVE_PATH, exist_ok=True)

# --- Word2Vec Model Hyperparameters ---
EMBEDDING_SIZE = 768 # consider smaller embedding sizes
WINDOW_SIZE = 5
MIN_COUNT = 2
WORKERS = os.cpu_count() or 1
# Use the Skip-gram model
SG = 1 
NEGATIVE_SAMPLES = 5
EPOCHS = 10

def create_subreddit_sentences(file_path: str) -> list:
    """
    Reads the user-community edge file and groups it to create "sentences,"
    where each sentence is a list of subreddits a user has interacted with.
    """
    print(f"--- Loading user-community edges from: {os.path.basename(file_path)} ---")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}. Please run the pre-training data collection script first.")
    
    df = pd.read_parquet(file_path)
    print(f"Loaded {len(df):,} total interactions.")
    
    print("\n--- Grouping interactions by user to create 'sentences' ---")
    tqdm.pandas(desc="Grouping by user")
    user_sentences = df.groupby('source_id')['target_id'].progress_apply(list)
    
    print(f"Created {len(user_sentences):,} unique user sentences.")
    
    # Convert the Series of lists into a final list of lists for Word2Vec.
    return user_sentences.tolist()

def train_subreddit2vec_model(sentences: list):
    """Trains a Word2Vec model on the provided sentences."""
    print("\n--- Training Subreddit2Vec (Word2Vec) model ---")
    
    model = Word2Vec(
        sentences=sentences,
        vector_size=EMBEDDING_SIZE,
        window=WINDOW_SIZE,
        min_count=MIN_COUNT,
        workers=WORKERS,
        sg=SG,
        negative=NEGATIVE_SAMPLES,
        epochs=EPOCHS,
        compute_loss=True
    )
    
    print("Model training complete.")
    return model

def main():
    """Main function to run the Subreddit2Vec pipeline."""
    
    # 1. Create the corpus of sentences from your existing graph data
    sentences = create_subreddit_sentences(USER_COMMUNITY_EDGES_PATH)
    
    if not sentences:
        print("No sentences were created. Exiting.")
        return
        
    # 2. Train the Word2Vec model
    model = train_subreddit2vec_model(sentences)
    
    # 3. Save the model and vectors
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_save_path = os.path.join(BASE_MODEL_SAVE_PATH, f"subreddit2vec_model_{timestamp}.bin")
    vectors_save_path = os.path.join(BASE_MODEL_SAVE_PATH, f"subreddit2vec_vectors_{timestamp}.kv")
    
    print(f"\n--- Saving model to: {model_save_path} ---")
    model.save(model_save_path)
    
    print(f"--- Saving vectors to: {vectors_save_path} ---")
    model.wv.save(vectors_save_path)
    
    print("\n--- Testing Model (example) ---")
    # Find a common subreddit to test that is likely in the vocab
    test_subreddit = "askreddit" 
    if test_subreddit in model.wv:
        print(f"Subreddits most similar to '{test_subreddit}':")
        for word, score in model.wv.most_similar(test_subreddit, topn=10):
            print(f"  - {word}: {score:.4f}")
    else:
        print(f"'{test_subreddit}' not in model vocabulary. Try another common subreddit.")
        try:
            example_key = next(iter(model.wv.key_to_index))
            print(f"\nExample from vocabulary. Subreddits most similar to '{example_key}':")
            for word, score in model.wv.most_similar(example_key, topn=10):
                print(f"  - {word}: {score:.4f}")
        except StopIteration:
            print("Model vocabulary is empty.")

    print("\n--- Subreddit2Vec generation complete! ---")

if __name__ == "__main__":
    main()