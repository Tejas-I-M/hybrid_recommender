import pandas as pd
import re
import os

def clean_text(text):
    """Converts text to lowercase and removes punctuation for PyTorch."""
    text = str(text).lower()
    # Keep only alphanumeric characters and spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def process_data(input_path, output_path):
    print(f"Loading raw data from: {input_path}")
    df = pd.read_csv(input_path)

    print("Selecting required columns...")
    columns_to_keep = ['UserId', 'ProductId', 'Score', 'Text']
    df = df[columns_to_keep]

    print("Dropping missing values...")
    initial_len = len(df)
    df = df.dropna()
    print(f"Dropped {initial_len - len(df)} invalid rows.")

    print("Mapping strings to integer IDs for Matrix Factorization & PyTorch...")
    # This creates two new columns with integer mappings (0, 1, 2, 3...)
    df['user_id_int'] = df['UserId'].astype('category').cat.codes
    df['item_id_int'] = df['ProductId'].astype('category').cat.codes

    print("Cleaning review text (this may take a minute due to 500k rows)...")
    df['clean_text'] = df['Text'].apply(clean_text)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Saving cleaned dataset to: {output_path}")
    df.to_csv(output_path, index=False)
    print("Day 1 Processing Complete!")

if __name__ == "__main__":
    # The exact path from your screenshot
    RAW_CSV_PATH = r"C:\Users\Tejas\Desktop\hybrid_recommender\data\raw\archive\Reviews.csv"
    # Where we want to save the final clean version
    PROCESSED_CSV_PATH = r"C:\Users\Tejas\Desktop\hybrid_recommender\data\processed\cleaned_reviews.csv"

    process_data(RAW_CSV_PATH, PROCESSED_CSV_PATH)