import pandas as pd
import spacy
import textstat
from textblob import TextBlob
import os
import string

# 1. Load your clean master dataset
input_file = 'data/processed/master_dataset.csv'
output_file = 'data/processed/stylometric_features.csv'

print(" Loading compiled text dataset...")
if not os.path.exists(input_file):
    raise FileNotFoundError(f" Could not find {input_file}. Did you compile it?")

df = pd.read_csv(input_file)
print(f" Matrix loaded successfully: {len(df)} total rows across 5 models.")

# 2. Load the spaCy English NLP engine
print(" Initializing spaCy English linguistic model...")
nlp = spacy.load("en_core_web_sm")

def extract_stylometrics(text):
    # Fallback for unexpected empty cells
    text_str = str(text) if pd.notna(text) else ""
    if not text_str.strip():
        return {
            "word_count": 0, "char_count": 0, "avg_word_length": 0,
            "avg_sentence_length": 0, "punctuation_density": 0,
            "noun_density": 0, "verb_density": 0, "adj_density": 0, "adv_density": 0,
            "flesch_reading_ease": 0, "gunning_fog": 0,
            "sentiment_polarity": 0, "sentiment_subjectivity": 0
        }
    
    # Run the raw string through the spaCy processing pipeline
    doc = nlp(text_str)
    
    # --- A. LEXICAL METRICS ---
    tokens = [token for token in doc if not token.is_space]
    words = [token for token in tokens if not token.is_punct]
    
    word_count = len(words)
    char_count = len(text_str)
    avg_word_len = sum(len(w.text) for w in words) / word_count if word_count > 0 else 0
    
    # --- B. SYNTACTICAL METRICS ---
    sentences = list(doc.sents)
    sentence_count = len(sentences)
    avg_sentence_len = word_count / sentence_count if sentence_count > 0 else 0
    
    # Count total explicit punctuation marks
    punct_count = sum(1 for token in tokens if token.is_punct)
    punct_density = punct_count / len(tokens) if len(tokens) > 0 else 0
    
    # Part-of-Speech Tagging Density distributions
    noun_count = sum(1 for token in words if token.pos_ in ["NOUN", "PROPN"])
    verb_count = sum(1 for token in words if token.pos_ == "VERB")
    adj_count = sum(1 for token in words if token.pos_ == "ADJ")
    adv_count = sum(1 for token in words if token.pos_ == "ADV")
    
    noun_density = noun_count / word_count if word_count > 0 else 0
    verb_density = verb_count / word_count if word_count > 0 else 0
    adj_density = adj_count / word_count if word_count > 0 else 0
    low_adv_density = adv_count / word_count if word_count > 0 else 0

    # --- C. SEMANTIC & READABILITY METRICS ---
    flesch = textstat.flesch_reading_ease(text_str)
    fog = textstat.gunning_fog(text_str)
    
    blob = TextBlob(text_str)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # Return as a clean row dictionary map
    return {
        "word_count": word_count,
        "char_count": char_count,
        "avg_word_length": avg_word_len,
        "avg_sentence_length": avg_sentence_len,
        "punctuation_density": punct_density,
        "noun_density": noun_density,
        "verb_density": verb_density,
        "adj_density": adj_density,
        "adv_density": low_adv_density,
        "flesch_reading_ease": flesch,
        "gunning_fog": fog,
        "sentiment_polarity": polarity,
        "sentiment_subjectivity": subjectivity
    }

print("\n Running feature extraction pipeline over dataset. Please stand by...")
features_list = []

# Process rows dynamically
for idx, row in df.iterrows():
    if (idx + 1) % 100 == 0 or (idx + 1) == len(df):
        print(f"   Processed {idx + 1}/{len(df)} entries...")
        
    metrics = extract_stylometrics(row['response'])
    
    # Merge structural keys together
    metrics['prompt_id'] = row['prompt_id']
    metrics['category'] = row['category']
    metrics['model'] = row['model']
    features_list.append(metrics)

# 3. Convert array to structured dataframe matrix and export
features_df = pd.DataFrame(features_list)

# Reorder columns logically to put anchors first
anchor_cols = ['prompt_id', 'category', 'model']
other_cols = [col for col in features_df.columns if col not in anchor_cols]
features_df = features_df[anchor_cols + other_cols]

features_df.to_csv(output_file, index=False)
print(f"\n Success! Numerical feature matrix exported to: {output_file}")
print(f" Transformed text dimension into structural grid shape: {features_df.shape}")