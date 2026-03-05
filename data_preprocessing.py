import argparse
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

# Ensure NLTK stopwords are downloaded:
nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z0-9#\s]', '', text)
    tokens = [w for w in text.split() if w not in STOPWORDS]
    return ' '.join(tokens)

def extract_hashtags(text):
    return ' '.join(re.findall(r'#\w+', text.lower()))

def main(args):
    df = pd.read_csv(args.input)
    df['cleaned'] = df['text'].apply(clean_text)
    df['hashtags'] = df['text'].apply(extract_hashtags)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_text = vectorizer.fit_transform(df['cleaned'])
    X_hash = vectorizer.fit_transform(df['hashtags'])

    X = np.hstack([X_text.toarray(), X_hash.toarray()])
    np.savez_compressed(args.output, features=X)
    print(f"Saved vectorized data to {args.output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    main(args)