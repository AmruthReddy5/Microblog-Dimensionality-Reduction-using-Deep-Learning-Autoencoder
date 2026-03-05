## Microblog Dimensionality Reduction using a Deep Learning Autoencoder

## Project Overview
This project performs dimensionality reduction on microblog (Twitter) text using a deep learning Autoencoder. Tweets are converted into high-dimensional TF-IDF vectors (text + hashtags) and then compressed into a low-dimensional latent embedding using a PyTorch model. The generated embeddings can be used for clustering, visualization, classification, and similarity search.

## Key Features
* Uses the provided dataset: `Tweets.csv`
* Tweet text cleaning (lowercasing, URL removal, special character removal, stopword removal)
* Hashtag extraction and inclusion as features
* TF-IDF vectorization (high-dimensional sparse → dense features)
* Deep Autoencoder for non-linear dimensionality reduction (PyTorch)
* Saves embeddings into `embeddings.npy`
* Visualization included:

  * Training loss curve
  * 2D embedding visualization using t-SNE or PCA

## Project Structure

microblog_dimred/
│
├── data/
│   └── Tweets.csv
│
├── src/
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── visualize.py
│
├── requirements.txt
└── README.md

## Requirements

Install dependencies using:
pip install -r requirements.txt

requirements.txt contains:

* numpy
* pandas
* scikit-learn
* torch
* torchvision
* nltk
* matplotlib

## Dataset

Place your dataset file here:
data/Tweets.csv

The dataset must contain a column named:

* text  (tweet content)

If your dataset uses a different column name, update it inside `src/data_preprocessing.py`.

## How to Run

### Step 1: Preprocess and Vectorize Tweets

This cleans the tweet text and generates TF-IDF features (tweet text + hashtags).
python src/data_preprocessing.py --input data/Tweets.csv --output data/vectorized.npz

Output:

* data/vectorized.npz

### Step 2: Train the Autoencoder

This trains the PyTorch autoencoder and saves low-dimensional embeddings.
python src/train.py --data data/vectorized.npz --epochs 50 --batch_size 128 --latent_dim 50 --loss_log loss.csv --output embeddings.npy

Outputs:

* loss.csv (loss per epoch)
* embeddings.npy (reduced embeddings)

### Step 3: Visualize Training and Embeddings

This plots the training loss and projects embeddings into 2D using PCA or t-SNE.
python src/visualize.py --loss_log loss.csv --embeddings embeddings.npy --method tsne

Options:

* --method tsne
* --method pca

## Outputs Explained

* vectorized.npz: processed input features used for training
* loss.csv: training loss values by epoch
* embeddings.npy: reduced-dimensional representation of tweets
* Plots: training loss curve + 2D embedding visualization

## Use Cases

You can use the generated embeddings for:

* Clustering similar tweets
* Similarity search / retrieval
* Downstream ML tasks (sentiment/topic classification)
* Visualization and exploratory analysis

## Technologies Used

Python, PyTorch, scikit-learn, NLTK, pandas, numpy, matplotlib

## Future Improvements

* Replace TF-IDF with BERT embeddings for better semantic encoding
* Add label-based coloring for t-SNE/PCA plots (if labels exist)
* Add clustering evaluation metrics (Silhouette Score, Davies-Bouldin)
* Implement Variational Autoencoder (VAE) for generative latent spaces
