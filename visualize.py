import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import csv


def plot_loss(csv_path):
    epochs, losses = [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            losses.append(float(row['loss']))
    plt.figure()
    plt.plot(epochs, losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()


def plot_embeddings(emb_path, method):
    embs = np.load(emb_path)
    if method=='pca':
        proj = PCA(n_components=2).fit_transform(embs)
    else:
        proj = TSNE(n_components=2).fit_transform(embs)
    plt.figure()
    plt.scatter(proj[:,0], proj[:,1], s=5)
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.title(f'Embeddings - {method.upper()}')
    plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--loss_log', required=True)
    p.add_argument('--embeddings', required=True)
    p.add_argument('--method', choices=['tsne','pca'], default='tsne')
    args = p.parse_args()
    plot_loss(args.loss_log)
    plot_embeddings(args.embeddings, args.method)

if __name__=='__main__':
    main()