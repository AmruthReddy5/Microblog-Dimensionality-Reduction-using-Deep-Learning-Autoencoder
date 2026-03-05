import argparse
import numpy as np
import torch
import csv
from torch.utils.data import DataLoader, TensorDataset
from model import Autoencoder


def main(args):
    data = np.load(args.data)['features']
    X = torch.tensor(data, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X, X), batch_size=args.batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Autoencoder(input_dim=X.shape[1], latent_dim=args.latent_dim).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    with open(args.loss_log, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss'])
        for epoch in range(1, args.epochs+1):
            total_loss = 0
            for bx, _ in loader:
                bx = bx.to(device)
                optimizer.zero_grad()
                recon, _ = model(bx)
                loss = criterion(recon, bx)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg = total_loss/len(loader)
            writer.writerow([epoch, avg])
            print(f"Epoch {epoch}/{args.epochs}, Loss: {avg:.4f}")

    model.eval()
    with torch.no_grad():
        emb = model.encoder(X.to(device))
        np.save(args.output, emb.cpu().numpy())
        print(f"Saved embeddings to {args.output}")

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--latent_dim', type=int, default=32)
    p.add_argument('--loss_log', default='loss.csv')
    p.add_argument('--output', default='embeddings.npy')
    args = p.parse_args()
    main(args)