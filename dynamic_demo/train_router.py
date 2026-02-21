"""
Train the hybrid router (features + confidence) under a latency-regularized objective.

Requires:
- artifacts/little.pt and artifacts/big.pt (from train_supernet.py)
- artifacts/latency.json (from measure_latency.py)
- models.Router signature: forward(feat, max_prob, ent)

Produces:
- artifacts/router.pt
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models import SmallCNN, Router


def entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return -(probs * (probs + eps).log()).sum(dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda_latency", type=float, default=0.02)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    # Load measured latency
    with open("artifacts/latency.json", "r") as f:
        lat = json.load(f)
    little_ms = lat["little_ms_median"]
    big_ms = lat["big_ms_median"]
    delta_ms = big_ms - little_ms

    # Load trained Big/Little
    ck_l = torch.load("artifacts/little.pt", map_location=device)
    ck_b = torch.load("artifacts/big.pt", map_location=device)

    little = SmallCNN(widths=tuple(ck_l["widths"])).to(device)
    big = SmallCNN(widths=tuple(ck_b["widths"])).to(device)
    little.load_state_dict(ck_l["state"])
    big.load_state_dict(ck_b["state"])

    # Freeze Big/Little
    little.eval()
    big.eval()
    for p in little.parameters():
        p.requires_grad = False
    for p in big.parameters():
        p.requires_grad = False

    # Hybrid router uses little stage-1 width as feature dimension
    in_ch = tuple(ck_l["widths"])[0]
    router = Router(in_ch=in_ch).to(device)

    # Data
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    tf_test = transforms.Compose([transforms.ToTensor()])

    train_ds = datasets.CIFAR10(root="data", train=True, download=True, transform=tf_train)
    test_ds = datasets.CIFAR10(root="data", train=False, download=True, transform=tf_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    opt = torch.optim.Adam(router.parameters(), lr=args.lr)
    ce = nn.NLLLoss()

    for epoch in range(1, args.epochs + 1):
        router.train()
        pbar = tqdm(train_loader, desc=f"Router epoch {epoch}/{args.epochs}")

        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                l_logits, feat = little(x, return_feat=True)
                b_logits = big(x)

            p_l = F.softmax(l_logits, dim=1)
            p_b = F.softmax(b_logits, dim=1)

            max_prob = p_l.max(dim=1).values
            ent = entropy(p_l)

            g = router(feat, max_prob, ent)  # (B,)

            # Differentiable mixture for router training
            p_mix = (1 - g).unsqueeze(1) * p_l + g.unsqueeze(1) * p_b
            loss_cls = ce(torch.log(p_mix + 1e-8), y)

            # Expected latency penalty (ms)
            exp_lat = little_ms + g.mean() * delta_ms
            loss = loss_cls + args.lambda_latency * exp_lat

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.set_postfix(
                loss=float(loss.detach().cpu()),
                g=float(g.mean().detach().cpu()),
                lat=float(exp_lat.detach().cpu()),
            )

        # Quick eval at tau=0.5
        router.eval()
        tau = 0.5
        correct = 0
        total = 0
        escal = 0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                l_logits, feat = little(x, return_feat=True)
                b_logits = big(x)

                p_l = F.softmax(l_logits, dim=1)
                max_prob = p_l.max(dim=1).values
                ent = entropy(p_l)

                g = router(feat, max_prob, ent)
                use_big = (g > tau)
                escal += use_big.sum().item()

                logits = l_logits.clone()
                logits[use_big] = b_logits[use_big]

                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)

        print(f"tau={tau} test_acc={correct/total:.4f} escalation={escal/total:.3f}")

    os.makedirs("artifacts", exist_ok=True)
    torch.save({"state": router.state_dict(), "in_ch": in_ch, "hybrid": True}, "artifacts/router.pt")
    print("Saved artifacts/router.pt")


if __name__ == "__main__":
    main()
