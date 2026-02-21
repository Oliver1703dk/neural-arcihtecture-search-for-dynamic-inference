"""
Step 1/4: Co-train a Big and a Little network on CIFAR-10 (with distillation)

What this script does:
- Trains two networks:
    (A) Big model: higher capacity (wider) -> higher accuracy, higher latency
    (B) Little model: lower capacity (narrower) -> lower latency, lower accuracy
- Both are trained in the SAME training loop (co-training), producing a consistent
  pair of models suitable for dynamic inference.

Why distillation is used:
- The Big model serves as a stronger teacher. The Little model is trained with:
    * standard cross-entropy to ground truth labels, PLUS
    * knowledge distillation (KL divergence) from the Big model’s softened logits
- Distillation improves the Little model’s accuracy and, importantly, helps its
  internal features become more informative, which tends to improve routing quality.

Key losses:
- Big loss:
    L_big = CE(big_logits, y)
- Little loss (CE + KD):
    L_little = (1 - alpha) * CE(little_logits, y)
               + alpha * KL(softmax(big/T) || softmax(little/T)) * T^2

The combined optimization:
    L_total = L_big + L_little

Artifacts produced:
- artifacts/little.pt  (state dict + widths)
- artifacts/big.pt     (state dict + widths)
- artifacts/big_best.pt (best Big checkpoint by test accuracy)

Why this satisfies the "minimum demo" requirements:
- Joint big/little training (co-trained pair).
- Per-stage width configuration (not a global multiplier), reflecting the kind of
  structural flexibility NAS would optimize in later versions.
- Provides stable models for the next step: learned routing with a latency-aware
  objective and real measured latency.
"""

import argparse, os
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models import SmallCNN

def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--little", type=str, default="24,48,96")
    ap.add_argument("--big", type=str, default="48,96,192")
    ap.add_argument("--kd_temp", type=float, default=4.0)
    ap.add_argument("--kd_alpha", type=float, default=0.7)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device=="cpu" else "cpu")

    little_w = tuple(map(int, args.little.split(",")))
    big_w    = tuple(map(int, args.big.split(",")))

    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    tf_test = transforms.Compose([transforms.ToTensor()])

    train_ds = datasets.CIFAR10(root="data", train=True, download=True, transform=tf_train)
    test_ds  = datasets.CIFAR10(root="data", train=False, download=True, transform=tf_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    little = SmallCNN(widths=little_w).to(device)
    big    = SmallCNN(widths=big_w).to(device)

    opt = torch.optim.Adam(list(little.parameters()) + list(big.parameters()), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    def kd_loss(student_logits, teacher_logits, T):
        # KL( softmax(teacher/T) || softmax(student/T) ) * T^2
        p_t = F.softmax(teacher_logits / T, dim=1)
        log_p_s = F.log_softmax(student_logits / T, dim=1)
        return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        little.train(); big.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            big_logits = big(x)
            little_logits = little(x)

            loss_big = ce(big_logits, y)
            loss_lit = (1 - args.kd_alpha) * ce(little_logits, y) + args.kd_alpha * kd_loss(little_logits, big_logits.detach(), args.kd_temp)
            loss = loss_big + loss_lit

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.set_postfix(loss=float(loss.detach().cpu()))

        # eval
        little.eval(); big.eval()
        with torch.no_grad():
            acc_l, acc_b = 0.0, 0.0
            n = 0
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                l = little(x)
                b = big(x)
                bs = y.size(0)
                acc_l += (l.argmax(1) == y).float().sum().item()
                acc_b += (b.argmax(1) == y).float().sum().item()
                n += bs
            acc_l /= n; acc_b /= n

        print(f"Test acc: little={acc_l:.4f} big={acc_b:.4f}")

        # save best big (for stability) + always save both latest
        os.makedirs("artifacts", exist_ok=True)
        torch.save({"state": little.state_dict(), "widths": little_w}, "artifacts/little.pt")
        torch.save({"state": big.state_dict(), "widths": big_w}, "artifacts/big.pt")

        if acc_b > best:
            best = acc_b
            torch.save({"state": big.state_dict(), "widths": big_w}, "artifacts/big_best.pt")

if __name__ == "__main__":
    main()
