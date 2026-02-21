"""
Evaluation + Plots: Compare learned routing against confidence/entropy baselines (CIFAR-10)

What this script evaluates:
1) Static baselines:
   - Little-only: always run the Little model
   - Big-only: always run the Big model

2) Dynamic inference policies (three routing strategies):
   A) Learned router (your trained Router):
      - Compute g(x) = P(escalate_to_big) from early Little features
      - For each threshold tau:
           use_big = (g(x) > tau)

   B) Confidence-threshold baseline (no learning):
      - Compute max_prob = max softmax probability from Little
      - Escalate if Little is NOT confident:
           use_big = (max_prob < t_conf)

   C) Entropy-threshold baseline (no learning):
      - Compute entropy of Little's softmax distribution
      - Escalate if prediction is uncertain:
           use_big = (entropy > t_ent)

Cost / objective proxy used for plotting:
- Expected latency is computed from *measured* latencies (latency.json):
      E[latency] = little_ms + r * (big_ms - little_ms)
  where r is the escalation rate (fraction of samples routed to Big).

Artifacts produced:
- artifacts/acc_vs_latency.png            (3 curves: learned/confidence/entropy)
- artifacts/escalation_vs_threshold.png   (3 curves: learned/confidence/entropy)

Why this is useful for your research gap narrative:
- Shows dynamic inference tradeoffs (accuracy vs cost).
- Includes non-learned confidence baselines, so you can demonstrate that learned
  routing can be compared against (and later improved over) standard threshold gating.
- Uses measured runtime latency rather than MACs as the cost signal.
"""

import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import SmallCNN, Router


def softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)


def entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # probs: (B, C)
    return -(probs * (probs + eps).log()).sum(dim=1)  # (B,)


def main():
    device = torch.device("cpu")

    # Load measured latency numbers
    with open("artifacts/latency.json", "r") as f:
        lat = json.load(f)
    little_ms = lat["little_ms_median"]
    big_ms = lat["big_ms_median"]
    delta_ms = big_ms - little_ms

    # Load checkpoints
    ck_l = torch.load("artifacts/little.pt", map_location=device)
    ck_b = torch.load("artifacts/big.pt", map_location=device)
    ck_r = torch.load("artifacts/router.pt", map_location=device)

    little = SmallCNN(widths=tuple(ck_l["widths"])).to(device)
    big = SmallCNN(widths=tuple(ck_b["widths"])).to(device)
    little.load_state_dict(ck_l["state"])
    big.load_state_dict(ck_b["state"])
    little.eval()
    big.eval()

    router = Router(in_ch=ck_r["in_ch"]).to(device)
    router.load_state_dict(ck_r["state"])
    router.eval()

    # Data
    tf_test = transforms.Compose([transforms.ToTensor()])
    test_ds = datasets.CIFAR10(root="data", train=False, download=True, transform=tf_test)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    # Threshold grids
    taus = np.linspace(0.05, 0.95, 19)        # learned router thresholds
    conf_ts = np.linspace(0.10, 0.99, 19)     # max-prob thresholds
    ent_ts = np.linspace(0.10, 2.50, 19)      # entropy thresholds (range is heuristic)

    # Storage for curves
    learned_accs, learned_lats, learned_escal = [], [], []
    conf_accs, conf_lats, conf_escal = [], [], []
    ent_accs, ent_lats, ent_escal = [], [], []

    with torch.no_grad():
        # ---------
        # Static baselines
        # ---------
        correct_l = 0
        correct_b = 0
        total = 0

        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            l_logits = little(x)
            b_logits = big(x)
            correct_l += (l_logits.argmax(1) == y).sum().item()
            correct_b += (b_logits.argmax(1) == y).sum().item()
            total += y.size(0)

        little_acc = correct_l / total
        big_acc = correct_b / total

        print(f"Little-only acc: {little_acc:.4f}, latency ~ {little_ms:.3f} ms")
        print(f"Big-only acc:    {big_acc:.4f}, latency ~ {big_ms:.3f} ms")

        # ---------
        # A) Learned router sweep
        # ---------
        for tau in taus:
            correct = 0
            total = 0
            escal = 0

            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                l_logits, feat = little(x, return_feat=True)
                b_logits = big(x)

                p_l = softmax_probs(l_logits)
                max_prob = p_l.max(dim=1).values
                ent = entropy(p_l)

                g = router(feat, max_prob, ent)        # (B,) hybrid router
                use_big = (g > tau)
                escal += use_big.sum().item()

                logits = l_logits.clone()
                logits[use_big] = b_logits[use_big]

                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)

            esc_rate = escal / total
            exp_lat = little_ms + esc_rate * delta_ms

            learned_accs.append(correct / total)
            learned_lats.append(exp_lat)
            learned_escal.append(esc_rate)

        # ---------
        # B) Confidence threshold sweep
        # ---------
        for t in conf_ts:
            correct = 0
            total = 0
            escal = 0

            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                l_logits = little(x)
                b_logits = big(x)

                p_l = softmax_probs(l_logits)
                maxp = p_l.max(dim=1).values
                use_big = (maxp < t)             # low confidence -> escalate
                escal += use_big.sum().item()

                logits = l_logits.clone()
                logits[use_big] = b_logits[use_big]

                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)

            esc_rate = escal / total
            exp_lat = little_ms + esc_rate * delta_ms

            conf_accs.append(correct / total)
            conf_lats.append(exp_lat)
            conf_escal.append(esc_rate)

        # ---------
        # C) Entropy threshold sweep
        # ---------
        for t in ent_ts:
            correct = 0
            total = 0
            escal = 0

            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                l_logits = little(x)
                b_logits = big(x)

                p_l = softmax_probs(l_logits)
                ent = entropy(p_l)
                use_big = (ent > t)              # high entropy -> escalate
                escal += use_big.sum().item()

                logits = l_logits.clone()
                logits[use_big] = b_logits[use_big]

                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)

            esc_rate = escal / total
            exp_lat = little_ms + esc_rate * delta_ms

            ent_accs.append(correct / total)
            ent_lats.append(exp_lat)
            ent_escal.append(esc_rate)

    # -----------------------------
    # Plot: accuracy vs expected latency (three routing strategies)
    # -----------------------------
    plt.figure()
    plt.plot(learned_lats, learned_accs, marker="o", label="Learned router (tau)")
    plt.plot(conf_lats, conf_accs, marker="o", label="Confidence threshold (t)")
    plt.plot(ent_lats, ent_accs, marker="o", label="Entropy threshold (t)")
    plt.xlabel("Expected latency (ms)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs expected latency (CIFAR-10)")
    plt.grid(True)
    plt.legend()
    plt.savefig("artifacts/acc_vs_latency.png", dpi=160)

    # -----------------------------
    # Plot: escalation vs threshold (three routing strategies)
    # -----------------------------
    plt.figure()
    plt.plot(taus, learned_escal, marker="o", label="Learned router (tau)")
    plt.plot(conf_ts, conf_escal, marker="o", label="Confidence (t)")
    plt.plot(ent_ts, ent_escal, marker="o", label="Entropy (t)")
    plt.xlabel("Threshold")
    plt.ylabel("Escalation rate (to Big)")
    plt.title("Escalation vs threshold (CIFAR-10)")
    plt.grid(True)
    plt.legend()
    plt.savefig("artifacts/escalation_vs_threshold.png", dpi=160)

    print("Saved plots:")
    print(" - artifacts/acc_vs_latency.png")
    print(" - artifacts/escalation_vs_threshold.png")


if __name__ == "__main__":
    main()
