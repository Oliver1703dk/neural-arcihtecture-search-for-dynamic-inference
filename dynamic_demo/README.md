# Dynamic Inference Demo (CIFAR-10): Big/Little + Learned Routing + Measured Latency

This folder contains a **minimal, working demo** of *dynamic inference*:

- We train a **Little** model (fast) and a **Big** model (more accurate).
- We train a **Router** that decides, *per input image*, whether to:
  - use the Little model’s prediction (cheap), or
  - **escalate** to the Big model (expensive but more accurate).
- We measure **real latency** (on the current machine) using ONNX Runtime.
- We plot the tradeoff between **accuracy** and **expected latency**.
- We also compare against standard **confidence/entropy threshold baselines**.

This is intended as a **foundation** for the research project (NAS + dynamic inference + hardware awareness), even though the demo is intentionally small and uses CIFAR-10.

---

## What this demo does (plain explanation)

Think of it like customer support:

- **Little model** = quick agent  
- **Big model** = expert agent  
- **Router** = receptionist deciding whether the case needs the expert  
- **Latency measurement** = how long each agent takes  

Some images are easy → Little is enough.  
Some images are hard → escalate to Big.  

This produces a **speed vs accuracy tradeoff** you can choose depending on your budget.

---

## Files (what each script does)

### `models.py`
Defines:
- `SmallCNN`: a simple CNN for CIFAR-10, parameterized by widths (Little vs Big).
- `Router`: a *hybrid* learned router that uses:
  - early features from Little,
  - Little’s confidence (max softmax probability),
  - Little’s uncertainty (entropy),
  to output `g(x)` = probability of escalating to Big.

### `train_supernet.py`
Trains:
- **Big** and **Little** classifiers on CIFAR-10.
- Little is trained with **distillation** from Big (improves Little and helps routing).
Outputs:
- `artifacts/little.pt`
- `artifacts/big.pt`
- `artifacts/big_best.pt`

### `export_onnx.py`
Exports both models into ONNX:
- `artifacts/little.onnx`
- `artifacts/big.onnx`

### `measure_latency.py`
Runs ONNX Runtime inference repeatedly and measures:
- median latency
- p90 latency  
Outputs:
- `artifacts/latency.json`

### `train_router.py`
Freezes Big and Little and trains only the **Router** with a latency-aware objective:

- classification loss (be correct)
- + latency penalty (don’t escalate too often)

Outputs:
- `artifacts/router.pt`

### `eval_routing.py`
Evaluates and plots:

1) **Static baselines**:
- Little-only (always run Little)
- Big-only (always run Big)

2) **Dynamic routing policies**:
- Learned Router (threshold `tau` on `g(x)`)
- Confidence threshold baseline
- Entropy threshold baseline

Outputs:
- `artifacts/acc_vs_latency.png`
- `artifacts/escalation_vs_threshold.png`

---

## Quickstart (step-by-step)

### 1) Setup
```bash
python3 -m venv .venv
source .venv/bin/activate

pip install torch torchvision torchaudio
pip install onnx onnxruntime onnxscript numpy tqdm matplotlib
mkdir -p artifacts
```

---
## Current status (what is already implemented)

[X] Big + Little classifiers trained on CIFAR-10
[X] Distillation from Big → Little
[X] Learned routing (hybrid: early features + confidence + entropy)
[X] Confidence/entropy routing baselines (non-learned)
[X] Measured latency using a real inference runtime (ONNX Runtime)
[X] Plots for accuracy–latency tradeoff and escalation behavior

### What is missing (checklist vs research gaps)
#### Gap: No joint big/little NAS (unified supernet + coupled search)
[] No NAS search over architectures (widths/structure are manually chosen)
[] No single weight-sharing unified supernet where big and little are subnets
[] No coupled NAS objective optimizing the (big,little) pair jointly

#### Gap: Limited routing mechanisms (NAS + routing co-optimization)
[] Router is learned, but not co-searched with architecture (no NAS+routing joint search)
[] Only a single routing decision point (no multi-stage routing)
[] No early exits / intermediate classifiers

#### Gap: Incomplete hardware awareness (real edge latency/energy)
[] Latency is measured, but only on desktop CPU so far (not Raspberry Pi / mobile runtime)
[] No energy measurement
[] No memory footprint / peak RAM constraint
[] No quantized (int8) profiling

#### Gap: Width-variable training without NAS optimization
[] Not a true slimmable/US-Net single model (one set of weights for many widths)
[] Width variation is currently stage-level (not per-layer)
[] No search discovering the best per-layer width patterns for Big and Little

#### Gap: No NAS + dynamic inference + compiler co-optimization
[] No compiler/schedule search (TVM/TensorRT/etc.)
[] Cost model uses one measured latency per model (not schedule-dependent per path)
[] No demonstration that compiler tuning changes routing / optimal policy

#### Evaluation completeness (nice-to-have for later)
[] Additional datasets beyond CIFAR-10 (for edge realism)
[] Calibration metrics (e.g., ECE), per-class routing analysis
[] Fair static baseline at similar latency budget