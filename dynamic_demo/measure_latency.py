"""
Step 3/4: Measure real inference latency of Big and Little using ONNX Runtime

What "real latency" means here:
- We run each exported ONNX model using ONNX Runtime on the current machine’s CPU.
- We record timing statistics (median and p90) over many inference runs.
- This is intentionally not a MACs proxy: it is an actual runtime measurement.

Why median and p90:
- Median gives a stable "typical" latency estimate.
- p90 gives a sense of tail latency (useful when thinking about interactive systems).

What this script produces:
- artifacts/latency.json containing:
    {
      "little_ms_median": ...,
      "little_ms_p90": ...,
      "big_ms_median": ...,
      "big_ms_p90": ...
    }

How this connects to dynamic inference:
- The router training objective will include expected latency:
    E[latency] = little_ms + P(escalate)* (big_ms - little_ms)
- Because these latencies are measured, routing is optimized against a device-relevant
  cost rather than a compute proxy.

Note:
- This demo measures on your development machine CPU. Later you can swap the runtime
  and device (e.g., TFLite on Android) with the same structure: export -> measure -> train router.
"""

import time, json
import numpy as np
import onnxruntime as ort

def measure(onnx_path, runs=500, warmup=50):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    x = np.random.randn(1,3,32,32).astype(np.float32)

    # warmup
    for _ in range(warmup):
        sess.run(None, {"input": x})

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {"input": x})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times = np.array(times)
    return float(np.median(times)), float(np.percentile(times, 90))

if __name__ == "__main__":
    little_med, little_p90 = measure("artifacts/little.onnx")
    big_med, big_p90 = measure("artifacts/big.onnx")

    out = {
        "little_ms_median": little_med,
        "little_ms_p90": little_p90,
        "big_ms_median": big_med,
        "big_ms_p90": big_p90
    }
    with open("artifacts/latency.json", "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))
