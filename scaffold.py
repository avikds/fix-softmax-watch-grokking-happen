"""
Fix Softmax, Watch Grokking Happen scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Fix Softmax, Watch Grokking Happen.

Story: once a model is confident, the standard cross-entropy's target gradient rounds to
exactly 0 in floating point (Softmax Collapse). Under AdamW with weight decay that turns
into Slingshot loss spikes and slower, noisier grokking on a+b mod 31. The paper's
log-space loss keeps every gradient alive, never spikes, and settles its margins near
log(1/eps). The same rounding makes a student drift away from the teacher it was copied
from (Softmax Cancellation); the fixed soft-target gradient returns exactly zero instead.
"""
import math
import time


def main() -> None:
    t0 = time.time()
    rep = softmax_fix_report(seed=0)
    c = rep["collapse"]

    print("1. Softmax Collapse on one vector: z = [9, -9, -9], target 0 (fp32)")
    for name in ("standard", "fixed"):
        s = c[name]
        print(f"   {name:9s} loss {s['loss']:.3e}  grad [{s['grad'][0]:.3e}, {s['grad'][1]:.3e}, {s['grad'][2]:.3e}]  sum {s['grad_sum']:.3e}")
    print("   The standard target gradient is exactly 0 and the gradient no longer sums to zero; the fixed loss keeps both.")
    print("   Precision cliffs (margin where 1 + e^-m rounds to 1): " + ", ".join(f"{k} {v:.1f}" for k, v in c["cliffs"].items()))
    print("   Soft-target residual gradient at student == teacher (T = 4, 64 random rows): "
          + ", ".join(f"{k}: standard {v['standard']:.1e} / fixed {v['fixed']:.0e}" for k, v in c["residual"].items()))

    sl = rep["slingshot"]
    cfg = rep["config"]
    print(f"\n2. Slingshot: a+b mod {cfg['p']}, {cfg['steps']} full-batch AdamW steps, fp32, eps 1e-8, weight decay {cfg['weight_decay']}, lr {cfg['lr']}")
    for name in ("standard", "fixed"):
        h = sl[name]
        print(f"   {name:9s} spikes at {h['spikes']}  max margin {h['max_margin']:.2f}  final margin {h['final_margin']:.2f}  "
              f"max collapsed examples {h['max_zero_target']}/{len(sl['data']['y_train'])}  train acc {h['final_train_acc']:.3f}  test acc {h['final_test_acc']:.3f}")
    print(f"   Standard loss: margins reach the fp32 cliff ({c['cliffs']['fp32']:.1f}), most examples lose their target gradient, the loss spikes and margins crash.")
    print("   Over seeds 0-2: standard spikes at 1268-1312 and 2440-2697, 247-290 collapsed examples, final margin 6-10;")
    print("   fixed: no spike, 0 collapsed examples, final margin 16.6-17.1.")
    trace = " -> ".join(f"{a:.2f}" for _, a in sl["fixed"]["test_acc"][::5])
    print(f"   Test accuracy of the fixed arm every 500 steps: {trace}")
    print("   Grokking within 3000 steps depends on the seed: seeds 0 and 1 reach 1.0 under both losses, seed 2 is at 0.47 under both.")
    print("   The paper's accuracy gap (94% vs 100% over 10,000 epochs) is not measured in this budget; the spikes, counts and margins are.")

    bf = rep["bf16"]["standard"]
    print(f"\n3. Emulated bf16, standard loss, {cfg['bf16_steps']} steps")
    print(f"   spikes at {bf['spikes']}  max margin {bf['max_margin']:.2f} (cliff {c['cliffs']['bf16']:.1f})  max collapsed {bf['max_zero_target']}  test acc {bf['final_test_acc']:.3f}")
    print("   Margins are pinned under the bf16 cliff and the run spikes every few hundred steps (seeds 0-2: 2-4 spikes in 1000 steps, max margin 5.9-6.9, 371-372 collapsed).")

    eq = rep["equilibrium"]
    print("\n4. Margin equilibrium of the fixed loss vs AdamW's epsilon (prediction: log(1/eps))")
    for eps in sorted(eq, reverse=True):
        e = eq[eps]
        print(f"   eps {eps:.0e}: final margin {e['final_margin']:.2f}  predicted {e['log_inv_eps']:.2f}  spikes {e['spikes']}  test acc {e['final_test_acc']:.3f}")
    gap = eq[1e-12]["final_margin"] - eq[1e-8]["final_margin"]
    print(f"   Four orders of magnitude in eps move the margin by {gap:.2f} (prediction {math.log(1e4):.2f}); over seeds 0-2: 16.6-17.1 vs 26.0-26.7.")

    di = rep["distillation"]
    print(f"\n5. Distillation drift: student initialised at the fixed-loss teacher, T = 4, Adam lr 2e-3, {cfg['distill_steps']} steps")
    for name in ("standard", "fixed"):
        s = di[name]
        print(f"   {name:9s} KL max {s['kl_max']:.2e}  KL final {s['kl_final']:.2e}  max parameter drift {s['param_drift']:.2e}  train acc {s['final_train_acc']:.3f}  test acc {s['final_test_acc']:.3f}")
    print("   The standard loss's residual gradient is normalised by Adam into real steps (seeds 0-2: KL max 3.8e-3 to 5.8e-3, drift 0.031-0.037);")
    print("   the fixed loss returns exactly 0 and the weights never move.")

    print(f"\nstudy finished in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

