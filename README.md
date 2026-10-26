# Fix Softmax, Watch Grokking Happen

Cross-entropy has a bug that every framework ships. Once a model is confident, the softmax probability of the target rounds to exactly 1 in floating point, the gradient on the target logit becomes exactly 0 while the other logits still get pushed, and the gradient stops summing to zero. Barron, Gnanapragasam and Tschannen ("A Fix for Softmax and Cross-Entropy", 2026) call this Softmax Collapse, show that it is what makes AdamW produce the Slingshot loss spikes seen in grokking runs, and fix it with a log-space formulation whose every term stays representable. This project reproduces their story end to end in tinygrad on a 1-layer, width-64 transformer learning a+b mod 31 from half of the pairs. You build the dataset and the model, then both losses with explicit gradients: the standard log-softmax cross-entropy, whose target gradient you watch hit exactly 0 past the fp32 precision cliff at a logit margin of 16.6 (5.5 in bf16, which you emulate with rounding casts), and the fixed one, which shifts by the argmax logit and uses a stable log1p so the target gradient is never rounded away and the gradient sums to zero by construction. You then build the soft-target pair for distillation: the standard loss leaves a residual gradient of about 1e-8 even when student and teacher logits are identical (Softmax Cancellation) while the paper's custom backward returns exactly 0. Around them you write a bias-corrected AdamW with an epsilon and weight decay, a TinyJit training step that logs loss, logit margins and the number of examples whose target gradient is exactly 0, and a spike detector. The experiments you run: with the standard loss in fp32, eps 1e-8 and weight decay 1, margins climb to the cliff, 247 to 290 of the 480 training examples lose their target gradient, and a Slingshot spike fires at step 1268 to 1312 and again at 2440 to 2697 (three seeds), crashing the margins to 6 to 10; the fixed loss never spikes, keeps every target gradient alive and settles its margins at 16.6 to 17.1, close to log(1/eps) = 18.4; at eps 1e-12 the fixed margins settle at 26.0 to 26.7, tracking log(1/eps) = 27.6; in emulated bf16 the standard loss is pinned under its 5.5 cliff and spikes every few hundred steps; and a student initialised at a trained teacher drifts under the standard soft-target loss at temperature 4 (parameter drift 0.03 to 0.04 in 300 Adam steps) while under the fixed loss its weights do not move by a single bit. Grokking itself you watch on two of the three seeds, where test accuracy climbs from 26 percent to 100 percent within the 3000 steps under both losses; on the third seed neither loss has left 47 percent yet, so the paper's 10,000-epoch accuracy gap (94 versus 100 percent) is not part of what this budget measures, and the project says so. Along the way you meet a genuine instance of the bug inside tinygrad itself, whose symbolic rewriter turns x/(1+x) into 1-1/(1+x). Every claim was checked across three seeds and every step prints the numbers it measures.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** modular_addition_dataset
- [x] **2.** one_hot_batch
- [x] **3.** GrokTransformer
- [x] **4.** accuracy
- [x] **5.** stable_log1p
- [x] **6.** stable_expm1
- [x] **7.** quantize
- [x] **8.** precision_cliff
- [x] **9.** cross_entropy_standard
- [x] **10.** logit_margin
- [x] **11.** cross_entropy_fixed
- [x] **12.** cross_entropy_stablemax
- [x] **13.** zero_sum_projection
- [x] **14.** soft_cross_entropy_standard
- [x] **15.** soft_cross_entropy_fixed
- [x] **16.** residual_gradient
- [x] **17.** init_adamw_state
- [x] **18.** adamw_update
- [x] **19.** GrokTrainer
- [x] **20.** train_run
- [x] **21.** find_spikes
- [x] **22.** slingshot_experiment
- [x] **23.** equilibrium_experiment

---

Built on Deep-ML.
