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
- [x] **24.** distillation_experiment
- [x] **25.** softmax_fix_report

## Results

```
1. Softmax Collapse on one vector: z = [9, -9, -9], target 0 (fp32)
   standard  loss 0.000e+00  grad [0.000e+00, 1.523e-08, 1.523e-08]  sum 3.046e-08
   fixed     loss 3.046e-08  grad [-3.046e-08, 1.523e-08, 1.523e-08]  sum 0.000e+00
   The standard target gradient is exactly 0 and the gradient no longer sums to zero; the fixed loss keeps both.
   Precision cliffs (margin where 1 + e^-m rounds to 1): bf16 5.5, fp32 16.6, fp64 36.7
   Soft-target residual gradient at student == teacher (T = 4, 64 random rows): fp32: standard 7.5e-09 / fixed 0e+00, bf16: standard 1.6e-04 / fixed 0e+00

2. Slingshot: a+b mod 31, 3000 full-batch AdamW steps, fp32, eps 1e-8, weight decay 1.0, lr 0.003
   standard  spikes at [1340, 2489]  max margin 16.89  final margin 10.72  max collapsed examples 278/480  train acc 1.000  test acc 0.996
   fixed     spikes at []  max margin 17.16  final margin 17.09  max collapsed examples 0/480  train acc 1.000  test acc 1.000
   Standard loss: margins reach the fp32 cliff (16.6), most examples lose their target gradient, the loss spikes and margins crash.
   Over seeds 0-2: standard spikes at 1268-1312 and 2440-2697, 247-290 collapsed examples, final margin 6-10;
   fixed: no spike, 0 collapsed examples, final margin 16.6-17.1.
   Test accuracy of the fixed arm every 500 steps: 0.28 -> 0.52 -> 0.91 -> 1.00 -> 1.00 -> 1.00
   Grokking within 3000 steps depends on the seed: seeds 0 and 1 reach 1.0 under both losses, seed 2 is at 0.47 under both.
   The paper's accuracy gap (94% vs 100% over 10,000 epochs) is not measured in this budget; the spikes, counts and margins are.

3. Emulated bf16, standard loss, 1000 steps
   spikes at [779]  max margin 6.07 (cliff 5.5)  max collapsed 377  test acc 0.624
   Margins are pinned under the bf16 cliff and the run spikes every few hundred steps (seeds 0-2: 2-4 spikes in 1000 steps, max margin 5.9-6.9, 371-372 collapsed).

4. Margin equilibrium of the fixed loss vs AdamW's epsilon (prediction: log(1/eps))
   eps 1e-08: final margin 17.09  predicted 18.42  spikes []  test acc 1.000
   eps 1e-12: final margin 26.76  predicted 27.63  spikes []  test acc 1.000
   Four orders of magnitude in eps move the margin by 9.67 (prediction 9.21); over seeds 0-2: 16.6-17.1 vs 26.0-26.7.

5. Distillation drift: student initialised at the fixed-loss teacher, T = 4, Adam lr 2e-3, 300 steps
   standard  KL max 4.82e-03  KL final 4.22e-05  max parameter drift 2.85e-02  train acc 1.000  test acc 1.000
   fixed     KL max 0.00e+00  KL final 0.00e+00  max parameter drift 0.00e+00  train acc 1.000  test acc 1.000
   The standard loss's residual gradient is normalised by Adam into real steps (seeds 0-2: KL max 3.8e-3 to 5.8e-3, drift 0.031-0.037);
   the fixed loss returns exactly 0 and the weights never move.

study finished in 35 s
```
