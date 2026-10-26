"""
Fix Softmax, Watch Grokking Happen

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - modular_addition_dataset
import numpy as np

def modular_addition_dataset(p, train_frac=0.5, seed=0):
    # Total number of ordered pairs (a, b).
    # Pair index i corresponds to a = i // p and b = i % p.
    n_total = p * p
    pair_indices = np.arange(n_total, dtype=np.int64)

    # Generate one seeded permutation to split the pairs reproducibly.
    perm = np.random.default_rng(seed).permutation(p * p)

    # Use the specified rounding rule to determine the training-set size.
    n_train = int(round(train_frac * p * p))

    train_indices = perm[:n_train]
    test_indices = perm[n_train:]

    def encode(indices):
        # Recover a and b from their a-major pair indices.
        a = indices // p
        b = indices % p

        # Encode each pair as [a, b, p], where p represents the "=" token.
        x = np.column_stack(
            (
                a,
                b,
                np.full(indices.shape, p, dtype=np.int64),
            )
        ).astype(np.int64, copy=False)

        # Calculate the modular-addition labels.
        y = ((a + b) % p).astype(np.int64, copy=False)

        return x, y

    # Encode the training and test splits.
    x_train, y_train = encode(train_indices)
    x_test, y_test = encode(test_indices)

    return {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "p": p,
    }

# Step 2 - one_hot_batch
def one_hot_batch(x, y, p):
    # Encode the three input tokens using a vocabulary of p + 1 symbols.
    # The extra symbol, p, represents the "=" token.
    inputs = np.eye(p + 1, dtype=np.float32)[x]

    # Encode the labels using p possible output classes.
    targets = np.eye(p, dtype=np.float32)[y]

    return inputs.astype(np.float32, copy=False), targets.astype(
        np.float32, copy=False
    )

# Step 3 - GrokTransformer
import math
from tinygrad import Tensor

class GrokTransformer:
    def __init__(self, p, d_model=64, n_heads=4, seed=0):
        self.p = p
        self.d_model = d_model
        self.n_heads = n_heads

        assert d_model % n_heads == 0, (
            "d_model must be divisible by n_heads."
        )

        # Head dimension.
        d = d_model
        dh = d // n_heads

        # Use a seeded random number generator for reproducible initialization.
        rng = np.random.default_rng(seed)

        # Initialize weights in the exact order specified by the task.
        def init_weight(shape, scale):
            values = rng.normal(0.0, scale, size=shape).astype(np.float32)
            return Tensor(values)

        # Token embeddings and positional embeddings.
        self.emb = init_weight((p + 1, d), 1.0 / math.sqrt(d))
        self.pos = init_weight((3, d), 1.0 / math.sqrt(d))

        # Query, key, value, and output projection weights.
        self.wq = init_weight((d, d), 1.0 / math.sqrt(d))
        self.wk = init_weight((d, d), 1.0 / math.sqrt(d))
        self.wv = init_weight((d, d), 1.0 / math.sqrt(d))
        self.wo = init_weight((d, d), 1.0 / math.sqrt(d))

        # Two-layer feed-forward network.
        self.w1 = init_weight((d, 4 * d), 1.0 / math.sqrt(d))
        self.w2 = init_weight((4 * d, d), 1.0 / math.sqrt(4 * d))

        # Final projection to the p modular-addition classes.
        self.wout = init_weight((d, p), 1.0 / math.sqrt(d))

        # Initialize biases as realized, contiguous zero tensors.
        self.b1 = Tensor.zeros(4 * d).contiguous().realize()
        self.b2 = Tensor.zeros(d).contiguous().realize()
        self.bout = Tensor.zeros(p).contiguous().realize()

    def __call__(self, inputs):
        # inputs has shape (n, 3, p + 1).
        n = inputs.shape[0]
        d = self.d_model
        dh = d // self.n_heads

        # Embed tokens and add positional embeddings.
        # h has shape (n, 3, d).
        h = inputs @ self.emb + self.pos

        # The query comes only from the final position.
        # q has shape (n, n_heads, 1, dh).
        q = (h[:, -1] @ self.wq).reshape(
            n, self.n_heads, 1, dh
        )

        # Keys and values come from all three positions.
        # Transpose to (n, n_heads, 3, dh).
        k = (h @ self.wk).reshape(
            n, 3, self.n_heads, dh
        ).transpose(1, 2)

        v = (h @ self.wv).reshape(
            n, 3, self.n_heads, dh
        ).transpose(1, 2)

        # Scaled dot-product attention over the three keys.
        # Attention scores have shape (n, n_heads, 1, 3).
        scores = (q @ k.transpose(2, 3)) / math.sqrt(dh)
        attn_weights = scores.softmax(axis=-1)

        # Combine the values from all attention heads.
        # Result has shape (n, d).
        attention = (attn_weights @ v).reshape(n, d)

        # Attention output projection and residual connection.
        x = h[:, -1] + attention @ self.wo

        # Two-layer ReLU feed-forward network with a residual connection.
        hidden = (x @ self.w1 + self.b1).relu()
        x = x + hidden @ self.w2 + self.b2

        # Return unnormalized class logits with shape (n, p).
        return x @ self.wout + self.bout

# Step 4 - accuracy
def accuracy(model, x, y, p):
    # Convert input tokens and labels into float32 one-hot arrays.
    inputs, targets = one_hot_batch(x, y, p)

    # Run a standard forward pass without JIT compilation or backpropagation.
    logits = model(Tensor(inputs))

    # Select the class with the highest logit for each example.
    predictions = logits.argmax(axis=-1).numpy()

    # Recover the ground-truth class labels from the one-hot targets.
    actual = np.argmax(targets, axis=-1)

    # Return the fraction of correctly classified examples as a Python float.
    return float(np.mean(predictions == actual))

# Step 5 - stable_log1p
def stable_log1p(x):
    # Taylor approximation of log(1 + x) for small |x|.
    # This avoids forming 1 + x, which may round to exactly 1
    # when x is very small in floating-point arithmetic.
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x

    poly = x - x2 / 2 + x3 / 3 - x4 / 4

    # Direct computation for values that are not close to zero.
    direct = (1 + x).log()

    # Select the Taylor approximation for |x| < 1e-2.
    # Both branches are evaluated, but neither causes an error
    # for the inputs targeted by this function.
    return (x.abs() < 1e-2).where(poly, direct)

# Step 6 - stable_expm1
def stable_expm1(x):
    # Taylor approximation of exp(x) - 1 for small |x|.
    # This avoids cancellation when exp(x) rounds close to 1.
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x

    poly = x + x2 / 2 + x3 / 6 + x4 / 24

    # Direct computation for values that are not close to zero.
    direct = x.exp() - 1

    # Use the polynomial when |x| < 1e-2.
    return (x.abs() < 1e-2).where(poly, direct)

# Step 7 - quantize
def quantize(x, precision):
    # Keep the tensor unchanged for full float32 precision.
    if precision == "fp32":
        return x

    if precision == "bf16":
        # Reinterpret float32 values as their 32-bit unsigned bit patterns.
        bits = x.bitcast(dtypes.uint32)

        # Round to nearest, ties to even.
        # Keep the upper 16 bits and use the lowest retained bit to
        # correctly handle halfway cases.
        rounding_bias = 0x7FFF + ((bits >> 16) & 1)
        rounded_bits = (bits + rounding_bias) & 0xFFFF0000

        # Reinterpret the rounded bit patterns as float32 values.
        rounded = rounded_bits.bitcast(dtypes.float32)

        # Straight-through estimator:
        # Forward value equals the rounded value, while the gradient
        # passes through as the identity.
        return x + (rounded - x).detach()

    # Reject unsupported precision modes.
    raise ValueError(f"Unsupported precision: {precision}")

# Step 8 - precision_cliff
def precision_cliff(precision):
    # Number of significand bits, including the implicit leading bit.
    significand_bits = {
        "bf16": 8,
        "fp16": 11,
        "fp32": 24,
        "fp64": 53,
    }

    # Reject unsupported precision formats.
    if precision not in significand_bits:
        raise ValueError(f"Unsupported precision: {precision}")

    # The cliff is bits * ln(2).
    return float(significand_bits[precision] * math.log(2))

# Step 9 - cross_entropy_standard
def cross_entropy_standard(logits, targets, precision="fp32"):
    # Q emulates the requested floating-point precision.
    Q = lambda v: quantize(v, precision)

    # Quantize the input logits.
    z = Q(logits)

    # Subtract the maximum logit for numerical stability.
    z_max = z.max(axis=-1, keepdim=True)
    zs = Q(z - z_max)

    # Compute and quantize the exponentials and their sum.
    e = Q(zs.exp())
    s = Q(e.sum(axis=-1, keepdim=True))

    # Compute log-probabilities, quantizing the logarithm and subtraction.
    log_s = Q(s.log())
    log_probs = Q(zs - log_s)

    # Compute the softmax probabilities.
    probs = Q(e / s)

    # Cross-entropy loss for each example, with shape (n,).
    loss = -(log_probs * targets).sum(axis=-1)

    # Exact analytical gradient of the standard cross-entropy
    # with respect to the logits, with shape (n, K).
    grad = probs - targets

    return loss, grad

# Step 10 - logit_margin
def logit_margin(logits, targets):
    # Extract the target logit for each row.
    # For one-hot targets, this selects the logit of the correct class.
    z_t = (logits * targets).sum(axis=-1)

    # Mask the target class so that only the other logits contribute.
    # The large negative value makes its exponential negligible.
    others = (targets > 0).where(-1e30, logits)

    # Compute the log-sum-exp of all non-target logits.
    logsumexp_others = others.logsumexp(axis=-1)

    # Return the target logit minus the log-sum-exp of the other logits.
    # The resulting Tensor has shape (n,).
    return z_t - logsumexp_others

# Step 11 - cross_entropy_fixed
def cross_entropy_fixed(logits, targets, precision="fp32"):
    # Quantization function for the requested floating-point precision.
    Q = lambda v: quantize(v, precision)

    # Quantize the input logits.
    z = Q(logits)

    # Extract the target logit for each example.
    # keepdim=True preserves shape (n, 1) for broadcasting.
    z_t = (z * targets).sum(axis=-1, keepdim=True)

    # Shift by the target logit so that the target entry is exactly zero.
    zs = Q(z - z_t)

    # Compute exponentials and explicitly zero the target entry.
    e = Q(zs.exp()) * (1 - targets)

    # Sum the exponentials of the non-target logits.
    # This equals exp(-margin).
    sm = Q(e.sum(axis=-1, keepdim=True))

    # Compute the loss using stable log1p to preserve tiny positive values.
    # Reshape from (n, 1) to (n,).
    loss = stable_log1p(sm).reshape(logits.shape[0])

    # Compute the gradient as ONE expression.
    # Keeping the combined numerator avoids a numerically unsafe rewrite
    # of the target gradient into 1 - 1/(1 + sm).
    grad = (e - targets * sm) / (1 + sm)

    return loss, grad

# Step 12 - cross_entropy_stablemax
def cross_entropy_stablemax(logits, targets):
    # StableMax activation:
    # s(x) = x + 1 for x >= 0, otherwise 1 / (1 - x).
    nonnegative = logits >= 0
    s = nonnegative.where(logits + 1, 1 / (1 - logits))

    # Derivative of the StableMax activation.
    # s'(x) = 1 for x >= 0, otherwise 1 / (1 - x)^2.
    ds = nonnegative.where(
        Tensor.ones_like(logits),
        1 / ((1 - logits) * (1 - logits)),
    )

    # Sum the StableMax activations across classes.
    # Shape: (n, 1).
    S = s.sum(axis=-1, keepdim=True)

    # Compute the target activation sum.
    # For one-hot targets, this selects the target class activation.
    s_t = (s * targets).sum(axis=-1, keepdim=True)

    # StableMax cross-entropy loss for each example.
    # Reshape from (n, 1) to (n,).
    loss = (S.log() - s_t.log()).reshape(logits.shape[0])

    # Analytical gradient with respect to the logits.
    grad = ds / S - targets * ds / s_t

    return loss, grad

# Step 13 - zero_sum_projection
def zero_sum_projection(grad):
    # Compute the mean gradient across the class dimension for each example.
    mean_grad = grad.mean(axis=-1, keepdim=True)

    # Subtract the mean so that each example's gradients sum to zero.
    return grad - mean_grad

# Step 14 - soft_cross_entropy_standard
def soft_cross_entropy_standard(
    student_logits,
    teacher_logits,
    temperature=1.0,
    precision="fp32",
):
    # Quantization function for the requested precision.
    Q = lambda v: quantize(v, precision)

    # Compute the teacher's soft probability distribution.
    p = Q((teacher_logits / temperature).softmax(axis=-1))

    # Compute the student's log-probability distribution.
    log_q = Q((student_logits / temperature).log_softmax(axis=-1))

    # Standard soft-target cross-entropy, one loss per example.
    loss = -(p * log_q).sum(axis=-1)

    # Return None for the hand-written gradient so that tinygrad
    # computes the gradient through automatic differentiation.
    return loss, None

# Step 15 - soft_cross_entropy_fixed
def soft_cross_entropy_fixed(
    student_logits,
    teacher_logits,
    temperature=1.0,
    precision="fp32",
):
    T = temperature

    # Quantize values to the requested emulated precision.
    Q = lambda v: quantize(v, precision)

    # Preserve exactly the standard soft-target loss for the forward pass.
    loss, _ = soft_cross_entropy_standard(
        student_logits,
        teacher_logits,
        temperature=T,
        precision=precision,
    )

    # Teacher and student probability distributions.
    p = Q((teacher_logits / T).softmax(axis=-1))
    q = Q((student_logits / T).softmax(axis=-1))

    # Difference between student and teacher logits.
    delta = Q((student_logits - teacher_logits) / T)

    # Center the difference using the teacher distribution.
    d = delta - (p * delta).sum(axis=-1, keepdim=True)

    # Stable log-space correction using log1p and expm1.
    s = d - stable_log1p(
        (p * stable_expm1(d)).sum(axis=-1, keepdim=True)
    )

    # Construct the corrected probability difference.
    u = p * stable_expm1(s)

    # Custom gradient: the subtraction removes the row-sum component,
    # ensuring the gradient sums to zero across classes.
    grad = (u - q * u.sum(axis=-1, keepdim=True)) / T

    return loss, grad

# Step 16 - residual_gradient
def residual_gradient(teacher_logits, temperature=1.0, precision="fp32"):
    # Create an independent student tensor with values identical to the teacher.
    student = Tensor(teacher_logits.numpy().copy())

    # Standard soft-target cross-entropy: let tinygrad compute the gradient
    # automatically with respect to the student logits.
    standard_loss, _ = soft_cross_entropy_standard(
        student,
        teacher_logits,
        temperature=temperature,
        precision=precision,
    )

    standard_grad = standard_loss.sum().gradient(student)[0]

    # Measure the maximum absolute autodiff gradient.
    standard_residual = float(
        np.max(np.abs(standard_grad.numpy()))
    )

    # Fixed soft-target cross-entropy: use the explicitly returned gradient.
    _, fixed_grad = soft_cross_entropy_fixed(
        student,
        teacher_logits,
        temperature=temperature,
        precision=precision,
    )

    # Measure the maximum absolute custom gradient.
    fixed_residual = float(
        np.max(np.abs(fixed_grad.numpy()))
    )

    return {
        "standard": standard_residual,
        "fixed": fixed_residual,
    }

# Step 17 - init_adamw_state
def init_adamw_state(params):
    # Initialize first-moment buffers, one for each parameter.
    m = [
        Tensor.zeros(*p.shape).contiguous().realize()
        for p in params
    ]

    # Initialize second-moment buffers, one for each parameter.
    v = [
        Tensor.zeros(*p.shape).contiguous().realize()
        for p in params
    ]

    # Initialize the AdamW step counter as a realized scalar tensor.
    t = Tensor([0.0]).contiguous().realize()

    return {
        "m": m,
        "v": v,
        "t": t,
    }

# Step 18 - adamw_update
def adamw_update(
    params,
    grads,
    state,
    lr,
    weight_decay=1.0,
    eps=1e-8,
    betas=(0.9, 0.98),
):
    b1, b2 = betas

    # Increment the optimizer step counter.
    t = state["t"].assign(state["t"] + 1)

    # Compute bias-correction factors.
    bc1 = 1 - b1 ** t
    bc2 = 1 - b2 ** t

    # Update each parameter and its first- and second-moment buffers.
    for q, g, m, v in zip(
        params, grads, state["m"], state["v"]
    ):
        m.assign(b1 * m + (1 - b1) * g)
        v.assign(b2 * v + (1 - b2) * g * g)

        # Bias-corrected Adam update plus decoupled weight decay.
        update = (
            (m / bc1) / ((v / bc2).sqrt() + eps)
            + weight_decay * q
        )

        q.assign(q - lr * update)

    # Return every updated tensor so the caller can realize them together.
    return [t] + state["m"] + state["v"] + list(params)

# Step 19 - GrokTrainer
from tinygrad import TinyJit
from tinygrad.nn.state import get_parameters


class GrokTrainer:
    def __init__(
        self,
        model,
        loss_fn,
        lr=3e-3,
        weight_decay=1.0,
        eps=1e-8,
        precision="fp32",
    ):
        # Store model, loss function, and optimizer configuration.
        self.model = model
        self.loss_fn = loss_fn
        self.weight_decay = weight_decay
        self.eps = eps
        self.precision = precision

        # Collect trainable parameters and initialize AdamW state.
        self.params = get_parameters(model)
        self.state = init_adamw_state(self.params)

        # Keep the learning rate as a Tensor so it can change without
        # requiring recompilation of the jitted training step.
        self.lr_t = Tensor([float(lr)]).contiguous().realize()

        # Compile the full-batch training step and prediction function.
        self._step = TinyJit(self._train_step)
        self._predict = TinyJit(self._predict_fn)

        self.last_logits = None

    def _train_step(self, inputs, targets, lr_t):
        # Forward pass and loss computation.
        logits = self.model(inputs)
        loss, grad = self.loss_fn(
            logits, targets, self.precision
        )
        n = inputs.shape[0]

        if grad is None:
            # Standard loss: compute gradients through autodiff.
            # The loss is averaged for parameter updates, but grad is
            # scaled back to the per-example gradient for diagnostics.
            gs = (loss.sum() / n).gradient(
                logits, *self.params
            )

            grad = gs[0] * n
            grads = gs[1:]

        else:
            # Custom loss: construct a surrogate scalar whose gradient
            # with respect to logits is the supplied gradient divided
            # by the batch size. Detaching grad prevents differentiation
            # through the explicitly supplied gradient itself.
            surrogate = (
                logits * (grad / n).detach()
            ).sum()

            grads = surrogate.gradient(*self.params)

        # Apply one bias-corrected AdamW update.
        updated = adamw_update(
            self.params,
            grads,
            self.state,
            lr_t,
            self.weight_decay,
            self.eps,
        )

        # Compute each example's target-logit margin.
        margin = logit_margin(logits, targets)

        # Count examples whose target-logit gradient is exactly zero.
        n_zero = (
            ((grad * targets).sum(axis=-1) == 0).sum()
        )

        # Diagnostics:
        # [mean loss, mean margin, minimum margin, collapsed examples].
        stats = Tensor.stack(
            loss.mean(),
            margin.mean(),
            margin.min(),
            n_zero.float(),
        )

        # Realize results and all updated optimizer/parameter buffers.
        Tensor.realize(stats, logits, *updated)

        return stats, logits

    def _predict_fn(self, inputs):
        # Return the predicted class for each example.
        return self.model(inputs).argmax(axis=-1).realize()

    def step(self, inputs, targets):
        # Execute one compiled full-batch training step.
        stats, logits = self._step(
            inputs, targets, self.lr_t
        )

        # Keep the logits generated by this training step.
        self.last_logits = logits

        # Return the four diagnostics as a NumPy array.
        return stats.numpy()

    def predict(self, inputs):
        # Execute the compiled prediction function.
        return self._predict(inputs).numpy()

    def snapshot_logits(self, inputs, targets):
        # Execute one training step with zero learning rate.
        # This computes the model logits without changing its parameters.
        zero_lr = Tensor([0.0]).contiguous().realize()

        _, logits = self._step(
            inputs, targets, zero_lr
        )

        # Copy the logits before resetting the optimizer state.
        snapshot = logits.numpy().copy()

        # Reset the existing moment buffers in place.
        # Keeping the same buffers is important for the compiled step.
        for m in self.state["m"]:
            m.assign(Tensor.zeros(*m.shape))

        for v in self.state["v"]:
            v.assign(Tensor.zeros(*v.shape))

        self.state["t"].assign(Tensor([0.0]))

        # Ensure that all reset operations have completed.
        Tensor.realize(
            *self.state["m"],
            *self.state["v"],
            self.state["t"],
        )

        return snapshot

# Step 20 - train_run
def train_run(
    model,
    loss_fn,
    data,
    steps,
    lr=3e-3,
    weight_decay=1.0,
    eps=1e-8,
    precision="fp32",
    eval_every=100,
):
    if eval_every <= 0:
        raise ValueError("eval_every must be a positive integer.")

    p = data["p"]

    # One-hot encode the training inputs and labels.
    x_train, y_train = one_hot_batch(
        data["x_train"], data["y_train"], p
    )

    # Realize the full training batch as tinygrad tensors.
    X = Tensor(x_train).realize()
    T = Tensor(y_train).realize()

    # One-hot encode the test inputs. The test labels remain integer labels.
    x_test, _ = one_hot_batch(
        data["x_test"], data["y_test"], p
    )
    X_test = Tensor(x_test).realize()

    # Initialize the trainer and its AdamW state.
    trainer = GrokTrainer(
        model,
        loss_fn,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        precision=precision,
    )

    # Histories for training diagnostics and test accuracy.
    loss = []
    margin_mean = []
    margin_min = []
    n_zero_target = []
    test_acc = []

    # Full-batch training loop.
    for s in range(steps):
        stats = trainer.step(X, T)

        # The four statistics are:
        # [mean loss, mean margin, minimum margin, zero target gradients].
        loss.append(float(stats[0]))
        margin_mean.append(float(stats[1]))
        margin_min.append(float(stats[2]))
        n_zero_target.append(int(stats[3]))

        # Evaluate at regular intervals and on the final training step.
        if (s + 1) % eval_every == 0 or s == steps - 1:
            predictions = trainer.predict(X_test)
            test_accuracy = float(
                (predictions == data["y_test"]).mean()
            )
            test_acc.append((s + 1, test_accuracy))

    # Compute the final evaluation summaries.
    final_test_acc = (
        test_acc[-1][1] if test_acc else float("nan")
    )

    # The jitted predictor uses the test-batch shape. Evaluate the training
    # split using the existing plain-forward accuracy function instead.
    final_train_acc = accuracy(
        model,
        data["x_train"],
        data["y_train"],
        p,
    )

    # Average the last 100 recorded mean margins, or all if fewer.
    final_margin = (
        float(np.mean(margin_mean[-100:]))
        if margin_mean
        else float("nan")
    )

    max_margin = (
        float(max(margin_mean))
        if margin_mean
        else float("nan")
    )

    max_zero_target = (
        int(max(n_zero_target))
        if n_zero_target
        else 0
    )

    final_zero_target = (
        int(n_zero_target[-1])
        if n_zero_target
        else 0
    )

    return {
        "loss": loss,
        "margin_mean": margin_mean,
        "margin_min": margin_min,
        "n_zero_target": n_zero_target,
        "test_acc": test_acc,
        "final_test_acc": final_test_acc,
        "final_train_acc": final_train_acc,
        "final_margin": final_margin,
        "max_margin": max_margin,
        "max_zero_target": max_zero_target,
        "final_zero_target": final_zero_target,
        "trainer": trainer,
    }

# Step 21 - find_spikes
def find_spikes(losses, factor=10.0, min_loss=1e-5, start=100, gap=50):
    spikes = []

    # Examine each loss value from the specified starting step.
    for s in range(max(1, start), len(losses)):
        current_loss = losses[s]
        previous_loss = losses[s - 1]

        # A spike requires both a sufficiently large relative jump
        # and an absolute loss above the minimum threshold.
        if current_loss > factor * previous_loss and current_loss > min_loss:
            # Merge nearby jumps into the same event, retaining
            # the step index of the first jump in each event.
            if not spikes or s - spikes[-1] > gap:
                spikes.append(s)

    return spikes

# Step 22 - slingshot_experiment
def slingshot_experiment(
    seed=0,
    steps=3000,
    precision="fp32",
    eps=1e-8,
    p=31,
    d_model=64,
    n_heads=4,
    lr=3e-3,
    weight_decay=1.0,
    train_frac=0.5,
    eval_every=100,
    losses=("standard", "fixed"),
):
    # Create one dataset and reuse the exact same train/test split.
    data = modular_addition_dataset(
        p=p,
        train_frac=train_frac,
        seed=seed,
    )

    # Select the required loss implementations.
    loss_functions = {
        "standard": cross_entropy_standard,
        "fixed": cross_entropy_fixed,
    }

    results = {}

    for name in losses:
        if name not in loss_functions:
            raise ValueError(f"Unsupported loss: {name}")

        # Initialize a fresh model with the same seed for a fair comparison.
        model = GrokTransformer(
            p,
            d_model=d_model,
            n_heads=n_heads,
            seed=seed,
        )

        # Train using the selected loss and experiment configuration.
        history = train_run(
            model=model,
            loss_fn=loss_functions[name],
            data=data,
            steps=steps,
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            precision=precision,
            eval_every=eval_every,
        )

        # Detect Slingshot events from the recorded training-loss curve.
        history["spikes"] = find_spikes(history["loss"])

        # Store the full history under the loss name.
        results[name] = history

    # Include the shared dataset for downstream analysis.
    results["data"] = data

    return results

# Step 23 - equilibrium_experiment
def equilibrium_experiment(
    seed=0,
    eps_values=(1e-8, 1e-12),
    steps=2500,
    precision="fp32",
    p=31,
    d_model=64,
    n_heads=4,
    lr=3e-3,
    weight_decay=1.0,
    train_frac=0.5,
    eval_every=100,
    precomputed=None,
):
    # Create the shared dataset for this seed.
    data = modular_addition_dataset(
        p=p,
        train_frac=train_frac,
        seed=seed,
    )

    results = {}

    for eps in eps_values:
        # Reuse an existing history when the requested epsilon is present.
        if precomputed is not None and eps in precomputed:
            history = precomputed[eps]
        else:
            # Create a fresh model for this epsilon.
            model = GrokTransformer(
                p,
                d_model=d_model,
                n_heads=n_heads,
                seed=seed,
            )

            # Train using the fixed cross-entropy loss.
            history = train_run(
                model=model,
                loss_fn=cross_entropy_fixed,
                data=data,
                steps=steps,
                lr=lr,
                weight_decay=weight_decay,
                eps=eps,
                precision=precision,
                eval_every=eval_every,
            )

        # Summarize the measured margin, gradient collapse, accuracy,
        # loss spikes, and theoretical equilibrium margin.
        results[eps] = {
            "final_margin": float(history["final_margin"]),
            "max_zero_target": int(history["max_zero_target"]),
            "final_test_acc": float(history["final_test_acc"]),
            "spikes": find_spikes(history["loss"]),
            "log_inv_eps": float(math.log(1 / eps)),
        }

    return results

# Step 24 - distillation_experiment
from tinygrad.nn.state import get_state_dict

def kl_from_logits(z_teacher, z_student, temperature=1.0):
    # KL divergence requires a positive temperature.
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    # Convert inputs to NumPy float64 arrays for accurate KL measurement.
    if isinstance(z_teacher, Tensor):
        z_teacher = z_teacher.numpy()
    if isinstance(z_student, Tensor):
        z_student = z_student.numpy()

    z_teacher = np.asarray(z_teacher, dtype=np.float64)
    z_student = np.asarray(z_student, dtype=np.float64)

    # Apply temperature scaling.
    teacher_scaled = z_teacher / temperature
    student_scaled = z_student / temperature

    # Compute stable log-softmax values using log-sum-exp.
    log_p = teacher_scaled - np.logaddexp.reduce(
        teacher_scaled, axis=-1, keepdims=True
    )
    log_q = student_scaled - np.logaddexp.reduce(
        student_scaled, axis=-1, keepdims=True
    )

    # Teacher probability distribution.
    p = np.exp(log_p)

    # KL(p || q), summed across classes and averaged across examples.
    kl_per_example = np.sum(p * (log_p - log_q), axis=-1)

    return float(np.mean(kl_per_example))


def distillation_experiment(
    teacher,
    data,
    seed=0,
    steps=300,
    temperature=4.0,
    lr=2e-3,
    eps=1e-8,
    precision="fp32",
    kl_every=10,
):
    if kl_every <= 0:
        raise ValueError("kl_every must be a positive integer.")

    p = data["p"]

    # Encode the training data and realize the input and dummy targets.
    x_train, y_train = one_hot_batch(
        data["x_train"], data["y_train"], p
    )
    X = Tensor(x_train).realize()
    T_dummy = Tensor(y_train).realize()

    # Save independent NumPy copies of the teacher's parameters.
    teacher_arrays = {
        name: tensor.numpy().copy()
        for name, tensor in get_state_dict(teacher).items()
    }

    results = {}

    # Compare standard and fixed soft-target cross-entropy.
    loss_functions = (
        ("standard", soft_cross_entropy_standard),
        ("fixed", soft_cross_entropy_fixed),
    )

    for name, loss_fn in loss_functions:
        # Create a fresh student with the same architecture as the teacher.
        student = GrokTransformer(
            p,
            teacher.d_model,
            teacher.n_heads,
            seed,
        )

        # Copy the teacher's weights into the corresponding student tensors.
        student_state = get_state_dict(student)

        if set(student_state) != set(teacher_arrays):
            raise ValueError(
                "Student and teacher parameter names do not match."
            )

        for param_name, teacher_array in teacher_arrays.items():
            student_state[param_name].assign(
                Tensor(teacher_array)
            ).realize()

        # Use no weight decay for the distillation experiment.
        trainer = GrokTrainer(
            student,
            lambda z, zt, prec: loss_fn(
                z, zt, temperature, prec
            ),
            lr=lr,
            weight_decay=0.0,
            eps=eps,
            precision=precision,
        )

        # Obtain the teacher logits through the compiled training step.
        # T_dummy is used only to trigger the forward pass.
        z_teacher = trainer.snapshot_logits(X, T_dummy)
        ZT = Tensor(z_teacher).realize()

        # Track KL divergence during student training.
        kl = []

        for s in range(steps):
            trainer.step(X, ZT)

            if (s + 1) % kl_every == 0:
                student_logits = trainer.last_logits.numpy()

                kl_value = kl_from_logits(
                    z_teacher,
                    student_logits,
                    temperature,
                )

                kl.append((s + 1, kl_value))

        # Snapshot the final student logits without applying a parameter update.
        z_final = trainer.snapshot_logits(X, ZT)

        # Maximum absolute parameter difference from the original teacher.
        final_student_state = get_state_dict(student)

        param_drift = max(
            (
                float(
                    np.max(
                        np.abs(
                            final_student_state[param_name].numpy()
                            - teacher_array
                        )
                    )
                )
                for param_name, teacher_array in teacher_arrays.items()
            ),
            default=0.0,
        )

        # Record the final KL divergence and accuracy measurements.
        kl_final = kl_from_logits(
            z_teacher,
            z_final,
            temperature,
        )

        final_train_acc = accuracy(
            student,
            data["x_train"],
            data["y_train"],
            p,
        )

        final_test_acc = accuracy(
            student,
            data["x_test"],
            data["y_test"],
            p,
        )

        results[name] = {
            "kl": kl,
            "kl_max": float(
                max((value for _, value in kl), default=0.0)
            ),
            "kl_final": kl_final,
            "param_drift": param_drift,
            "final_train_acc": final_train_acc,
            "final_test_acc": final_test_acc,
        }

    return results

