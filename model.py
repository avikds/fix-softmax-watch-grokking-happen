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

