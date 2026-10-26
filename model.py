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

