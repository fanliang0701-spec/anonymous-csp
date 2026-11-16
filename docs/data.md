# Prepared task bundles

Use the dataset acquisition, target bytes and profiling-only windows described in the paper. The code consumes prepared arrays so the artifact does not depend on a particular machine layout or distribute trace datasets.

## Input to `pack`

Create one NPZ file with these arrays:

| Field | Shape | Meaning |
| --- | --- | --- |
| `train_traces`, `val_traces`, `attack_traces` | `[N, time]` | Traces after selecting the task window, before per-trace standardization |
| `train_labels`, `val_labels`, `attack_labels` | `[N]` | Unmasked 256-class intermediate-value labels |
| `train_ids`, `val_ids`, `attack_ids` | `[N]` | Unique integer sample IDs from the same acquisition; split IDs must be disjoint |
| `attack_key_byte` | scalar or `[N_attack]` | Actual target key byte; use the final-round byte for `last_transition` |
| `attack_plaintext` | `[N_attack]` | Selected plaintext byte for `first_sbox` |
| `attack_ciphertext` | `[N_attack, 16]` | Actual ciphertext bytes for `last_transition` |
| `attack_group_id` | `[N_attack]`, optional | Independent fixed-key group identifiers when the attack file contains multiple keys |

`pack` validates sample-ID separation and the attack-label/key relationship, then writes `train.npz`, `validation.npz`, and `attack.npz`. The output contains relative task directories, not the original dataset paths. Inputs with mask or offset fields are rejected by this paper-protocol interface.

```python
import numpy as np

# Arrays below come from the selected, disjoint dataset partitions.
np.savez("prepared/F0.npz",
         train_traces=x_train, train_labels=y_train, train_ids=train_ids,
         val_traces=x_val, val_labels=y_val, val_ids=val_ids,
         attack_traces=x_attack, attack_labels=y_attack, attack_ids=attack_ids,
         attack_plaintext=plaintext_byte, attack_key_byte=key_byte)
```

```bash
csp-sca pack --input prepared/F0.npz --output data/F0 --label-rule first_sbox
csp-sca pack --input prepared/DPA-v2.npz --output data/DPA-v2 --label-rule last_transition
```

## Label rules

- `first_sbox`: `SBOX[plaintext_byte XOR key_byte]` for STM32, XMEGA, ASCAD-0, ASCAD-2 and DPA-v4.
- `last_transition`: `INV_SBOX[ciphertext[11] XOR last_round_key_byte] XOR ciphertext[7]` for DPA-v2 and AES_HD; byte indices are zero-based.

The last rule is the byte-valued state transition used by the experimental code, not just an inverse-S-box value. No mask is added to either hypothesis. Supply actual recorded ciphertext and the corresponding key for a physical key-recovery evaluation; the loader verifies consistency with the supplied labels.

The supplied main configuration expects:

| Tasks | Training | Validation | Attack |
| --- | ---: | ---: | ---: |
| F0–F4, XMEGA, ASCAD-0, ASCAD-2 | 45,000 | 5,000 | 5,000 |
| DPA-v4, DPA-v2, AES_HD | 12,000 | 3,000 | 5,000 |

MCU bundles can retain their native 500-sample windows. Main experiments interpolate these to 700 samples; the six-task/token/order studies use 500 samples. Public-dataset windows must already match the configured input length. Every trace is standardized independently by its own mean and standard deviation, with a denominator floor of `1e-6`.

CSP and the covariance/activation-memory baselines require all 256 classes in each task's training split for the balanced memory anchors. Window selection and memory extraction use profiling data. Attack files are not read by `train`.

## Files produced by a run

```text
run.json                       Scientific configuration, method, and seed
checkpoints/stage00.pt          Selected source parameters and initial memory
checkpoints/stageNN.pt          Selected increment, input memory/design, next memory
stages/NN/history.json          Current-task training and validation metrics
stages/NN/training.json         Selected epoch and trainable-parameter count
training_complete.json         Completed stages
evaluation.json                Per-stage/task ACC, NLL, GE, success and recovery
predictions/                   Saved logits and labels from evaluation
```

Checkpoints contain tensors and ordinary containers and are read with `weights_only=True`. Model reconstruction also requires the same pretrained model directory used to fit the source. Generated outputs are local experiment artifacts and are ignored by Git.
