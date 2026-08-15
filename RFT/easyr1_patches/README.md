# EasyR1 RFT Patch

This patch makes EasyR1 pass each sample's original prompt text to RoboGPT-R1's batch reward function:

```text
parquet.reward_model.content
  -> data.non_tensor_batch["reward_model"]
  -> reward_inputs[i]["content"]
  -> LCS_reward.py extracts the action map
```

The patch modifies only EasyR1's `verl/workers/reward/function.py`; it does not replace the complete upstream file.

## Prerequisites

- EasyR1 has been installed.
- RoboGPT-R1's RFT configuration uses `reward_type: batch`.

The patch was verified with EasyR1 commit `b44b311b669bf1fd1aa2fc36f2251482ba33cb16`. Run the preflight check before using a newer upstream version. If the check fails, the target function's context has changed and the patch must be updated for that version; do not force-apply it.

## Apply the patch

With the example directory layout:

```text
/workspace/Codes/
├── RoboGPT-R1/
└── EasyR1/
```

run:

```bash
cd /workspace/Codes/RoboGPT-R1
bash RFT/easyr1_patches/apply_patch.sh
```

If EasyR1 is located elsewhere, pass its absolute path:

```bash
bash RFT/easyr1_patches/apply_patch.sh /path/to/EasyR1
```

The script runs `git apply --check` before writing to EasyR1. It only applies the patch when it can be applied cleanly, and exits successfully without changes when the patch is already applied.

## Manual application and verification

To apply the patch manually:

```bash
cd /workspace/Codes/EasyR1
git apply --check /workspace/Codes/RoboGPT-R1/RFT/easyr1_patches/0001-pass-reward-model-content.patch
git apply /workspace/Codes/RoboGPT-R1/RFT/easyr1_patches/0001-pass-reward-model-content.patch
```

Afterward, `git diff -- verl/workers/reward/function.py` should show only the added `content` forwarding logic when building batch `reward_inputs`.

`ACTION_MAP_PATH` remains a fallback mechanism used only when the reward function cannot extract an action map from `content`.
