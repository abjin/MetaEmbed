"""Evaluate saved LoRA checkpoints with a training config's eval settings, without training.

The model is built once from the config (base model + fresh LoRA), then each checkpoint's
adapter weights are loaded into it and `ColModelTraining.eval()` runs, writing
`<out_dir>/<checkpoint name>/vidore_v1_dist.json` and `vidore_v2_dist.json`.
Use it to add metrics (e.g. the mixed-budget group) to runs that are already trained.

Usage (one GPU; a one-process torch.distributed group is created on MAIN_PROCESS_PORT,
default 29511, so it can run next to a training job that holds 29500):
    PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=3 DATA_DIR=.../data_dir/ \
      .venv/bin/python scripts/eval/eval_checkpoints.py <config.yaml> <out_dir> <checkpoint_dir> [...]

`eval_mrl_groups` in the config decides the groups, e.g.
    eval_mrl_groups: [[16, 1], [16, 4], [16, 8], [16, 16], [16, 64], [16, [1, 4, 8, 16, 64]]]
"""

import os
import sys
from types import SimpleNamespace

import configue
import torch
import torch.distributed as dist
from accelerate import PartialState
from peft import set_peft_model_state_dict
from safetensors.torch import load_file

from colpali_engine.trainer.colmodel_training import (
    ColModelTraining,
    ColModelTrainingConfig,
)


def main(config_file, out_dir, checkpoints):
    state = PartialState()
    if not dist.is_initialized():
        # a single-process launch does not create a process group, but the eval scorers
        # and the dataset loader call torch.distributed, so make a world of one
        port = os.environ.get("MAIN_PROCESS_PORT", "29511")
        dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)
    config = configue.load(config_file, sub_path="config")
    assert isinstance(config, ColModelTrainingConfig)
    app = ColModelTraining(config, config_file)
    model = app.model.to(state.device)
    model.eval()
    # eval() reads a few trainer fields (main-process check, wandb callback, step) only to
    # write results; a stub stands in for the HF Trainer that init_trainer() would build
    app.trainer = SimpleNamespace(
        accelerator=SimpleNamespace(is_main_process=state.is_main_process),
        callback_handler=SimpleNamespace(callbacks=[]),
        state=SimpleNamespace(global_step=0),
    )

    for ckpt in checkpoints:
        weights = load_file(os.path.join(ckpt, "adapter_model.safetensors"))
        result = set_peft_model_state_dict(model, weights)
        missing = [k for k in result.missing_keys if "lora_" in k]
        assert not result.unexpected_keys and not missing, (result.unexpected_keys, missing)
        app.config.output_dir = os.path.join(out_dir, os.path.basename(os.path.normpath(ckpt)))
        os.makedirs(app.config.output_dir, exist_ok=True)
        print(f"Evaluating {ckpt} -> {app.config.output_dir}", flush=True)
        with torch.no_grad():
            app.eval()


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
