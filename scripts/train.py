import argparse
import os
import time
import glob
import json
from contextlib import nullcontext

import numpy as np
import torch
import yaml

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import polars as pl
    import av
except ImportError:
    pl = None
    av = None

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from nitrogen.cfg import CkptConfig, ModalityConfig
from nitrogen.flow_matching_transformer.nitrogen import NitroGen, NitroGen_Config
from nitrogen.mm_tokenizers import NitrogenTokenizer, NitrogenTokenizerConfig
from nitrogen.lora import LoRALinear, mark_only_lora_as_trainable
from transformers import AutoImageProcessor


class NitroGenDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        num_frames: int,
        image_size: int,
        action_horizon: int,
        buttons_dim: int,
    ):
        if av is None:
            raise ImportError(
                "av is required for RealNitroGenDataset. Install with `pip install av`"
            )

        self.data_dir = data_dir
        self.num_frames = num_frames
        self.image_size = image_size
        self.action_horizon = action_horizon
        self.buttons_dim = buttons_dim

        # Scan for metadata.json to find chunks
        self.samples = sorted(
            glob.glob(os.path.join(data_dir, "**", "metadata.json"), recursive=True)
        )
        if not self.samples:
            print(
                f"Warning: No metadata.json files found in {data_dir}. Dataset is empty."
            )
        else:
            print(f"Found {len(self.samples)} chunks in {data_dir}")

        # Mapping from parquet column names to Nitrogen button indices
        # Based on BUTTON_ACTION_TOKENS in shared.py
        # ['BACK', 'DPAD_DOWN', 'DPAD_LEFT', 'DPAD_RIGHT', 'DPAD_UP', 'EAST', 'GUIDE',
        #  'LEFT_SHOULDER', 'LEFT_THUMB', 'LEFT_TRIGGER', 'NORTH', 'RIGHT_BOTTOM', 'RIGHT_LEFT',
        #  'RIGHT_RIGHT', 'RIGHT_SHOULDER', 'RIGHT_THUMB', 'RIGHT_TRIGGER', 'RIGHT_UP', 'SOUTH',
        #  'START', 'WEST']

        self.token_order = [
            "back",
            "dpad_down",
            "dpad_left",
            "dpad_right",
            "dpad_up",
            "east",
            "guide",
            "left_shoulder",
            "left_thumb",
            "left_trigger",
            "north",
            "right_bottom",
            "right_left",
            "right_right",
            "right_shoulder",
            "right_thumb",
            "right_trigger",
            "right_up",
            "south",
            "start",
            "west",
        ]
        self.col_map = {token: i for i, token in enumerate(self.token_order)}

    def __len__(self) -> int:
        return len(self.samples)

    def _load_video_frames(self, video_path, start_idx, num_frames):
        container = av.open(video_path)
        stream = container.streams.video[0]

        avg_fps = stream.average_rate
        time_base = stream.time_base
        target_pts = int(start_idx / avg_fps / time_base)
        container.seek(target_pts, stream=stream)

        frames = []
        for frame in container.decode(stream):
            if len(frames) >= num_frames:
                break

            img = frame.to_ndarray(format="rgb24")
            import cv2
            img = cv2.resize(img, (self.image_size, self.image_size))
            frames.append(img)

        container.close()

        # Pad if not enough frames
        while len(frames) < num_frames:
            frames.append(
                np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            )

        return np.stack(frames)  # [T, H, W, C]

    def __getitem__(self, idx: int) -> dict:
        meta_path = self.samples[idx]
        chunk_dir = os.path.dirname(meta_path)

        # 1. Load Metadata
        with open(meta_path, "r") as f:
            _ = json.load(f)

        # 2. Load Actions (Prioritize NPZ, then Parquet)
        npz_path = os.path.join(chunk_dir, "actions.npz")
        pq_path = os.path.join(chunk_dir, "actions_processed.parquet")
        if not os.path.exists(pq_path):
            pq_path = os.path.join(chunk_dir, "actions_raw.parquet")

        buttons_tensor = None
        j_left = None
        j_right = None
        total_frames = 0

        if os.path.exists(npz_path):
            # Load from .npz
            # Expects keys: 'buttons', 'j_left', 'j_right'
            # buttons: [T, 21] (or matches buttons_dim)
            # j_left: [T, 2]
            # j_right: [T, 2]
            with np.load(npz_path) as data:
                # We assume the user has pre-processed data into the correct format for NPZ
                loaded_buttons = data["buttons"]
                loaded_j_left = data["j_left"]
                loaded_j_right = data["j_right"]
                total_frames = len(loaded_buttons)

                # Determine sampling window
                max_len = total_frames
                needed = max(self.num_frames, self.action_horizon)
                if max_len < needed:
                    start_idx = 0
                else:
                    start_idx = np.random.randint(0, max_len - needed + 1)

                # Slice
                # Handle if slice goes out of bounds (padding) if short
                end_idx = start_idx + self.action_horizon

                # Helper to safe slice
                def safe_slice(arr, start, end):
                    res = arr[start:end]
                    if len(res) < (end - start):
                        # Pad with last frame or zeros? Zeros for now.
                        pad_len = (end - start) - len(res)
                        res = np.concatenate(
                            [
                                res,
                                np.zeros((pad_len, *arr.shape[1:]), dtype=arr.dtype),
                            ],
                            axis=0,
                        )
                    return res

                buttons_tensor = safe_slice(loaded_buttons, start_idx, end_idx).astype(
                    np.float32
                )
                j_left = safe_slice(loaded_j_left, start_idx, end_idx).astype(
                    np.float32
                )
                j_right = safe_slice(loaded_j_right, start_idx, end_idx).astype(
                    np.float32
                )

        elif os.path.exists(pq_path):
            if pl is None:
                raise ImportError(
                    "polars is required to load parquet files. Install `pip install polars` or provide actions.npz"
                )

            df = pl.read_parquet(pq_path)
            total_frames = len(df)

            # Determine sampling window
            max_len = total_frames
            needed = max(self.num_frames, self.action_horizon)
            if max_len < needed:
                start_idx = 0
            else:
                start_idx = np.random.randint(0, max_len - needed + 1)

            # Extract Actions from Parquet
            buttons_tensor = np.zeros(
                (self.action_horizon, self.buttons_dim), dtype=np.float32
            )
            cols = df.columns
            for i, token_name in enumerate(self.token_order):
                col_name = token_name.lower()
                if col_name in cols:
                    val = (
                        df[col_name]
                        .slice(start_idx, self.action_horizon)
                        .to_numpy()
                        .astype(np.float32)
                    )
                    length = len(val)
                    buttons_tensor[:length, i] = val

            j_left = np.zeros((self.action_horizon, 2), dtype=np.float32)
            j_right = np.zeros((self.action_horizon, 2), dtype=np.float32)

            if "j_left" in cols:
                jl_data = df["j_left"].slice(start_idx, self.action_horizon).to_list()
                for i, val in enumerate(jl_data):
                    if val and len(val) >= 2:
                        j_left[i] = val[:2]

            if "j_right" in cols:
                jr_data = df["j_right"].slice(start_idx, self.action_horizon).to_list()
                for i, val in enumerate(jr_data):
                    if val and len(val) >= 2:
                        j_right[i] = val[:2]
        else:
            raise FileNotFoundError(
                f"No actions.npz or actions parquet found in {chunk_dir}"
            )

        # 5. Load Video
        # Look for video file in chunk dir
        vid_extensions = ["*.mp4", "*.mkv", "*.avi", "*.webm"]
        vid_path = None
        for ext in vid_extensions:
            candidates = glob.glob(os.path.join(chunk_dir, ext))
            if candidates:
                vid_path = candidates[0]
                break

        if vid_path:
            frames_hwc = self._load_video_frames(vid_path, start_idx, self.num_frames)
            # _preprocess_frames expects [T, C, H, W] in [0, 1] (but processed later)
            # Our loader returns uint8 [T, H, W, 3] (RGB)
            # To match Synthetic, we should return floats [T, 3, H, W] in [0, 1]

            frames_float = frames_hwc.astype(np.float32) / 255.0
            frames_chw = np.transpose(frames_float, (0, 3, 1, 2))  # HWC -> CHW
        else:
            # Fallback if no video found: black frames
            frames_chw = np.zeros(
                (self.num_frames, 3, self.image_size, self.image_size), dtype=np.float32
            )

        return {
            "frames": frames_chw,
            "dropped_frames": np.zeros((self.num_frames,), dtype=bool),
            "buttons": buttons_tensor[None, ...],
            "j_left": j_left[None, ...],
            "j_right": j_right[None, ...],
            "game": None,  # Could map game string from metadata to ID if game_mapping exists
        }


def _stack_tokenized(samples: list[dict]) -> dict:
    stacked = {}
    keys = samples[0].keys()
    for key in keys:
        values = [s[key] for s in samples]
        first = values[0]
        if isinstance(first, np.ndarray):
            arr = np.stack(values)
            if key in {"vl_token_ids", "sa_token_ids"}:
                stacked[key] = torch.tensor(arr, dtype=torch.long)
            elif key in {"actions", "images"}:
                stacked[key] = torch.tensor(arr, dtype=torch.float32)
            elif key in {"actions_mask"}:
                stacked[key] = torch.tensor(arr, dtype=torch.float32)
            else:
                stacked[key] = torch.tensor(arr)
        elif torch.is_tensor(first):
            stacked[key] = torch.stack(values)
        else:
            stacked[key] = values
    return stacked


def _preprocess_frames(img_proc, frames: np.ndarray) -> torch.Tensor:
    # frames: [T, C, H, W] in [0, 1]; convert to HWC uint8 like inference inputs.
    frames_hwc = np.transpose(frames, (0, 2, 3, 1))
    frames_u8 = (frames_hwc * 255.0).clip(0, 255).astype(np.uint8)
    pixel_values = img_proc(list(frames_u8), return_tensors="pt")["pixel_values"]
    return pixel_values


def _resolve_module(root: torch.nn.Module, path: str) -> torch.nn.Module:
    current = root
    for part in path.split("."):
        if part.isdigit() and isinstance(
            current, (torch.nn.ModuleList, torch.nn.Sequential, list, tuple)
        ):
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def _replace_linear_with_lora(
    module: torch.nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
) -> int:
    replaced = 0
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            setattr(module, name, LoRALinear(child, rank, alpha, dropout))
            replaced += 1
        else:
            replaced += _replace_linear_with_lora(child, rank, alpha, dropout)
    return replaced


def _to_device(batch: dict, device: torch.device) -> dict:
    for key, value in batch.items():
        if torch.is_tensor(value):
            batch[key] = value.to(device, non_blocking=True)
    return batch


def _validate_shapes(model_cfg: NitroGen_Config) -> None:
    inner_dim = (
        model_cfg.diffusion_model_cfg.num_attention_heads
        * model_cfg.diffusion_model_cfg.attention_head_dim
    )
    if inner_dim != model_cfg.hidden_size:
        raise ValueError(
            f"DiT inner dim {inner_dim} must match hidden_size {model_cfg.hidden_size}."
        )
    sa_inner = (
        model_cfg.vl_self_attention_cfg.num_attention_heads
        * model_cfg.vl_self_attention_cfg.attention_head_dim
    )
    if sa_inner != model_cfg.vision_hidden_size:
        raise ValueError(
            f"VL inner dim {sa_inner} must match vision_hidden_size {model_cfg.vision_hidden_size}."
        )
    if model_cfg.diffusion_model_cfg.output_dim != model_cfg.hidden_size:
        raise ValueError(
            f"DiT output_dim {model_cfg.diffusion_model_cfg.output_dim} must match hidden_size {model_cfg.hidden_size}."
        )


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="NitroGen synthetic LoRA trainer")
    parser.add_argument("--config", type=str, default="configs/train_synth.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["train"]
    lora_cfg = cfg["lora"]

    # Extract parameters from config instead of CLI args
    checkpoint_path = train_cfg.get("checkpoint")
    data_dir = train_cfg.get("data_dir")
    no_save = train_cfg.get("no_save", False)
    save_lora_only_path = train_cfg.get("save_lora_only")
    lora_targets = lora_cfg.get("target_modules", ["model", "vl_self_attention_model"])

    model_cfg = NitroGen_Config.model_validate(cfg["model"])
    tokenizer_cfg = NitrogenTokenizerConfig.model_validate(cfg["tokenizer"])
    modality_cfg = ModalityConfig.model_validate(cfg["modality"])

    # If checkpoint is provided, we optionally verify or override config to ensure architecture match
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Try to load config from checkpoint if available to guarantee architecture match
        if isinstance(ckpt, dict) and "ckpt_config" in ckpt:
            print("Found config in checkpoint. Verifying architecture compatibility...")
            loaded_ckpt_cfg = CkptConfig.model_validate(ckpt["ckpt_config"])
            # We generally trust the checkpoint config for the model architecture
            # but we might want to keep the training params from the YAML.
            # For safety, let's assume we use the checkpoint's model config.
            model_cfg = loaded_ckpt_cfg.model_cfg
            tokenizer_cfg = loaded_ckpt_cfg.tokenizer_cfg
            # Modality config might change for fine-tuning (e.g. different context length?)
            # but usually it's safer to stick to what the model was trained with.
            modality_cfg = loaded_ckpt_cfg.modality_cfg
            print("Model configuration updated from checkpoint.")

        state_dict = (
            ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        )
    else:
        raise ValueError(
            "No 'checkpoint' provided in config! Training on a randomly initialized model is not supported."
        )

    _validate_shapes(model_cfg)

    torch.manual_seed(train_cfg["seed"])
    np.random.seed(train_cfg["seed"])

    device_str = train_cfg["device"]
    if device_str == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requests 'cuda' but CUDA is not available!")

    device = torch.device(device_str)
    print(f"Training on device: {device}")

    if device.type == "cpu":
        print("WARNING: Training is running on CPU! This will be extremely slow.")

    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        # Verify that we can actually move a tensor to cuda
        try:
            torch.zeros(1).to(device)
        except Exception as e:
            raise RuntimeError(f"Failed to move tensor to CUDA device {device}: {e}")

    tokenizer = NitrogenTokenizer(tokenizer_cfg)
    img_proc = AutoImageProcessor.from_pretrained(model_cfg.vision_encoder_name)
    model = NitroGen(config=model_cfg, game_mapping=tokenizer.game_mapping)

    if state_dict is not None:
        try:
            model.load_state_dict(state_dict, strict=True)
            print("Successfully loaded model weights from checkpoint.")
        except RuntimeError as exc:
            raise ValueError(
                "Failed to load checkpoint with strict=True. The model architecture in the config "
                "does not match the weights in the checkpoint."
            ) from exc

    if not data_dir:
        raise ValueError("Config field 'data_dir' must be provided for training.")

    # lora_targets is already a list from the config, ensure it's normalized
    if not isinstance(lora_targets, list):
        raise ValueError("lora_targets must be a list in config")

    replaced = 0
    replaced_by_target = {}
    for target in lora_targets:
        try:
            target_module = _resolve_module(model, target)
        except AttributeError as exc:
            raise ValueError(f"LoRA target '{target}' not found on model") from exc
        count = _replace_linear_with_lora(
            target_module,
            rank=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=lora_cfg["dropout"],
        )
        replaced_by_target[target] = count
        replaced += count
    if replaced == 0:
        raise ValueError("LoRA injection did not replace any Linear layers.")
    print(f"LoRA injected into {replaced} Linear layers.")
    for target, count in replaced_by_target.items():
        print(f"  - {target}: {count}")
    mark_only_lora_as_trainable(model)
    # Keep frozen modules in eval mode during forward() for LoRA training.
    cfg.setdefault("train", {})["runtime_overrides"] = {
        "lora_targets": lora_targets,
    }

    # Training configuration verification
    print("\nTraining Configuration:")

    def count_params(module):
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable

    ve_total, ve_train = count_params(model.vision_encoder)
    print(f"  Vision Encoder: {ve_total:,} total, {ve_train:,} trainable", end="")
    print(" (FROZEN)" if ve_train == 0 else " (TRAINING!)")

    dit_total, dit_train = count_params(model.model)
    vl_total, vl_train = count_params(model.vl_self_attention_model)
    base_total = dit_total + vl_total
    print(f"  Base Model: {base_total:,} total params")

    lora_params = sum(
        p.numel()
        for n, p in model.named_parameters()
        if "lora_" in n and p.requires_grad
    )
    print(f"  LoRA Adapters: {lora_params:,} trainable params")

    if ve_train == 0 and lora_params > 0:
        print("  Status: Training LoRA adapters only (base model frozen)")
    else:
        print("  WARNING: Unexpected parameter configuration")
    print()

    model.to(device)
    model.train()

    # Set frozen modules to eval mode to prevent BatchNorm/Dropout issues
    def set_frozen_to_eval(module: torch.nn.Module) -> None:
        for name, child in module.named_children():
            has_trainable = any(p.requires_grad for p in child.parameters())
            if not has_trainable:
                child.eval()
            else:
                set_frozen_to_eval(child)

    set_frozen_to_eval(model)

    dataset_cfg = train_cfg["synthetic"]
    print(f"Loading dataset from {data_dir}")
    dataset = NitroGenDataset(
        data_dir=data_dir,
        num_frames=dataset_cfg["num_frames"],
        image_size=dataset_cfg["image_size"],
        action_horizon=tokenizer_cfg.action_horizon,
        buttons_dim=dataset_cfg["buttons_dim"],
    )

    def collate_fn(batch: list[dict]) -> dict:
        processed = []
        for sample in batch:
            sample = dict(sample)
            sample["frames"] = _preprocess_frames(img_proc, sample["frames"])
            processed.append(sample)
        tokenized = [tokenizer.encode(sample) for sample in processed]
        return _stack_tokenized(tokenized)

    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=use_cuda,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    precision = train_cfg["precision"].lower()
    if use_cuda and precision == "bf16":
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        scaler = None
    elif use_cuda and precision == "fp16":
        autocast = torch.autocast(device_type="cuda", dtype=torch.float16)
        scaler = torch.cuda.amp.GradScaler()
    else:
        autocast = nullcontext()
        scaler = None

    steps = train_cfg["steps"]
    grad_accum = train_cfg["grad_accum_steps"]
    save_every = train_cfg["save_every"]
    out_dir = train_cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    ckpt_config = CkptConfig(
        experiment_name=train_cfg["experiment_name"],
        model_cfg=model_cfg,
        tokenizer_cfg=tokenizer_cfg,
        modality_cfg=modality_cfg,
    )

    data_iter = iter(loader)
    optimizer.zero_grad(set_to_none=True)

    epoch = 1
    step_start = time.time()

    pbar = tqdm(range(1, steps + 1), desc="Training", unit="step")
    for step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
            epoch += 1

        batch = _to_device(batch, device)

        with autocast:
            output = model(batch)
            loss = output["loss"] / grad_accum

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step % grad_accum == 0:
            if train_cfg["grad_clip_norm"] is not None:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    trainable_params, train_cfg["grad_clip_norm"]
                )

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        current_loss = loss.item() * grad_accum
        step_time = time.time() - step_start
        step_start = time.time()

        pbar.set_postfix({
            "loss": f"{current_loss:.4f}",
            "epoch": epoch,
            "step_time": f"{step_time:.2f}s"
        })

        if (not no_save) and save_every and step % save_every == 0:
            ckpt_path = os.path.join(out_dir, f"ckpt_step_{step}.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "ckpt_config": ckpt_config.model_dump(),
                    "train_config": cfg,
                },
                ckpt_path,
            )
            pbar.write(f"Saved checkpoint to {ckpt_path}")

            if save_lora_only_path:
                lora_state = {
                    k: v
                    for k, v in model.state_dict().items()
                    if ".lora_A" in k or ".lora_B" in k
                }
                torch.save(
                    {
                        "lora": lora_state,
                        "lora_config": lora_cfg,
                        "lora_targets": lora_targets,
                        "base_checkpoint": checkpoint_path,
                        "ckpt_config": ckpt_config.model_dump(),
                    },
                    save_lora_only_path,
                )
                pbar.write(f"Saved LoRA weights to {save_lora_only_path}")

    final_path = os.path.join(out_dir, "ckpt_final.pt")
    if not no_save:
        torch.save(
            {
                "model": model.state_dict(),
                "ckpt_config": ckpt_config.model_dump(),
                "train_config": cfg,
            },
            final_path,
        )
        print(f"\nSaved final checkpoint to {final_path}")

    if save_lora_only_path:
        lora_state = {
            k: v
            for k, v in model.state_dict().items()
            if ".lora_A" in k or ".lora_B" in k
        }
        torch.save(
            {
                "lora": lora_state,
                "lora_config": lora_cfg,
                "lora_targets": lora_targets,
                "base_checkpoint": checkpoint_path,
                "ckpt_config": ckpt_config.model_dump(),
            },
            save_lora_only_path,
        )
        print(f"Saved final LoRA weights to {save_lora_only_path}")


if __name__ == "__main__":
    main()
