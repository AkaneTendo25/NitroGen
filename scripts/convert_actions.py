import argparse
import os
import numpy as np
import polars as pl
from glob import glob


def convert_parquet_to_npz(data_dir):
    """
    Scans data_dir for actions_processed.parquet files and converts them
    to actions.npz in the same directory.
    """

    # Official NitroGen token order
    TOKEN_ORDER = [
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
    BUTTONS_DIM = len(TOKEN_ORDER)

    print(f"Scanning {data_dir} for parquet files...")
    parquet_files = sorted(
        glob(os.path.join(data_dir, "**", "actions_processed.parquet"), recursive=True)
    )

    if not parquet_files:
        print(
            "No actions_processed.parquet files found. Checking actions_raw.parquet..."
        )
        parquet_files = sorted(
            glob(os.path.join(data_dir, "**", "actions_raw.parquet"), recursive=True)
        )

    if not parquet_files:
        print("No parquet files found.")
        return

    print(f"Found {len(parquet_files)} files. Converting...")

    for pq_path in parquet_files:
        chunk_dir = os.path.dirname(pq_path)
        npz_path = os.path.join(chunk_dir, "actions.npz")

        if os.path.exists(npz_path):
            print(f"Skipping {npz_path} (already exists)")
            continue

        print(f"Converting {pq_path} -> {npz_path}")

        try:
            df = pl.read_parquet(pq_path)
            num_frames = len(df)
            cols = df.columns

            # 1. Extract Buttons
            buttons = np.zeros((num_frames, BUTTONS_DIM), dtype=np.float32)
            for i, token in enumerate(TOKEN_ORDER):
                col = token.lower()
                if col in cols:
                    buttons[:, i] = df[col].to_numpy().astype(np.float32)

            # 2. Extract Joysticks
            # Assuming columns 'j_left' and 'j_right' exist and are lists/structs of [x, y]
            j_left = np.zeros((num_frames, 2), dtype=np.float32)
            j_right = np.zeros((num_frames, 2), dtype=np.float32)

            if "j_left" in cols:
                # Handle list column
                jl_list = df["j_left"].to_list()
                for i, val in enumerate(jl_list):
                    if val and len(val) >= 2:
                        j_left[i] = val[:2]

            if "j_right" in cols:
                jr_list = df["j_right"].to_list()
                for i, val in enumerate(jr_list):
                    if val and len(val) >= 2:
                        j_right[i] = val[:2]

            # 3. Save compressed NPZ
            np.savez_compressed(
                npz_path, buttons=buttons, j_left=j_left, j_right=j_right
            )

        except Exception as e:
            print(f"Failed to convert {pq_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert NitroGen parquet actions to NPZ"
    )
    parser.add_argument("data_dir", help="Root directory of the dataset")
    args = parser.parse_args()

    convert_parquet_to_npz(args.data_dir)
