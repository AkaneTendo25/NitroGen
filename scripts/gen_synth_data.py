import os
import json
import argparse
import numpy as np
import cv2


def generate_synthetic_dataset(
    output_dir, num_chunks=5, chunk_length=100, image_size=224
):
    """
    Generates a synthetic dataset with the structure required for 'real' training.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating {num_chunks} chunks of synthetic data in '{output_dir}'...")

    # Official NitroGen token order (21 buttons)
    # We match the order expected by the training script
    token_order = [
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
    buttons_dim = len(token_order)

    for i in range(num_chunks):
        chunk_name = f"synth_video_{i:03d}_chunk_0000"
        chunk_dir = os.path.join(output_dir, chunk_name)
        os.makedirs(chunk_dir, exist_ok=True)

        # 1. Generate Metadata
        metadata = {
            "uuid": f"{chunk_name}_actions",
            "chunk_id": "0000",
            "chunk_size": chunk_length,
            "original_video": {
                "resolution": [image_size, image_size],
                "video_id": f"synth_video_{i:03d}",
                "source": "synthetic",
                "url": "http://localhost/synthetic.mp4",
                "start_time": 0.0,
                "end_time": float(chunk_length) / 30.0,
                "duration": float(chunk_length) / 30.0,
                "start_frame": 0,
                "end_frame": chunk_length,
            },
            "game": "synthetic_game",
            "controller_type": "xbox",
        }

        with open(os.path.join(chunk_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        # 2. Generate Actions (NPZ)
        # Random buttons (sparse to be realistic-ish)
        buttons = (np.random.random((chunk_length, buttons_dim)) > 0.95).astype(
            np.float32
        )

        # Random joysticks [-1, 1]
        j_left = np.random.uniform(-1, 1, (chunk_length, 2)).astype(np.float32)
        j_right = np.random.uniform(-1, 1, (chunk_length, 2)).astype(np.float32)

        np.savez_compressed(
            os.path.join(chunk_dir, "actions.npz"),
            buttons=buttons,
            j_left=j_left,
            j_right=j_right,
        )

        # 3. Generate Video (MP4)
        # Create a random noise video
        video_path = os.path.join(chunk_dir, "video.mp4")
        fps = 30
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, fps, (image_size, image_size))

        for _ in range(chunk_length):
            # Generate random colored frame
            frame = np.random.randint(
                0, 256, (image_size, image_size, 3), dtype=np.uint8
            )
            out.write(frame)

        out.release()

        print(f"  - Created {chunk_name} (frames={chunk_length})")

    print("\nDone! To train on this data:")
    print(f"1. Edit configs/train_synth.yaml to set 'data_dir' to '{output_dir}'")
    print(f"2. Run: python scripts/train.py --config configs/train_synth.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic 'real' dataset for testing"
    )
    parser.add_argument(
        "--out-dir", type=str, default="data/synth_test", help="Output directory"
    )
    parser.add_argument(
        "--num-chunks", type=int, default=5, help="Number of chunks to generate"
    )
    parser.add_argument(
        "--len", type=int, default=100, help="Length of each chunk in frames"
    )
    args = parser.parse_args()

    generate_synthetic_dataset(args.out_dir, args.num_chunks, args.len)
