import os

# Must be set before importing numpy/pandas/pyarrow-backed code.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["ARROW_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR = Path(".")
OUT_DIR = Path("processed")
OUT_DIR.mkdir(exist_ok=True)

N_FRAMES = 64

LANDMARK_LAYOUT = {
    "left_hand": {
        "offset": 0,
        "n_landmarks": 21,
    },
    "right_hand": {
        "offset": 21 * 3,
        "n_landmarks": 21,
    },
    "pose": {
        "offset": (21 + 21) * 3,
        "n_landmarks": 33,
    },
}

FEATURE_DIM = (21 + 21 + 33) * 3


def resize_sequence(seq, n_frames=N_FRAMES):
    old_len = seq.shape[0]

    if old_len == n_frames:
        return seq.astype(np.float32)

    if old_len <= 1:
        return np.repeat(seq, n_frames, axis=0).astype(np.float32)

    old_idx = np.linspace(0.0, 1.0, old_len)
    new_idx = np.linspace(0.0, 1.0, n_frames)

    out = np.empty((n_frames, seq.shape[1]), dtype=np.float32)

    for j in range(seq.shape[1]):
        out[:, j] = np.interp(new_idx, old_idx, seq[:, j])

    return out


def torso_normalize(seq):
    """
    Pose-aware normalization.

    Layout:
      left_hand  : 0:63
      right_hand : 63:126
      pose       : 126:225

    MediaPipe pose landmarks:
      11 = left shoulder
      12 = right shoulder

    For each frame:
      - subtract shoulder centre
      - divide by shoulder width
    """

    seq = seq.copy()

    pose_offset = LANDMARK_LAYOUT["pose"]["offset"]

    left_shoulder = seq[:, pose_offset + 11 * 3 : pose_offset + 11 * 3 + 3]
    right_shoulder = seq[:, pose_offset + 12 * 3 : pose_offset + 12 * 3 + 3]

    centre = (left_shoulder + right_shoulder) / 2.0

    shoulder_width = np.linalg.norm(
        left_shoulder[:, :2] - right_shoulder[:, :2],
        axis=1,
    )

    valid = shoulder_width > 1e-6

    if valid.any():
        fallback_scale = np.median(shoulder_width[valid])
    else:
        nonzero = seq[seq != 0]
        fallback_scale = np.std(nonzero) if len(nonzero) else 1.0

    if fallback_scale < 1e-6:
        fallback_scale = 1.0

    shoulder_width[~valid] = fallback_scale

    for start in range(0, FEATURE_DIM, 3):
        seq[:, start : start + 3] -= centre
        seq[:, start : start + 3] /= shoulder_width[:, None]

    return seq.astype(np.float32)


def load_sequence_fast(path):
    df = pd.read_parquet(
        path,
        columns=["frame", "type", "landmark_index", "x", "y", "z"],
    )

    frames = np.sort(df["frame"].unique())
    n_frames = len(frames)

    if n_frames == 0:
        return np.zeros((1, FEATURE_DIM), dtype=np.float32)

    frame_to_i = {frame: i for i, frame in enumerate(frames)}

    arr = np.zeros((n_frames, FEATURE_DIM), dtype=np.float32)

    for lm_type, spec in LANDMARK_LAYOUT.items():
        sub = df[df["type"] == lm_type]

        if sub.empty:
            continue

        frame_idx = sub["frame"].map(frame_to_i).to_numpy(dtype=np.int64)
        landmark_idx = sub["landmark_index"].to_numpy(dtype=np.int64)

        valid = landmark_idx < spec["n_landmarks"]

        frame_idx = frame_idx[valid]
        landmark_idx = landmark_idx[valid]

        base = spec["offset"] + landmark_idx * 3

        x = sub["x"].to_numpy(dtype=np.float32)[valid]
        y = sub["y"].to_numpy(dtype=np.float32)[valid]
        z = sub["z"].to_numpy(dtype=np.float32)[valid]

        arr[frame_idx, base + 0] = x
        arr[frame_idx, base + 1] = y
        arr[frame_idx, base + 2] = z

    return arr


def process_one(job):
    idx, rel_path, label, participant_id, sign, sequence_id = job

    path = DATA_DIR / rel_path

    seq = load_sequence_fast(path)
    original_frames = seq.shape[0]

    seq = resize_sequence(seq)
    seq = torso_normalize(seq)

    return {
        "idx": idx,
        "seq": seq,
        "label": label,
        "original_frames": original_frames,
        "participant_id": participant_id,
        "sign": sign,
        "sequence_id": sequence_id,
        "path": rel_path,
        "error": None,
    }


def process_one_safe(job):
    try:
        return process_one(job)
    except Exception as e:
        idx, rel_path, label, participant_id, sign, sequence_id = job

        return {
            "idx": idx,
            "seq": np.zeros((N_FRAMES, FEATURE_DIM), dtype=np.float32),
            "label": label,
            "original_frames": -1,
            "participant_id": participant_id,
            "sign": sign,
            "sequence_id": sequence_id,
            "path": rel_path,
            "error": repr(e),
        }


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")

    with open(DATA_DIR / "sign_to_prediction_index_map.json") as f:
        sign_map = json.load(f)

    jobs = [
        (
            i,
            row["path"],
            sign_map[row["sign"]],
            row["participant_id"],
            row["sign"],
            row["sequence_id"],
        )
        for i, row in train.iterrows()
    ]

    n_samples = len(train)

    X_path = OUT_DIR / f"X_{N_FRAMES}f_{FEATURE_DIM}d.dat"
    y_path = OUT_DIR / f"y_{N_FRAMES}f.npy"
    meta_path = OUT_DIR / f"metadata_{N_FRAMES}f.csv"
    error_path = OUT_DIR / f"errors_{N_FRAMES}f.csv"

    X = np.memmap(
        X_path,
        dtype=np.float32,
        mode="w+",
        shape=(n_samples, N_FRAMES, FEATURE_DIM),
    )

    y = np.zeros(n_samples, dtype=np.int64)

    metadata_rows = []
    error_rows = []

    n_workers = int(os.environ.get("N_WORKERS", "8"))

    print(f"Samples     : {n_samples}")
    print(f"N_FRAMES    : {N_FRAMES}")
    print(f"FEATURE_DIM : {FEATURE_DIM}")
    print(f"Workers     : {n_workers}")
    print(f"Output X    : {X_path}")

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_one_safe, job) for job in jobs]

        for future in tqdm(as_completed(futures), total=n_samples):
            result = future.result()

            idx = result["idx"]

            X[idx] = result["seq"]
            y[idx] = result["label"]

            metadata_rows.append(
                {
                    "idx": idx,
                    "path": result["path"],
                    "participant_id": result["participant_id"],
                    "sequence_id": result["sequence_id"],
                    "sign": result["sign"],
                    "label": result["label"],
                    "original_frames": result["original_frames"],
                    "error": result["error"],
                }
            )

            if result["error"] is not None:
                error_rows.append(
                    {
                        "idx": idx,
                        "path": result["path"],
                        "error": result["error"],
                    }
                )

    X.flush()

    np.save(y_path, y)

    metadata = pd.DataFrame(metadata_rows).sort_values("idx")
    metadata.to_csv(meta_path, index=False)

    errors = pd.DataFrame(error_rows)
    errors.to_csv(error_path, index=False)

    print("\nSaved:")
    print(X_path)
    print(y_path)
    print(meta_path)
    print(error_path)

    print("\nErrors:", len(error_rows))

    print("\nTo load X later:")
    print(
        f'X = np.memmap("{X_path}", dtype=np.float32, mode="r", '
        f"shape=({n_samples}, {N_FRAMES}, {FEATURE_DIM}))"
    )


if __name__ == "__main__":
    main()