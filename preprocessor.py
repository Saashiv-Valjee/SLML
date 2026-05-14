import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

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

    old_idx = np.linspace(0, 1, old_len)
    new_idx = np.linspace(0, 1, n_frames)

    out = np.empty((n_frames, seq.shape[1]), dtype=np.float32)

    for j in range(seq.shape[1]):
        out[:, j] = np.interp(new_idx, old_idx, seq[:, j])

    return out


def torso_normalize(seq):
    """
    Pose-aware normalization.

    Feature layout:
      left_hand  : 0:63
      right_hand : 63:126
      pose       : 126:225

    MediaPipe pose landmarks:
      11 = left shoulder
      12 = right shoulder

    Normalization:
      - subtract shoulder centre per frame
      - divide by shoulder width per frame
      - fallback to global non-zero std if shoulders are degenerate
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

    valid_scale = shoulder_width > 1e-6

    if valid_scale.any():
        fallback_scale = np.median(shoulder_width[valid_scale])
    else:
        nonzero = seq[seq != 0]
        fallback_scale = np.std(nonzero) if len(nonzero) else 1.0

    if fallback_scale < 1e-6:
        fallback_scale = 1.0

    shoulder_width[~valid_scale] = fallback_scale

    # Apply to every xyz triplet.
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

    return idx, seq, label, original_frames, participant_id, sign, sequence_id, rel_path


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

    X = np.memmap(
        X_path,
        dtype=np.float32,
        mode="w+",
        shape=(n_samples, N_FRAMES, FEATURE_DIM),
    )

    y = np.zeros(n_samples, dtype=np.int64)

    metadata_rows = []

    n_workers = max(1, cpu_count() - 1)

    print(f"Samples     : {n_samples}")
    print(f"N_FRAMES    : {N_FRAMES}")
    print(f"FEATURE_DIM : {FEATURE_DIM}")
    print(f"Workers     : {n_workers}")
    print(f"Output X    : {X_path}")

    with Pool(n_workers) as pool:
        iterator = pool.imap_unordered(process_one, jobs, chunksize=64)

        for result in tqdm(iterator, total=n_samples):
            (
                idx,
                seq,
                label,
                original_frames,
                participant_id,
                sign,
                sequence_id,
                rel_path,
            ) = result

            X[idx] = seq
            y[idx] = label

            metadata_rows.append(
                {
                    "idx": idx,
                    "path": rel_path,
                    "participant_id": participant_id,
                    "sequence_id": sequence_id,
                    "sign": sign,
                    "label": label,
                    "original_frames": original_frames,
                }
            )

    X.flush()

    np.save(y_path, y)

    metadata = pd.DataFrame(metadata_rows).sort_values("idx")
    metadata.to_csv(meta_path, index=False)

    print("\nSaved:")
    print(X_path)
    print(y_path)
    print(meta_path)

    print("\nTo load X later:")
    print(
        f'X = np.memmap("{X_path}", dtype=np.float32, mode="r", '
        f"shape=({n_samples}, {N_FRAMES}, {FEATURE_DIM}))"
    )


if __name__ == "__main__":
    main()