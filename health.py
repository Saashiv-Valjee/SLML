import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

DATA_DIR = Path(".")
OUT_DIR = Path("health")
OUT_DIR.mkdir(exist_ok=True)


def inspect_one(args):
    idx, rel_path = args
    path = DATA_DIR / rel_path

    df = pd.read_parquet(
        path,
        columns=["frame", "type", "landmark_index"],
    )

    frames = df["frame"].unique()
    n_frames = len(frames)

    types = set(df["type"].unique())

    has_left = "left_hand" in types
    has_right = "right_hand" in types
    has_pose = "pose" in types
    has_face = "face" in types

    # Per-frame detection rates
    if n_frames > 0:
        frame_type_counts = (
            df[["frame", "type"]]
            .drop_duplicates()
            .groupby("type")["frame"]
            .count()
            .to_dict()
        )

        left_frame_frac = frame_type_counts.get("left_hand", 0) / n_frames
        right_frame_frac = frame_type_counts.get("right_hand", 0) / n_frames
        pose_frame_frac = frame_type_counts.get("pose", 0) / n_frames
        face_frame_frac = frame_type_counts.get("face", 0) / n_frames
    else:
        left_frame_frac = 0.0
        right_frame_frac = 0.0
        pose_frame_frac = 0.0
        face_frame_frac = 0.0

    return {
        "idx": idx,
        "n_frames": n_frames,
        "has_left_hand": has_left,
        "has_right_hand": has_right,
        "has_pose": has_pose,
        "has_face": has_face,
        "left_frame_frac": left_frame_frac,
        "right_frame_frac": right_frame_frac,
        "pose_frame_frac": pose_frame_frac,
        "face_frame_frac": face_frame_frac,
    }


def describe_numeric(name, arr):
    print(f"\n{name}")
    print("-" * len(name))
    print("min   :", np.min(arr))
    print("mean  :", np.mean(arr))
    print("median:", np.median(arr))
    print("max   :", np.max(arr))

    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"{p:2d}% :", np.percentile(arr, p))


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")

    print("\nBasic dataset")
    print("-------------")
    print("samples:", len(train))
    print("signs:", train["sign"].nunique())
    print("participants:", train["participant_id"].nunique())

    jobs = list(enumerate(train["path"].tolist()))

    n_workers = max(1, cpu_count() - 1)

    print(f"\nUsing {n_workers} workers")

    results = []

    with Pool(n_workers) as pool:
        for out in tqdm(
            pool.imap_unordered(inspect_one, jobs, chunksize=128),
            total=len(jobs),
        ):
            results.append(out)

    health = pd.DataFrame(results).sort_values("idx").reset_index(drop=True)

    full = pd.concat([train.reset_index(drop=True), health.drop(columns=["idx"])], axis=1)

    full.to_csv(OUT_DIR / "dataset_health.csv", index=False)

    print("\nSaved:")
    print(OUT_DIR / "dataset_health.csv")

    # Global frame stats
    frame_counts = full["n_frames"].to_numpy()
    describe_numeric("Frame count stats", frame_counts)

    # Missing modality stats
    print("\nWhole-sequence modality presence")
    print("--------------------------------")
    for col in ["has_left_hand", "has_right_hand", "has_pose", "has_face"]:
        frac = full[col].mean()
        print(f"{col:15s}: {frac * 100:6.2f}%")

    print("\nHand presence combinations")
    print("--------------------------")
    both = (full["has_left_hand"] & full["has_right_hand"]).mean()
    left_only = (full["has_left_hand"] & ~full["has_right_hand"]).mean()
    right_only = (~full["has_left_hand"] & full["has_right_hand"]).mean()
    neither = (~full["has_left_hand"] & ~full["has_right_hand"]).mean()

    print(f"both hands : {both * 100:6.2f}%")
    print(f"left only  : {left_only * 100:6.2f}%")
    print(f"right only : {right_only * 100:6.2f}%")
    print(f"neither    : {neither * 100:6.2f}%")

    # Per-frame modality detection
    describe_numeric(
        "Left hand frame fraction",
        full["left_frame_frac"].to_numpy(),
    )
    describe_numeric(
        "Right hand frame fraction",
        full["right_frame_frac"].to_numpy(),
    )
    describe_numeric(
        "Pose frame fraction",
        full["pose_frame_frac"].to_numpy(),
    )
    describe_numeric(
        "Face frame fraction",
        full["face_frame_frac"].to_numpy(),
    )

    # Per-sign frame stats
    sign_stats = (
        full.groupby("sign")
        .agg(
            n_samples=("sign", "size"),
            mean_frames=("n_frames", "mean"),
            median_frames=("n_frames", "median"),
            p90_frames=("n_frames", lambda x: np.percentile(x, 90)),
            max_frames=("n_frames", "max"),
            left_present_frac=("has_left_hand", "mean"),
            right_present_frac=("has_right_hand", "mean"),
            both_hands_frac=(
                "has_left_hand",
                lambda x: np.nan,
            ),
        )
        .reset_index()
    )

    # Add both_hands_frac cleanly
    both_by_sign = (
        full.assign(both_hands=full["has_left_hand"] & full["has_right_hand"])
        .groupby("sign")["both_hands"]
        .mean()
        .reset_index(name="both_hands_frac")
    )

    sign_stats = sign_stats.drop(columns=["both_hands_frac"]).merge(
        both_by_sign,
        on="sign",
        how="left",
    )

    sign_stats = sign_stats.sort_values("mean_frames", ascending=False)
    sign_stats.to_csv(OUT_DIR / "per_sign_stats.csv", index=False)

    print("\nSaved:")
    print(OUT_DIR / "per_sign_stats.csv")

    print("\nLongest signs by mean frame count")
    print("---------------------------------")
    print(sign_stats.head(20).to_string(index=False))

    print("\nShortest signs by mean frame count")
    print("----------------------------------")
    print(sign_stats.sort_values("mean_frames").head(20).to_string(index=False))

    # Participant stats
    participant_stats = (
        full.groupby("participant_id")
        .agg(
            n_samples=("participant_id", "size"),
            n_signs=("sign", "nunique"),
            mean_frames=("n_frames", "mean"),
            median_frames=("n_frames", "median"),
        )
        .reset_index()
        .sort_values("n_samples", ascending=False)
    )

    participant_stats.to_csv(OUT_DIR / "participant_stats.csv", index=False)

    print("\nSaved:")
    print(OUT_DIR / "participant_stats.csv")

    print("\nParticipants by sample count")
    print("----------------------------")
    print(participant_stats.head(20).to_string(index=False))

    # Class balance
    sign_counts = (
        full["sign"]
        .value_counts()
        .rename_axis("sign")
        .reset_index(name="n_samples")
    )

    sign_counts.to_csv(OUT_DIR / "sign_counts.csv", index=False)

    print("\nSaved:")
    print(OUT_DIR / "sign_counts.csv")

    print("\nClass balance")
    print("-------------")
    print("min samples per sign:", sign_counts["n_samples"].min())
    print("mean samples per sign:", sign_counts["n_samples"].mean())
    print("max samples per sign:", sign_counts["n_samples"].max())


if __name__ == "__main__":
    main()