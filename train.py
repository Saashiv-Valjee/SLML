import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path

# ============================================================
# Mixed precision
# ============================================================

tf.keras.mixed_precision.set_global_policy("mixed_float16")

# ============================================================
# Paths / constants
# ============================================================

PROCESSED = Path("processed")
MODELS = Path("models")

MODELS.mkdir(exist_ok=True)

N_SAMPLES = 94477
N_FRAMES = 64
FEATURE_DIM = 225
NUM_CLASSES = 250

# ============================================================
# GPU memory growth
# ============================================================

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        print("GPUs detected:")
        for gpu in gpus:
            print(" ", gpu)

    except RuntimeError as e:
        print(e)

# ============================================================
# Load dataset
# ============================================================

X = np.memmap(
    PROCESSED / "X_64f_225d.dat",
    dtype=np.float32,
    mode="r",
    shape=(N_SAMPLES, N_FRAMES, FEATURE_DIM),
)

y = np.load(PROCESSED / "y_64f.npy")

meta = pd.read_csv(PROCESSED / "metadata_64f.csv")

# ============================================================
# Save label mapping
# ============================================================

label_map = (
    meta[["label", "sign"]]
    .drop_duplicates()
    .sort_values("label")
)

label_map.to_csv(MODELS / "label_map.csv", index=False)

print("\nSaved label map:")
print(MODELS / "label_map.csv")

# ============================================================
# Participant-level split
# ============================================================

groups = meta["participant_id"].values

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.15,
    random_state=42,
)

train_idx, val_idx = next(
    splitter.split(X, y, groups=groups)
)

print("\nTrain samples:", len(train_idx))
print("Val samples  :", len(val_idx))

print(
    "Train participants:",
    meta.iloc[train_idx]["participant_id"].nunique(),
)

print(
    "Val participants:",
    meta.iloc[val_idx]["participant_id"].nunique(),
)

# ============================================================
# tf.data
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE


def make_dataset(indices, batch_size=256, shuffle=False):

    indices = np.array(indices)

    def gen():
        for i in indices:
            yield X[i], y[i]

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(
                shape=(N_FRAMES, FEATURE_DIM),
                dtype=tf.float32,
            ),
            tf.TensorSpec(
                shape=(),
                dtype=tf.int64,
            ),
        ),
    )

    if shuffle:
        ds = ds.shuffle(8192)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)

    return ds


train_ds = make_dataset(
    train_idx,
    shuffle=True,
)

val_ds = make_dataset(
    val_idx,
    shuffle=False,
)

# ============================================================
# TCN model
# ============================================================


def tcn_block(
    x,
    filters,
    dilation,
    kernel_size=5,
    dropout=0.15,
):

    residual = x

    x = tf.keras.layers.SeparableConv1D(
        filters,
        kernel_size=kernel_size,
        padding="same",
        dilation_rate=dilation,
        use_bias=False,
        activation="relu",
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.SeparableConv1D(
        filters,
        kernel_size=3,
        padding="same",
        use_bias=False,
        activation="relu",
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    if residual.shape[-1] != filters:
        residual = tf.keras.layers.Dense(filters)(residual)

    x = tf.keras.layers.Add()([x, residual])

    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Dropout(dropout)(x)

    return x


def build_model():

    inputs = tf.keras.Input(
        shape=(N_FRAMES, FEATURE_DIM)
    )

    x = tf.keras.layers.Dense(
        128,
        activation="relu",
    )(inputs)

    # --------------------------------------------------------
    # TCN stack
    # --------------------------------------------------------

    for filters, dilation in [

        (128, 1),
        (128, 2),
        (128, 4),
        (128, 8),

        # Reduced from 256 → 128
        # Better latency / quantization
        (128, 1),

    ]:

        x = tcn_block(
            x,
            filters=filters,
            dilation=dilation,
        )

    # --------------------------------------------------------
    # Pooling head
    # --------------------------------------------------------

    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dense(
        256,
        activation="relu",
    )(x)

    x = tf.keras.layers.Dropout(0.30)(x)

    # --------------------------------------------------------
    # IMPORTANT:
    # output forced to float32 for mixed precision stability
    # --------------------------------------------------------

    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        dtype="float32",
    )(x)

    return tf.keras.Model(inputs, outputs)


model = build_model()

# ============================================================
# Compile
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),

    loss="sparse_categorical_crossentropy",

    metrics=[

        "accuracy",

        tf.keras.metrics.SparseTopKCategoricalAccuracy(
            k=5,
            name="top5",
        ),

    ],
)

model.summary()

# ============================================================
# Callbacks
# ============================================================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(

        filepath=MODELS / "asl_tcn_64f_best.keras",

        monitor="val_accuracy",
        mode="max",

        save_best_only=True,

        verbose=1,
    ),

    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,
        patience=3,

        min_lr=1e-5,

        verbose=1,
    ),

    tf.keras.callbacks.EarlyStopping(

        monitor="val_accuracy",
        mode="max",

        patience=8,

        restore_best_weights=True,

        verbose=1,
    ),

    tf.keras.callbacks.CSVLogger(
        MODELS / "training_log.csv"
    ),

]

# ============================================================
# Train
# ============================================================

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=50,

    callbacks=callbacks,

)

# ============================================================
# Save final model
# ============================================================

model.save(
    MODELS / "asl_tcn_64f_final.keras"
)

print("\nSaved final model:")
print(MODELS / "asl_tcn_64f_final.keras")

# ============================================================
# Export SavedModel
# ============================================================

saved_model_dir = MODELS / "saved_model"

model.export(saved_model_dir)

print("\nSaved exported model:")
print(saved_model_dir)

# ============================================================
# Save config
# ============================================================

config = {

    "n_frames": N_FRAMES,

    "feature_dim": FEATURE_DIM,

    "num_classes": NUM_CLASSES,

    "landmarks": [

        "left_hand",
        "right_hand",
        "pose",

    ],

    "normalization": "torso_relative",

    "model_type": "TCN",

    "mixed_precision": True,

    "top5_metric": True,

}

import json

with open(
    MODELS / "preprocessing_config.json",
    "w",
) as f:

    json.dump(config, f, indent=2)

print("\nSaved config:")
print(MODELS / "preprocessing_config.json")