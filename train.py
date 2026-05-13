import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path

PROCESSED = Path("processed")

N_SAMPLES = 94477
N_FRAMES = 64
FEATURE_DIM = 225
NUM_CLASSES = 250

X = np.memmap(
    PROCESSED / "X_64f_225d.dat",
    dtype=np.float32,
    mode="r",
    shape=(N_SAMPLES, N_FRAMES, FEATURE_DIM),
)

y = np.load(PROCESSED / "y_64f.npy")
meta = pd.read_csv(PROCESSED / "metadata_64f.csv")

groups = meta["participant_id"].values

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.15,
    random_state=42,
)

train_idx, val_idx = next(splitter.split(X, y, groups=groups))

print("Train samples:", len(train_idx))
print("Val samples:", len(val_idx))
print("Train participants:", meta.iloc[train_idx]["participant_id"].nunique())
print("Val participants:", meta.iloc[val_idx]["participant_id"].nunique())


def make_dataset(indices, batch_size=256, shuffle=False):
    indices = np.array(indices)

    def gen():
        for i in indices:
            yield X[i], y[i]

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(N_FRAMES, FEATURE_DIM), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int64),
        ),
    )

    if shuffle:
        ds = ds.shuffle(8192)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_dataset(train_idx, shuffle=True)
val_ds = make_dataset(val_idx, shuffle=False)


def build_model():
    inputs = tf.keras.Input(shape=(N_FRAMES, FEATURE_DIM))

    x = tf.keras.layers.Dense(128, activation="relu")(inputs)

    for filters, dilation in [
        (128, 1),
        (128, 2),
        (128, 4),
        (128, 8),
        (256, 1),
    ]:
        residual = x

        x = tf.keras.layers.SeparableConv1D(
            filters,
            kernel_size=5,
            padding="same",
            dilation_rate=dilation,
            activation="relu",
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.SeparableConv1D(
            filters,
            kernel_size=3,
            padding="same",
            activation="relu",
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)

        if residual.shape[-1] != filters:
            residual = tf.keras.layers.Dense(filters)(residual)

        x = tf.keras.layers.Add()([x, residual])
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Dropout(0.15)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs)


model = build_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

Path("models").mkdir(exist_ok=True)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "models/asl_tcn_64f_best.keras",
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-5,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=8,
        restore_best_weights=True,
    ),
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=callbacks,
)

model.save("models/asl_tcn_64f_final.keras")