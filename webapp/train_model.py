"""Retrain the notebook's MobileNetV2 baseline on this computer and save it for the web app.

Follows sections 2-13 of SCIN_skin_condition_classification_baseline.ipynb: same
label parsing, cohort filter, case-level split (seed 42), preprocessing,
architecture and training settings (see scin_data.py for the data side).

The MobileNetV2 base is frozen and called with training=False, so its pooled
output for an image never changes during training. This script computes that
output once per image and fits only the dropout + dense head on top. The result
is the same model the notebook trains end to end, in minutes on a CPU instead of
hours.

usage:
    python train_model.py          # downloads anything missing, trains, saves into model/
"""
import argparse
import json
import random
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)

import keras
import tensorflow as tf
from keras import layers

import scin_data

HERE = Path(__file__).resolve().parent

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
DROPOUT = 0.35
LEARNING_RATE = 1e-3

MODEL_FILE = "scin_skin_condition_classifier.keras"
CLASS_NAMES_FILE = "scin_class_names.json"
REPORT_FILE = "training_report.json"
PREDICTIONS_FILE = "scin_test_predictions.csv"

log = scin_data.log


def decode_resize_preprocess(path):
    # Same steps as the notebook's input pipeline (section 7).
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    return keras.applications.mobilenet_v2.preprocess_input(image)


def image_dataset(paths):
    return (
        tf.data.Dataset.from_tensor_slices(list(paths))
        .map(decode_resize_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )


def make_base():
    base = keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(*IMAGE_SIZE, 3)
    )
    base.trainable = False
    return base


def extract_features(base, paths):
    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
    pooled = layers.GlobalAveragePooling2D()(base(inputs, training=False))
    return keras.Model(inputs, pooled).predict(image_dataset(paths), verbose=1)


def train_head(x_train, y_train, x_val, y_val, num_classes, checkpoint_path, epochs):
    head = keras.Sequential([
        layers.Input(shape=(x_train.shape[1],)),
        layers.Dropout(DROPOUT),
        layers.Dense(num_classes, activation="softmax", name="condition_probabilities"),
    ])
    head.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, keras.utils.to_categorical(y_train, num_classes)))
        .shuffle(min(len(x_train), 5000), seed=scin_data.SEED, reshuffle_each_iteration=True)
        .batch(BATCH_SIZE)
    )
    val_ds = tf.data.Dataset.from_tensor_slices(
        (x_val, keras.utils.to_categorical(y_val, num_classes))
    ).batch(BATCH_SIZE)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(checkpoint_path), monitor="val_loss", save_best_only=True, save_weights_only=True
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1
        ),
    ]
    history = head.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=2)
    head.load_weights(str(checkpoint_path))

    val_losses = history.history["val_loss"]
    return head, int(np.argmin(val_losses)) + 1, len(val_losses)


def assemble(base, head, num_classes):
    inputs = layers.Input(shape=(*IMAGE_SIZE, 3), name="skin_image")
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(DROPOUT)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="condition_probabilities")(x)
    model = keras.Model(inputs, outputs, name="SCIN_MobileNetV2_baseline")
    model.get_layer("condition_probabilities").set_weights(
        head.get_layer("condition_probabilities").get_weights()
    )
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=HERE / "model")
    parser.add_argument("--workers", type=int, default=8, help="parallel image downloads")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    random.seed(scin_data.SEED)
    np.random.seed(scin_data.SEED)
    keras.utils.set_random_seed(scin_data.SEED)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    images_df, class_names, stats = scin_data.load_dataset(workers=args.workers)
    num_classes = len(class_names)
    splits = {
        name: images_df[images_df["split"] == name].reset_index(drop=True)
        for name in ("train", "validation", "test")
    }
    for name, frame in splits.items():
        log(f"{name}: {len(frame)} images, {frame['case_id'].nunique()} cases")

    log("Computing MobileNetV2 features for every image")
    base = make_base()
    features = extract_features(base, images_df["local_path"])
    feats = {name: features[(images_df["split"] == name).to_numpy()] for name in splits}
    ys = {name: frame["label_id"].to_numpy() for name, frame in splits.items()}

    log("Training the classification head")
    checkpoint = scin_data.DATA_DIR / "head_best.weights.h5"
    head, best_epoch, epochs_run = train_head(
        feats["train"], ys["train"], feats["validation"], ys["validation"],
        num_classes, checkpoint, args.epochs,
    )

    probs = head.predict(feats["test"], verbose=0)
    y_true, y_pred = ys["test"], probs.argmax(axis=1)
    report = classification_report(
        y_true, y_pred, labels=np.arange(num_classes), target_names=class_names,
        zero_division=0, output_dict=True,
    )
    test_metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    log(classification_report(
        y_true, y_pred, labels=np.arange(num_classes), target_names=class_names, zero_division=0
    ))
    log("Test accuracy {accuracy:.4f} | balanced accuracy {balanced_accuracy:.4f}".format(**test_metrics))

    model = assemble(base, head, num_classes)
    sample = splits["test"]["local_path"].iloc[:64]
    drift = float(np.abs(model.predict(image_dataset(sample), verbose=0) - probs[: len(sample)]).max())
    if drift > 1e-3:
        raise RuntimeError(f"Assembled model disagrees with the trained head (max diff {drift:.2e})")

    model_path = args.model_dir / MODEL_FILE
    model.save(model_path)
    (args.model_dir / CLASS_NAMES_FILE).write_text(json.dumps(class_names, indent=2), encoding="utf-8")

    predictions = splits["test"][["case_id", "gcs_path", "target_label", "label_id"]].copy()
    predictions["predicted_label_id"] = y_pred
    predictions["predicted_label"] = [class_names[i] for i in y_pred]
    predictions["predicted_probability"] = probs.max(axis=1)
    predictions.to_csv(args.model_dir / PREDICTIONS_FILE, index=False)

    def split_counts(frame):
        return [int(len(frame)), int(frame["case_id"].nunique())]

    training_report = {
        "source": "The numbers on this page are from the model trained locally "
                  f"with train_model.py on {date.today().isoformat()}.",
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "data": {
            **stats,
            "train": split_counts(splits["train"]),
            "validation": split_counts(splits["validation"]),
            "test": split_counts(splits["test"]),
        },
        "test": test_metrics,
        "per_class": [
            {
                "name": name,
                "precision": round(report[name]["precision"], 4),
                "recall": round(report[name]["recall"], 4),
                "f1": round(report[name]["f1-score"], 4),
                "support": int(report[name]["support"]),
            }
            for name in class_names
        ],
    }
    (args.model_dir / REPORT_FILE).write_text(json.dumps(training_report, indent=2), encoding="utf-8")

    for name in (MODEL_FILE, CLASS_NAMES_FILE, REPORT_FILE, PREDICTIONS_FILE):
        log(f"Saved {args.model_dir / name}")
    log("Start the web app with: python app.py")


if __name__ == "__main__":
    main()
