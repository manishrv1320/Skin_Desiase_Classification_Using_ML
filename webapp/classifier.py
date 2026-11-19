import io
import json
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

# Order produced by sorted(CLASS_NAMES) in the training notebook.
# Used only when scin_class_names.json is missing.
DEFAULT_CLASS_NAMES = [
    "Acne",
    "Acute dermatitis, NOS",
    "Allergic Contact Dermatitis",
    "CD - Contact dermatitis",
    "Drug Rash",
    "Eczema",
    "Folliculitis",
    "Herpes Simplex",
    "Herpes Zoster",
    "Impetigo",
    "Insect Bite",
    "Irritant Contact Dermatitis",
    "Keratosis pilaris",
    "Lichen Simplex Chronicus",
    "Pigmented purpuric eruption",
    "Pityriasis rosea",
    "Psoriasis",
    "Tinea",
    "Tinea Versicolor",
    "Urticaria",
]

ACCEPTED_FORMATS = {"JPEG", "MPO", "PNG", "WEBP", "BMP"}

# SCIN training photos are at most 1080 px on the long side. Bigger uploads are
# brought down to that size first, so the final resize to 224 px behaves the
# way it did in training (and a 48 MP photo doesn't need gigabytes of memory).
MAX_WORKING_SIDE = 1080


class ModelNotAvailable(RuntimeError):
    pass


class InvalidImage(ValueError):
    pass


class SkinClassifier:
    def __init__(self, model_path, class_names_path):
        self.model_path = Path(model_path)
        self.class_names_path = Path(class_names_path)
        self.model = None
        self.class_names = []
        self.input_size = (224, 224)
        self.error = None

        try:
            self._load()
        except Exception as exc:
            self.model = None
            self.error = f"Could not load the model: {exc}"

    @property
    def ready(self):
        return self.model is not None

    def _load(self):
        if not self.model_path.is_file():
            self.error = f"Model file not found: {self.model_path}"
            return

        model = keras.saving.load_model(self.model_path, compile=False)

        _, height, width, _ = model.input_shape
        if height and width:
            self.input_size = (int(height), int(width))

        num_outputs = int(model.output_shape[-1])
        self.class_names = self._read_class_names(num_outputs)

        model(np.zeros((1, *self.input_size, 3), dtype=np.float32), training=False)
        self.model = model

    def _read_class_names(self, num_outputs):
        if self.class_names_path.is_file():
            names = json.loads(self.class_names_path.read_text(encoding="utf-8"))
        elif num_outputs == len(DEFAULT_CLASS_NAMES):
            names = DEFAULT_CLASS_NAMES
        else:
            raise ValueError(
                f"{self.class_names_path.name} is missing and the model has "
                f"{num_outputs} outputs, so the default class list can't be used."
            )

        if len(names) != num_outputs:
            raise ValueError(
                f"The model has {num_outputs} outputs but "
                f"{self.class_names_path.name} lists {len(names)} classes."
            )
        return [str(name) for name in names]

    def predict(self, image_bytes, top_k=5):
        if not self.ready:
            raise ModelNotAvailable(self.error or "Model is not loaded.")

        image = self._open_image(image_bytes)
        batch = self._to_batch(image)
        probs = self.model(batch, training=False).numpy()[0]

        order = np.argsort(probs)[::-1][:top_k]
        return [(self.class_names[i], float(probs[i])) for i in order]

    @staticmethod
    def _open_image(image_bytes):
        if not image_bytes:
            raise InvalidImage("The uploaded file is empty.")
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise InvalidImage("The file could not be read as an image.") from exc

        if image.format not in ACCEPTED_FORMATS:
            raise InvalidImage(
                f"{image.format or 'This'} images are not supported. "
                "Use JPEG, PNG, WebP or BMP."
            )

        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")

    def _to_batch(self, image):
        image.thumbnail((MAX_WORKING_SIDE, MAX_WORKING_SIDE), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.uint8)

        # Same steps as decode_resize_preprocess() in the notebook:
        # bilinear resize to the model input, then MobileNetV2 scaling to [-1, 1].
        resized = tf.image.resize(pixels, self.input_size)
        scaled = keras.applications.mobilenet_v2.preprocess_input(tf.cast(resized, tf.float32))
        return tf.expand_dims(scaled, 0)
