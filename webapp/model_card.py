import json
from pathlib import Path

REPORT_PATH = Path(__file__).resolve().parent / "model" / "training_report.json"

INPUT = "224 x 224 RGB, scaled to [-1, 1]"
LABEL = (
    "Highest-weight condition in the dermatologists' weighted differential "
    "(weighted_skin_condition_label)"
)

# What the Colab run of the notebook printed. Shown until train_model.py has
# written model/training_report.json for a locally trained model.
NOTEBOOK_REPORT = {
    "source": "The numbers on this page are from the Colab run of the training notebook.",
    "best_epoch": 5,
    "epochs_run": 10,
    "data": {
        "cases_total": 5033,
        "cases_labelled": 3061,
        "cases_kept": 2140,
        "images": 4583,
        "train": [3209, 1493],
        "validation": [685, 321],
        "test": [689, 321],
    },
    "test": {"accuracy": 0.1829, "balanced_accuracy": 0.0885, "macro_f1": 0.08, "weighted_f1": 0.16},
    "per_class": [
        {"name": n, "precision": p, "recall": r, "f1": f, "support": s}
        for n, p, r, f, s in [
            ("Acne", 0.20, 0.12, 0.15, 17),
            ("Acute dermatitis, NOS", 0.00, 0.00, 0.00, 15),
            ("Allergic Contact Dermatitis", 0.16, 0.38, 0.23, 94),
            ("CD - Contact dermatitis", 0.00, 0.00, 0.00, 15),
            ("Drug Rash", 0.08, 0.12, 0.10, 17),
            ("Eczema", 0.28, 0.31, 0.30, 165),
            ("Folliculitis", 0.24, 0.33, 0.27, 43),
            ("Herpes Simplex", 0.50, 0.05, 0.09, 20),
            ("Herpes Zoster", 0.08, 0.05, 0.06, 21),
            ("Impetigo", 0.00, 0.00, 0.00, 23),
            ("Insect Bite", 0.09, 0.07, 0.08, 54),
            ("Irritant Contact Dermatitis", 0.00, 0.00, 0.00, 12),
            ("Keratosis pilaris", 0.00, 0.00, 0.00, 13),
            ("Lichen Simplex Chronicus", 0.00, 0.00, 0.00, 8),
            ("Pigmented purpuric eruption", 0.50, 0.10, 0.17, 20),
            ("Pityriasis rosea", 0.00, 0.00, 0.00, 15),
            ("Psoriasis", 0.15, 0.09, 0.11, 35),
            ("Tinea", 0.00, 0.00, 0.00, 28),
            ("Tinea Versicolor", 0.00, 0.00, 0.00, 11),
            ("Urticaria", 0.12, 0.16, 0.14, 63),
        ]
    ],
}


def _pct(value):
    return f"{value * 100:.1f}%"


def _count(value):
    return f"{value:,}"


def load():
    report = NOTEBOOK_REPORT
    if REPORT_PATH.is_file():
        try:
            report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return summarize(report)


def summarize(report):
    data, test, per_class = report["data"], report["test"], report["per_class"]
    num_classes = len(per_class)
    test_images = data["test"][0]
    largest = max(per_class, key=lambda c: c["support"])

    def split(name):
        images, cases = data[name]
        return f"{_count(images)} images / {_count(cases)} cases"

    return {
        "source": report["source"],
        "num_classes": num_classes,
        "test_images": test_images,
        "model": [
            ("Architecture", "MobileNetV2 (ImageNet weights, frozen) + global average pooling "
                             f"+ dropout 0.35 + softmax over {num_classes} conditions"),
            ("Input", INPUT),
            ("Training", "Adam, learning rate 1e-3, batch size 32, early stopping on "
                         f"validation loss (best epoch {report['best_epoch']} of {report['epochs_run']})"),
            ("Target label", LABEL),
        ],
        "data": [
            ("SCIN cases in the release", _count(data["cases_total"])),
            ("Cases with a usable dermatologist label", _count(data["cases_labelled"])),
            ("Cases kept (classes with at least 30 cases)", _count(data["cases_kept"])),
            ("Images used", _count(data["images"])),
            ("Training split", split("train")),
            ("Validation split", split("validation")),
            ("Test split", split("test")),
        ],
        "test_metrics": [
            ("Accuracy", _pct(test["accuracy"])),
            ("Balanced accuracy", _pct(test["balanced_accuracy"])),
            ("Macro F1", f"{test['macro_f1']:.2f}"),
            ("Weighted F1", f"{test['weighted_f1']:.2f}"),
        ],
        "per_class": per_class,
        "majority_label": largest["name"],
        "majority_share": _pct(largest["support"] / test_images),
        "chance": _pct(1 / num_classes),
        "never_correct": sum(1 for c in per_class if c["recall"] == 0),
    }
