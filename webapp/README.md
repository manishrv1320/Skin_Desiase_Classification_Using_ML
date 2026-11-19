# Skin Condition Classifier: web app

A small Flask app that serves the MobileNetV2 model from
`notebooks/SCIN_skin_condition_classification_baseline.ipynb`. You upload a photo, and it
returns the five most likely of the 20 SCIN conditions with a probability and a
short description for each.

## 1. Install

Windows (PowerShell):

```powershell
cd webapp
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-train.txt
```

macOS / Linux:

```bash
cd webapp
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-train.txt
```

`requirements-train.txt` adds pandas and scikit-learn for training. If you only
want to serve a model you already have, `requirements.txt` is enough.

## 2. Model

A trained model is already in `model/`, so you can skip to step 3. To train it
again, or to use one exported from Colab, use one of these:

**Train it locally (no Colab needed).**

```powershell
python scin_data.py     # downloads the SCIN images (5.6 GB, only needed once)
python train_model.py   # trains and saves the model (a few minutes on a CPU)
```

This repeats the notebook's baseline: the same SCIN label parsing, the same 20
classes, the same seed-42 case-level split, the same preprocessing, and the same
MobileNetV2 model and training settings, using only free, open-source software.

The only network use is downloading the dataset: 4,583 images (5.6 GB) from
the public SCIN bucket, with no Google account needed. They are saved
unchanged in `data/scin_images/`. If the download is interrupted, run
`scin_data.py` again and it carries on where it stopped. `train_model.py`
also downloads anything still missing before it trains. Once the model is
saved you can delete `data/scin_images/` to free the space; you only need it
again to retrain.

When it finishes, `model/` contains:

- `scin_skin_condition_classifier.keras`: the model
- `scin_class_names.json`: the class order
- `training_report.json`: test metrics, shown on the About page
- `scin_test_predictions.csv`: the prediction for every test image

Accuracy will be close to the Colab run but not identical, because training
involves random shuffling and dropout.

**Or copy it from Colab.** Run the notebook through section 13, then:

```python
from google.colab import files
files.download("scin_skin_condition_classifier.keras")
files.download("scin_class_names.json")
```

and put both files in `model/`.

## 3. Run

Double-click `start_app.bat`, or from the `webapp` folder with the venv active:

```powershell
python app.py
```

Open http://127.0.0.1:5000. Stop the server with Ctrl+C.

## Troubleshooting

**`DLL load failed ... An Application Control policy has blocked this file`**
when TensorFlow is imported. This is Windows 11 Smart App Control checking
TensorFlow's library files the first time they load. It can block a file on the
first try and allow it when the command is run again. If it keeps blocking,
check Windows Security > App & browser control > Smart App Control. Turning
Smart App Control off can't be undone without resetting Windows.

**pip install stalls or crawls.** Some networks limit each connection to
PyPI's file server to a few tens of KB/s, so the 350 MB TensorFlow package can
take hours. Downloads over many connections in parallel are much faster: fetch
the packages with a parallel downloader, check them against PyPI's sha256
hashes, then install offline with
`pip install --no-index --find-links <folder> -r requirements-train.txt`.

TensorFlow is pinned to 2.21.0, the version the notebook printed in Colab. If you
trained with a different version, check it in Colab with `tf.__version__` and
change the pin to match, otherwise the `.keras` file may fail to load.

## Configuration

| Variable           | Default                                       |
| ------------------ | --------------------------------------------- |
| `MODEL_PATH`       | `model/scin_skin_condition_classifier.keras`  |
| `CLASS_NAMES_PATH` | `model/scin_class_names.json`                 |
| `PORT`             | `5000`                                        |

If `scin_class_names.json` is missing, the app falls back to the 20 class names
from the notebook, in the same sorted order.

## API

`POST /api/predict` with a multipart field named `image` (JPEG, PNG, WebP or
BMP, up to 10 MB):

```json
{
  "predictions": [
    {"label": "Eczema", "probability": 0.2412, "summary": "...", "note": ""},
    ...
  ],
  "low_confidence": true
}
```

`low_confidence` is true when the top probability is under 40%.

`GET /api/health` reports whether the model loaded and how many classes it has.

## Preprocessing

Uploads go through the same steps as `decode_resize_preprocess()` in the
notebook: RGB, bilinear resize to 224 x 224, then MobileNetV2 scaling to
[-1, 1]. The app also applies the photo's EXIF rotation so phone pictures
aren't fed in sideways, and shrinks anything larger than 1080 px on its long
side (the largest size in SCIN) before the final resize.

Uploaded images are processed in memory and never written to disk.

## Files

```
app.py            routes and request handling
classifier.py     model loading, image preprocessing, prediction
scin_data.py      SCIN download, label parsing, cohort and split (no TensorFlow)
train_model.py    retrains the notebook's baseline locally
conditions.py     plain-language description of each condition
model_card.py     training and test numbers shown on the About page
templates/        Jinja templates
static/css/       stylesheet (light and dark)
static/js/        upload, drag-and-drop, paste, and results rendering
model/            the .keras model, class names and training report
data/             SCIN CSVs and downloaded images (created by scin_data.py)
```

The About page reads its numbers from `model/training_report.json` when it
exists, and otherwise shows the numbers from the Colab notebook run.
