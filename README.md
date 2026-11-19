# Skin Condition Classification (SCIN)

Image classifier for 20 common skin conditions, trained on Google Research's
SCIN dataset, with a small Flask web app for trying it on a photo.

For research and education only. Predictions are not a medical diagnosis.

## Contents

```
notebooks/   training notebook (Colab): data preparation, MobileNetV2 baseline,
             and the fine-tuning / EfficientNet / focal loss experiments
webapp/      Flask app, local training scripts and the trained model
```

## Run the app

The trained model is included in `webapp/model/`, so the app runs without any
training. Windows:

```powershell
cd webapp
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000. On macOS/Linux, use `python3 -m venv .venv`
and `source .venv/bin/activate`.

## Retrain

```powershell
pip install -r requirements-train.txt
python scin_data.py      # downloads 4,583 SCIN images (5.6 GB), resumable
python train_model.py    # a few minutes on a CPU
```

This repeats the notebook's baseline without Colab: the same label parsing, the
same 20 classes, the same case-level split (seed 42) and the same MobileNetV2
settings. See [webapp/README.md](webapp/README.md) for details.

## Results

Held-out test set: 689 images from 321 cases. The split is by case, so photos of
the same person never appear in both training and test data.

| Model                                   | Accuracy | Balanced accuracy | Correct in top 5 |
| --------------------------------------- | -------- | ----------------- | ---------------- |
| MobileNetV2 baseline, retrained locally | 22.4%    | 8.2%              | 62.3%            |
| Same baseline, Colab notebook run       | 18.3%    | 8.9%              | not measured     |

For comparison, always predicting the most common class (Eczema) scores 23.9%,
and random guessing scores 5%.

The labels themselves are uncertain. Each case's label is the top diagnosis in
the dermatologists' weighted differential, and the dermatologists fully agreed
on only 23% of cases (the median weight of the top diagnosis is 0.55). That puts
a low ceiling on single-label accuracy for this dataset.

## Data

[SCIN](https://github.com/google-research-datasets/scin) (Skin Condition Image
Network) by Google Research. The images are not included in this repository;
`webapp/scin_data.py` downloads them from the public SCIN bucket. Use of the
data is subject to the SCIN data use license.
