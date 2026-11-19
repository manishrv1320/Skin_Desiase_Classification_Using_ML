The trained model the app loads:

- `scin_skin_condition_classifier.keras`: MobileNetV2 baseline from the notebook
- `scin_class_names.json`: class order of the model outputs
- `training_report.json`: test metrics shown on the About page

`python train_model.py` writes all three again (plus `scin_test_predictions.csv`).
To load a model from somewhere else, set the `MODEL_PATH` and
`CLASS_NAMES_PATH` environment variables before starting the app.
