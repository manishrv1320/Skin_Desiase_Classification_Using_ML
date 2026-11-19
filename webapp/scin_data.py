"""SCIN download and cohort building, following sections 2-6 of the training notebook.

Does not need TensorFlow, so the (slow) image download can run on its own:

    python scin_data.py                # download the CSVs and every image the model uses
    python scin_data.py --workers 4    # fewer parallel downloads

Images are saved unchanged in data/scin_images/. Interrupted downloads resume
where they stopped. The SCIN bucket is public; no Google login is needed.
"""
import argparse
import ast
import hashlib
import http.client
import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
IMAGE_DIR = DATA_DIR / "scin_images"

BUCKET = "dx-scin-public-data"
BUCKET_URL = f"https://storage.googleapis.com/{BUCKET}"
CASES_CSV = "dataset/scin_cases.csv"
LABELS_CSV = "dataset/scin_labels.csv"

SEED = 42
MIN_CASES_PER_CLASS = 30
MIN_TOP_LABEL_WEIGHT = 0.0


def log(message):
    print(message, flush=True)


def fetch(object_path, retries=6):
    url = f"{BUCKET_URL}/{urllib.parse.quote(object_path)}"
    last_error = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=300) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise FileNotFoundError(object_path) from exc
            last_error = exc
        except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
            last_error = exc
        time.sleep(min(60, 2 ** attempt))
    raise last_error


def read_csv(object_path, data_dir=DATA_DIR):
    data_dir.mkdir(parents=True, exist_ok=True)
    local = data_dir / Path(object_path).name
    if not local.is_file():
        log(f"Downloading {object_path}")
        local.write_bytes(fetch(object_path))
    return pd.read_csv(local, dtype={"case_id": str})


def parse_weighted_label(value):
    if pd.isna(value):
        return None
    if isinstance(value, dict):
        parsed = value
    else:
        try:
            parsed = ast.literal_eval(str(value))
        except (ValueError, SyntaxError):
            return None
    if not isinstance(parsed, dict) or not parsed:
        return None
    cleaned = {}
    for name, weight in parsed.items():
        try:
            w = float(weight)
        except (TypeError, ValueError):
            continue
        if isinstance(name, str) and name.strip() and np.isfinite(w) and w >= 0:
            cleaned[name.strip()] = w
    return cleaned or None


def build_cohort(cases, labels):
    cases["case_id"] = cases["case_id"].astype(str)
    labels["case_id"] = labels["case_id"].astype(str)
    df = cases.merge(labels, on="case_id", how="inner", validate="one_to_one")

    df["parsed_weighted_label"] = df["weighted_skin_condition_label"].apply(parse_weighted_label)
    df["target_label"] = df["parsed_weighted_label"].apply(lambda d: max(d, key=d.get) if d else None)
    df["top_label_weight"] = df["parsed_weighted_label"].apply(lambda d: max(d.values()) if d else np.nan)

    cohort = df[df["target_label"].notna()].copy()
    cohort = cohort[cohort["top_label_weight"] >= MIN_TOP_LABEL_WEIGHT].copy()
    counts = cohort["target_label"].value_counts()
    kept = counts[counts >= MIN_CASES_PER_CLASS].index.tolist()
    cohort = cohort[cohort["target_label"].isin(kept)].copy()

    class_names = sorted(cohort["target_label"].unique().tolist())
    class_to_index = {name: i for i, name in enumerate(class_names)}
    cohort["label_id"] = cohort["target_label"].map(class_to_index).astype(int)

    stats = {
        "cases_total": int(len(df)),
        "cases_labelled": int(df["target_label"].notna().sum()),
        "cases_kept": int(len(cohort)),
    }
    return cohort, class_names, stats


def split_images(cohort):
    path_columns = [c for c in ["image_1_path", "image_2_path", "image_3_path"] if c in cohort.columns]
    case_table = cohort[["case_id", "target_label", "label_id"]].drop_duplicates("case_id").copy()

    train_cases, temp_cases = train_test_split(
        case_table, test_size=0.30, random_state=SEED, stratify=case_table["label_id"]
    )
    val_cases, test_cases = train_test_split(
        temp_cases, test_size=0.50, random_state=SEED, stratify=temp_cases["label_id"]
    )

    split_by_case = {}
    split_by_case.update({cid: "train" for cid in train_cases["case_id"]})
    split_by_case.update({cid: "validation" for cid in val_cases["case_id"]})
    split_by_case.update({cid: "test" for cid in test_cases["case_id"]})

    records = []
    for _, row in cohort.iterrows():
        for col in path_columns:
            path = row.get(col)
            if isinstance(path, str) and path.strip():
                records.append({
                    "case_id": str(row["case_id"]),
                    "gcs_path": path.strip(),
                    "target_label": row["target_label"],
                    "label_id": int(row["label_id"]),
                    "split": split_by_case[str(row["case_id"])],
                })
    return pd.DataFrame(records).drop_duplicates(subset=["gcs_path"]).reset_index(drop=True)


def local_image_path(gcs_path, image_dir=IMAGE_DIR):
    suffix = Path(gcs_path).suffix.lower() or ".jpg"
    key = hashlib.sha256(gcs_path.encode("utf-8")).hexdigest()
    return image_dir / f"{key}{suffix}"


def download_image(gcs_path, dest):
    data = fetch(gcs_path)
    with Image.open(io.BytesIO(data)) as im:
        im.verify()
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return len(data)


def cache_images(images_df, image_dir=IMAGE_DIR, workers=8):
    """Download any missing images. Returns images_df with local_path, minus images absent from the bucket."""
    image_dir.mkdir(parents=True, exist_ok=True)
    missing_log = image_dir / "missing_from_bucket.json"
    known_missing = set(json.loads(missing_log.read_text())) if missing_log.is_file() else set()

    paths = [local_image_path(p, image_dir) for p in images_df["gcs_path"]]
    todo = [
        (g, p) for g, p in zip(images_df["gcs_path"], paths)
        if g not in known_missing and not (p.is_file() and p.stat().st_size > 0)
    ]
    log(f"Images: {len(paths)} needed, {len(paths) - len(todo) - len(known_missing)} already downloaded, "
        f"{len(todo)} to download")

    failed = {}
    received = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_image, g, p): g for g, p in todo}
        for done, future in enumerate(as_completed(futures), 1):
            gcs_path = futures[future]
            try:
                received += future.result()
            except FileNotFoundError:
                known_missing.add(gcs_path)
                log(f"  not in bucket, skipped: {gcs_path}")
            except Exception as exc:
                failed[gcs_path] = f"{type(exc).__name__}: {exc}"
            if done % 25 == 0 or done == len(todo):
                elapsed = max(time.time() - started, 1e-6)
                rate = done / elapsed
                eta_min = (len(todo) - done) / max(rate, 1e-6) / 60
                log(f"  {done}/{len(todo)} images, {received / 1e6:.0f} MB, "
                    f"{received / 1e6 / elapsed:.2f} MB/s, about {eta_min:.0f} min left")

    missing_log.write_text(json.dumps(sorted(known_missing), indent=2))

    if failed:
        log(f"{len(failed)} downloads failed, for example:")
        for path, err in list(failed.items())[:5]:
            log(f"  {path}: {err}")
        log("Run the script again to retry them. Finished images are kept.")
        sys.exit(1)

    images_df = images_df.copy()
    images_df["local_path"] = [str(p) for p in paths]
    return images_df[~images_df["gcs_path"].isin(known_missing)].reset_index(drop=True)


def load_dataset(workers=8):
    """Everything up to and including the image download. Returns (images_df, class_names, stats)."""
    cases = read_csv(CASES_CSV)
    labels = read_csv(LABELS_CSV)
    cohort, class_names, stats = build_cohort(cases, labels)
    log(f"{stats['cases_labelled']} of {stats['cases_total']} cases have a usable label; "
        f"{stats['cases_kept']} kept across {len(class_names)} classes")
    images_df = cache_images(split_images(cohort), workers=workers)
    stats["images"] = int(len(images_df))
    return images_df, class_names, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=8, help="parallel image downloads")
    args = parser.parse_args()

    images_df, class_names, _ = load_dataset(workers=args.workers)
    for name in ("train", "validation", "test"):
        frame = images_df[images_df["split"] == name]
        log(f"{name}: {len(frame)} images, {frame['case_id'].nunique()} cases")
    log("Download complete. Next: python train_model.py")


if __name__ == "__main__":
    main()
