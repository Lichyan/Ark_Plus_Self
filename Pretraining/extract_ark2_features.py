#!/usr/bin/env python3
"""Batch extract Ark+ encoder/projector features for MIMIC-CXR JPG images."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
PRETRAINING_DIR = REPO_ROOT / "Pretraining"
if str(PRETRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(PRETRAINING_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

from Pretraining.models import build_omni_model_from_checkpoint  # noqa: E402


DEFAULT_MIMIC_ROOT = Path(
    "/media/daiju/f76132cd-833d-4c51-955a-e444dc79f8db/dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/media/daiju/f76132cd-833d-4c51-955a-e444dc79f8db/dataset/physionet.org/files/mimic-cxr-jpg/ark2_feat/"
)
DEFAULT_NUM_CLASSES_LIST = (14, 14, 14, 3, 6, 1)


@dataclass
class ImageSample:
    image_path: Path
    relative_path: Path
    subject_id: str
    study_id: str
    image_filename: str


class MIMICImageDataset(Dataset):
    def __init__(self, samples: Sequence[ImageSample], transform) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        try:
            image = Image.open(sample.image_path).convert("RGB")
            tensor = self.transform(image)
            return {
                "ok": True,
                "tensor": tensor,
                "sample": sample,
                "error_type": "",
                "error_message": "",
            }
        except Exception as err:  # noqa: BLE001 - continue on bad samples
            return {
                "ok": False,
                "tensor": None,
                "sample": sample,
                "error_type": "image_load_failed",
                "error_message": str(err),
            }


def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    text = str(v).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {v}")


def parse_num_classes_list(raw: Optional[str]) -> Sequence[int]:
    if raw is None:
        return DEFAULT_NUM_CLASSES_LIST
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--num_classes_list cannot be empty")
    return tuple(values)


def build_transform(input_size: int, normalization: str = "imagenet"):
    ops: List[object] = [transforms.Resize((input_size, input_size)), transforms.ToTensor()]
    if normalization.lower() == "imagenet":
        ops.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    elif normalization.lower() == "none":
        pass
    else:
        raise ValueError(f"Unsupported normalization: {normalization}")
    return transforms.Compose(ops)


def discover_samples(
    mimic_root: Path,
    start_prefix: int,
    end_prefix: int,
    max_images: Optional[int],
    error_rows: List[dict],
) -> List[ImageSample]:
    samples: List[ImageSample] = []

    for prefix in range(start_prefix, end_prefix + 1):
        top_dir = mimic_root / f"p{prefix}"
        if not top_dir.is_dir():
            error_rows.append(
                {
                    "path": str(top_dir),
                    "error_type": "prefix_dir_missing",
                    "error_message": "prefix directory does not exist",
                }
            )
            continue

        subject_dirs = sorted([p for p in top_dir.iterdir() if p.is_dir() and p.name.startswith("p")])
        for subject_dir in subject_dirs:
            study_dirs = sorted([p for p in subject_dir.iterdir() if p.is_dir() and p.name.startswith("s")])
            for study_dir in study_dirs:
                jpg_files = sorted(list(study_dir.glob("*.jpg")))
                if not jpg_files:
                    error_rows.append(
                        {
                            "path": str(study_dir),
                            "error_type": "no_jpg_in_study_dir",
                            "error_message": "no .jpg file found in study directory",
                        }
                    )
                    continue

                subject_id = subject_dir.name[1:] if subject_dir.name.startswith("p") else subject_dir.name
                study_id = study_dir.name[1:] if study_dir.name.startswith("s") else study_dir.name
                for img in jpg_files:
                    rel = img.relative_to(mimic_root)
                    samples.append(
                        ImageSample(
                            image_path=img,
                            relative_path=rel,
                            subject_id=subject_id,
                            study_id=study_id,
                            image_filename=img.name,
                        )
                    )
                    if max_images is not None and len(samples) >= max_images:
                        return samples
    return samples


def discover_samples_from_csv(
    csv_path: Path,
    images_root: Path,
    path_column: str,
    max_images: Optional[int],
    error_rows: List[dict],
) -> List[ImageSample]:
    samples: List[ImageSample] = []
    if not csv_path.is_file():
        error_rows.append(
            {
                "path": str(csv_path),
                "error_type": "csv_missing",
                "error_message": "csv file does not exist",
            }
        )
        return samples

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            error_rows.append(
                {
                    "path": str(csv_path),
                    "error_type": "csv_empty",
                    "error_message": "csv header is missing",
                }
            )
            return samples
        if path_column not in reader.fieldnames:
            error_rows.append(
                {
                    "path": str(csv_path),
                    "error_type": "csv_column_missing",
                    "error_message": f"missing required column: {path_column}",
                }
            )
            return samples

        for row in reader:
            rel_raw = str(row.get(path_column, "")).strip()
            if not rel_raw:
                continue

            rel_path = Path(rel_raw)
            img_abs = (images_root / rel_path).resolve()
            if not img_abs.is_file():
                error_rows.append(
                    {
                        "path": str(img_abs),
                        "error_type": "image_path_missing",
                        "error_message": f"path from csv not found: {rel_raw}",
                    }
                )
                continue

            parts = list(rel_path.parts)
            subject_id = ""
            study_id = ""
            if len(parts) >= 3 and parts[-3].startswith("patient"):
                subject_id = parts[-3].replace("patient", "")
            if len(parts) >= 2 and parts[-2].startswith("study"):
                study_id = parts[-2].replace("study", "")

            samples.append(
                ImageSample(
                    image_path=img_abs,
                    relative_path=rel_path,
                    subject_id=subject_id,
                    study_id=study_id,
                    image_filename=img_abs.name,
                )
            )
            if max_images is not None and len(samples) >= max_images:
                break
    return samples


def collate_keep(items: Sequence[dict]):
    oks = [x for x in items if x["ok"]]
    fails = [x for x in items if not x["ok"]]
    batch = None
    if oks:
        batch = torch.stack([x["tensor"] for x in oks], dim=0)
    return {"oks": oks, "fails": fails, "batch": batch}


def build_model(args: argparse.Namespace):
    model_args = SimpleNamespace(
        model_name=args.model_name,
        projector_features=args.projector_features,
        use_mlp=args.use_mlp,
        pretrained_weights=args.weights,
    )
    model = build_omni_model_from_checkpoint(
        model_args,
        num_classes_list=args.num_classes_list,
        key=args.checkpoint_key,
    )
    model.to(args.device)
    model.eval()
    return model


def feature_dim_from_existing(path: Path) -> Optional[int]:
    try:
        arr = np.load(path, mmap_mode="r")
        if arr.ndim == 0:
            return 1
        return int(arr.shape[-1])
    except Exception:
        return None


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Ark+ encoder/projector features from MIMIC-CXR JPG")
    parser.add_argument("--input_mode", choices=["auto", "mimic_tree", "csv"], default="auto")
    parser.add_argument("--mimic_root", type=Path, default=DEFAULT_MIMIC_ROOT)
    parser.add_argument("--images_root", type=Path, default=None, help="image root used by csv mode")
    parser.add_argument("--csv_path", type=Path, default=None, help="csv path used by csv mode")
    parser.add_argument("--csv_path_col", default="Path", help="column name containing relative image paths")
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--checkpoint_key", default="teacher")
    parser.add_argument("--model_name", default="swin_large_768")
    parser.add_argument("--projector_features", type=int, default=1376)
    parser.add_argument("--input_size", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_encoder", type=str2bool, default=True)
    parser.add_argument("--save_projector", type=str2bool, default=True)
    parser.add_argument("--skip_existing", type=str2bool, default=True)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--start_prefix", type=int, default=10)
    parser.add_argument("--end_prefix", type=int, default=19)
    parser.add_argument("--normalization", default="imagenet")
    parser.add_argument("--use_mlp", type=str2bool, default=False)
    parser.add_argument("--num_classes_list", default=None)
    args = parser.parse_args()

    args.weights = args.weights.expanduser().resolve()
    args.mimic_root = args.mimic_root.expanduser().resolve()
    args.images_root = (args.images_root if args.images_root is not None else args.mimic_root).expanduser().resolve()
    args.csv_path = args.csv_path.expanduser().resolve() if args.csv_path is not None else None
    args.output_root = args.output_root.expanduser().resolve()
    args.device = torch.device(args.device)
    args.num_classes_list = parse_num_classes_list(args.num_classes_list)

    if not args.save_encoder and not args.save_projector:
        raise ValueError("At least one of --save_encoder/--save_projector must be True")
    return args


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    encoder_root = output_root / "encoder"
    projector_root = output_root / "projector"

    encoder_index_path = output_root / "encoder_features_index.csv"
    projector_index_path = output_root / "projector_features_index.csv"
    error_csv_path = output_root / "extract_errors.csv"

    error_rows: List[dict] = []

    use_csv_mode = args.input_mode == "csv" or (args.input_mode == "auto" and args.csv_path is not None)
    if use_csv_mode:
        samples = discover_samples_from_csv(
            csv_path=args.csv_path if args.csv_path is not None else args.images_root / "test.csv",
            images_root=args.images_root,
            path_column=args.csv_path_col,
            max_images=args.max_images,
            error_rows=error_rows,
        )
    else:
        samples = discover_samples(
            mimic_root=args.mimic_root,
            start_prefix=args.start_prefix,
            end_prefix=args.end_prefix,
            max_images=args.max_images,
            error_rows=error_rows,
        )
    print(f"[INFO] discovered {len(samples)} images")
    if not samples:
        write_csv(error_csv_path, ["path", "error_type", "error_message"], error_rows)
        print("[WARN] no image found. error log written.")
        return

    transform = build_transform(input_size=args.input_size, normalization=args.normalization)
    dataset = MIMICImageDataset(samples, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.type == "cuda",
        collate_fn=collate_keep,
    )

    model = build_model(args)

    encoder_rows: List[dict] = []
    projector_rows: List[dict] = []

    stats = Counter()
    printed_shape = False

    with torch.no_grad():
        for packed in tqdm(loader, desc="Extracting", unit="batch"):
            for fail in packed["fails"]:
                sample = fail["sample"]
                error_rows.append(
                    {
                        "path": str(sample.image_path),
                        "error_type": fail["error_type"],
                        "error_message": fail["error_message"],
                    }
                )
                stats["failed"] += 1
                stats[fail["error_type"]] += 1

            oks = packed["oks"]
            if not oks:
                continue

            batch = packed["batch"].to(args.device, non_blocking=True)

            try:
                encoder_feats = model.generate_embeddings(batch, after_proj=False) if args.save_encoder else None
                projector_feats = model.generate_embeddings(batch, after_proj=True) if args.save_projector else None
            except Exception as err:  # noqa: BLE001
                for ok in oks:
                    sample = ok["sample"]
                    error_rows.append(
                        {
                            "path": str(sample.image_path),
                            "error_type": "feature_extract_failed",
                            "error_message": str(err),
                        }
                    )
                    stats["failed"] += 1
                    stats["feature_extract_failed"] += 1
                continue

            if args.save_encoder:
                if encoder_feats is None or encoder_feats.ndim != 2 or encoder_feats.shape[0] != len(oks):
                    for ok in oks:
                        sample = ok["sample"]
                        error_rows.append(
                            {
                                "path": str(sample.image_path),
                                "error_type": "unexpected_shape",
                                "error_message": f"encoder shape: {None if encoder_feats is None else tuple(encoder_feats.shape)}",
                            }
                        )
                        stats["failed"] += 1
                        stats["unexpected_shape"] += 1
                    continue

            if args.save_projector:
                if projector_feats is None or projector_feats.ndim != 2 or projector_feats.shape[0] != len(oks):
                    for ok in oks:
                        sample = ok["sample"]
                        error_rows.append(
                            {
                                "path": str(sample.image_path),
                                "error_type": "unexpected_shape",
                                "error_message": f"projector shape: {None if projector_feats is None else tuple(projector_feats.shape)}",
                            }
                        )
                        stats["failed"] += 1
                        stats["unexpected_shape"] += 1
                    continue

            encoder_np = encoder_feats.detach().cpu().numpy() if encoder_feats is not None else None
            projector_np = projector_feats.detach().cpu().numpy() if projector_feats is not None else None

            if not printed_shape:
                if encoder_np is not None:
                    print(f"[INFO] encoder first-batch shape: {tuple(encoder_np.shape)}")
                if projector_np is not None:
                    print(f"[INFO] projector first-batch shape: {tuple(projector_np.shape)}")
                printed_shape = True

            for i, ok in enumerate(oks):
                sample = ok["sample"]
                rel_npy = sample.relative_path.with_suffix(".npy")

                if args.save_encoder:
                    enc_out = encoder_root / rel_npy
                    enc_out.parent.mkdir(parents=True, exist_ok=True)
                    enc_dim: Optional[int] = None
                    try:
                        if args.skip_existing and enc_out.exists():
                            enc_dim = feature_dim_from_existing(enc_out)
                            stats["encoder_skipped_existing"] += 1
                        else:
                            np.save(enc_out, encoder_np[i].astype(np.float32))
                            enc_dim = int(encoder_np[i].shape[-1])
                            stats["encoder_saved"] += 1
                    except Exception as err:  # noqa: BLE001
                        error_rows.append(
                            {
                                "path": str(sample.image_path),
                                "error_type": "save_failed",
                                "error_message": f"encoder save failed: {err}",
                            }
                        )
                        stats["failed"] += 1
                        stats["save_failed"] += 1
                    else:
                        encoder_rows.append(
                            {
                                "image_path": str(sample.image_path),
                                "feature_path": str(enc_out),
                                "feature_dim": enc_dim if enc_dim is not None else -1,
                                "subject_id": sample.subject_id,
                                "study_id": sample.study_id,
                                "image_filename": sample.image_filename,
                                "relative_path_from_mimic_root": str(sample.relative_path),
                            }
                        )

                if args.save_projector:
                    proj_out = projector_root / rel_npy
                    proj_out.parent.mkdir(parents=True, exist_ok=True)
                    proj_dim: Optional[int] = None
                    try:
                        if args.skip_existing and proj_out.exists():
                            proj_dim = feature_dim_from_existing(proj_out)
                            stats["projector_skipped_existing"] += 1
                        else:
                            np.save(proj_out, projector_np[i].astype(np.float32))
                            proj_dim = int(projector_np[i].shape[-1])
                            stats["projector_saved"] += 1
                    except Exception as err:  # noqa: BLE001
                        error_rows.append(
                            {
                                "path": str(sample.image_path),
                                "error_type": "save_failed",
                                "error_message": f"projector save failed: {err}",
                            }
                        )
                        stats["failed"] += 1
                        stats["save_failed"] += 1
                    else:
                        projector_rows.append(
                            {
                                "image_path": str(sample.image_path),
                                "feature_path": str(proj_out),
                                "feature_dim": proj_dim if proj_dim is not None else -1,
                                "subject_id": sample.subject_id,
                                "study_id": sample.study_id,
                                "image_filename": sample.image_filename,
                                "relative_path_from_mimic_root": str(sample.relative_path),
                            }
                        )

                stats["images_processed"] += 1

    feature_columns = [
        "image_path",
        "feature_path",
        "feature_dim",
        "subject_id",
        "study_id",
        "image_filename",
        "relative_path_from_mimic_root",
    ]
    if args.save_encoder:
        write_csv(encoder_index_path, feature_columns, encoder_rows)
    if args.save_projector:
        write_csv(projector_index_path, feature_columns, projector_rows)
    write_csv(error_csv_path, ["path", "error_type", "error_message"], error_rows)

    print("\n[SUMMARY]")
    print(f"successful images processed: {stats['images_processed']}")
    print(f"encoder features rows: {len(encoder_rows)}")
    print(f"projector features rows: {len(projector_rows)}")
    print(f"failed count: {stats['failed']}")
    if error_rows:
        err_counter = Counter(row["error_type"] for row in error_rows)
        for err_type, cnt in sorted(err_counter.items()):
            print(f"  - {err_type}: {cnt}")
    print(f"encoder index: {encoder_index_path}")
    print(f"projector index: {projector_index_path}")
    print(f"error log: {error_csv_path}")


if __name__ == "__main__":
    main()
