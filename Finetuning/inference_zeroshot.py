"""Ark+ zero-shot inference helper.

This script mirrors the zero-shot workflow showcased in the Ark+ notebooks.
It keeps the pre-trained multi-task heads intact, decodes the CheXpert head
(probabilities for 14 classes), and exports both raw probabilities and
thresholded decisions.
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Ensure the repository root is on the import path so we can reuse the
# omni-pretraining model builders.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from Pretraining.models import build_omni_model_from_checkpoint  # noqa: E402

try:  # Optional dependency used in other parts of the project.
    from torch.serialization import add_safe_globals
except ImportError:  # pragma: no cover - compatibility for older PyTorch.
    add_safe_globals = None

if add_safe_globals is not None:
    try:
        import numpy as _np  # noqa: WPS433 - imported only for allow listing.

        add_safe_globals([_np.dtype, _np.core.multiarray.scalar])
    except Exception:  # pragma: no cover - best-effort safeguard.
        pass

CHEXPERT_CLASS_NAMES: Sequence[str] = (
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
)

MIMIC_CLASS_NAMES: Sequence[str] = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
)

NIH14_CLASS_NAMES: Sequence[str] = (
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
)

RSNA_CLASS_NAMES: Sequence[str] = (
    "No Lung Opacity/Not Normal",
    "Normal",
    "Lung Opacity",
)

VINDR_CLASS_NAMES: Sequence[str] = (
    "PE",
    "Lung tumor",
    "Pneumonia",
    "Tuberculosis",
    "Other diseases",
    "No finding",
)

SHENZHEN_CLASS_NAMES: Sequence[str] = ("TB",)

# The Ark+ pre-training head order showcased in the zero-shot notebook.
DEFAULT_HEAD_CATALOG: Sequence[Tuple[str, Sequence[str]]] = (
    ("mimic", MIMIC_CLASS_NAMES),
    ("chexpert", CHEXPERT_CLASS_NAMES),
    ("nih14", NIH14_CLASS_NAMES),
    ("rsna", RSNA_CLASS_NAMES),
    ("vindr", VINDR_CLASS_NAMES),
    ("shenzhen", SHENZHEN_CLASS_NAMES),
)

DEFAULT_NUM_CLASSES_LIST: Sequence[int] = tuple(len(names) for _, names in DEFAULT_HEAD_CATALOG)

# Ark+ 官方仓库未提供 CheXpert 验证集最优阈值的脚本或常量，因此默认使用 0.5。
ARKPLUS_CHEXPERT_THRESHOLDS: Optional[Sequence[float]] = None


class ZeroShotInferenceDataset(Dataset):
    """Dataset that only relies on the CSV first column for image paths."""

    def __init__(self, root_dir: Path, csv_path: Path, transform) -> None:
        self.root_dir = root_dir
        self.csv_path = csv_path
        self.transform = transform
        self.samples: List[Tuple[str, Path]] = []

        with csv_path.open("r", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue
                rel_path = row[0].strip()
                if not rel_path:
                    continue
                full_path = root_dir / rel_path
                if not full_path.is_file():
                    # 如果首行是表头，则跳过；否则提示错误以便用户修正。
                    if not self.samples:
                        continue
                    raise FileNotFoundError(
                        f"无法找到图像文件: {full_path}. 请检查CSV第一列路径是否与数据集根目录匹配。"
                    )
                self.samples.append((rel_path, full_path))

        if not self.samples:
            raise ValueError("CSV文件中未解析到有效图像路径，请确认第一列是否包含文件相对路径。")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        rel_path, full_path = self.samples[index]
        image = Image.open(full_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, rel_path

    @property
    def relative_paths(self) -> List[str]:
        return [rel for rel, _ in self.samples]


@dataclass
class InferenceResults:
    probabilities: np.ndarray
    binary: np.ndarray
    paths: List[str]
    class_names: Sequence[str]


def build_transform(input_size: int, normalization: str = "imagenet"):
    """构造与zero-shot notebook一致的推理图像预处理流程."""
    from torchvision import transforms

    resize_size = (input_size, input_size)
    ops = [transforms.Resize(resize_size), transforms.ToTensor()]

    if normalization.lower() == "imagenet":
        ops.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    elif normalization.lower() == "none":
        pass
    else:
        raise ValueError(f"暂不支持的归一化方案: {normalization}")

    return transforms.Compose(ops)


def parse_num_classes_list(raw: Optional[str]) -> Sequence[int]:
    if raw is None:
        return DEFAULT_NUM_CLASSES_LIST
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--num-classes-list 不能为空")
    return tuple(values)


def select_head_index(head: str, catalog: Sequence[Tuple[str, Sequence[str]]]) -> int:
    try:
        return next(idx for idx, (name, _) in enumerate(catalog) if name.lower() == head.lower())
    except StopIteration as exc:  # pragma: no cover - CLI validation
        available = ", ".join(name for name, _ in catalog)
        raise ValueError(f"未找到名为 '{head}' 的head，可选项为: {available}") from exc


def resolve_class_names(
    target_head: int,
    catalog: Sequence[Tuple[str, Sequence[str]]],
    override: Optional[Sequence[str]] = None,
) -> Sequence[str]:
    if override is not None:
        return list(override)
    _, names = catalog[target_head]
    return names


def _load_thresholds(path: Optional[str], expected: int) -> Optional[np.ndarray]:
    if path is None:
        return None
    thresholds_path = Path(path)
    if not thresholds_path.is_file():
        raise FileNotFoundError(f"无法读取阈值文件: {thresholds_path}")
    values: List[float] = []
    with thresholds_path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            for item in row:
                item = item.strip()
                if not item:
                    continue
                values.append(float(item))
    if len(values) != expected:
        raise ValueError(
            f"阈值数量({len(values)})与类别数({expected})不符，请检查阈值文件。"
        )
    return np.asarray(values, dtype=np.float32)


class ArkZeroShotRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device)

        self.head_catalog: Sequence[Tuple[str, Sequence[str]]] = DEFAULT_HEAD_CATALOG
        if len(self.args.num_classes_list) != len(self.head_catalog):
            # 若用户提供了自定义的多头结构，则仅保留类别数量信息，类别名称需要用户手动指定。
            self.head_catalog = tuple((f"head_{idx}", tuple()) for idx, _ in enumerate(self.args.num_classes_list))

        self.target_head_index = select_head_index(self.args.target_head, self.head_catalog)
        self.class_names = resolve_class_names(
            self.target_head_index,
            self.head_catalog,
            override=self.args.class_names,
        )
        if len(self.class_names) != self.args.num_classes_list[self.target_head_index]:
            if self.class_names:
                raise ValueError(
                    "提供的类别名称数量与目标head输出维度不一致，请检查 --class-names 配置。"
                )
            # 自动生成占位名称。
            self.class_names = [f"Class_{i}" for i in range(self.args.num_classes_list[self.target_head_index])]

        self.transform = build_transform(self.args.input_size, normalization=self.args.normalization)
        self.dataset = ZeroShotInferenceDataset(
            root_dir=Path(self.args.data_root),
            csv_path=Path(self.args.csv_path),
            transform=self.transform,
        )
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

        self.thresholds = self._prepare_thresholds()
        self._warn_thresholds_if_needed()

    def _prepare_thresholds(self) -> np.ndarray:
        num_classes = self.args.num_classes_list[self.target_head_index]
        thresholds = _load_thresholds(self.args.thresholds_file, num_classes)
        if thresholds is not None:
            return thresholds
        if ARKPLUS_CHEXPERT_THRESHOLDS is not None and self.args.target_head.lower() == "chexpert":
            return np.asarray(ARKPLUS_CHEXPERT_THRESHOLDS, dtype=np.float32)
        return np.full(num_classes, self.args.default_threshold, dtype=np.float32)

    def _warn_thresholds_if_needed(self) -> None:
        if self.args.thresholds_file is None and ARKPLUS_CHEXPERT_THRESHOLDS is None:
            if self.args.target_head.lower() == "chexpert":
                print("[提示] 仓库中未提供CheXpert验证集最优阈值，已退回使用默认阈值"
                      f" {self.args.default_threshold:.2f}。")

    def run(self) -> InferenceResults:
        model = self._build_model().to(self.device)
        model.eval()

        probabilities: List[np.ndarray] = []
        with torch.no_grad():
            for images, _ in tqdm(self.loader, desc="Zero-shot inference", unit="batch"):
                images = images.to(self.device, non_blocking=True)
                logits = self._forward(model, images)
                probs = torch.sigmoid(logits).cpu().numpy()
                probabilities.append(probs)

        prob_array = np.concatenate(probabilities, axis=0)
        binary = (prob_array >= self.thresholds.reshape(1, -1)).astype(np.int32)
        return InferenceResults(
            probabilities=prob_array,
            binary=binary,
            paths=self.dataset.relative_paths,
            class_names=self.class_names,
        )

    def _build_model(self):
        args = SimpleNamespace(
            model_name=self.args.model_name,
            projector_features=self.args.projector_features,
            use_mlp=self.args.use_mlp,
            pretrained_weights=self.args.weights,
        )
        model = build_omni_model_from_checkpoint(
            args,
            num_classes_list=self.args.num_classes_list,
            key=self.args.checkpoint_key,
        )
        return model

    def _forward(self, model, images: torch.Tensor) -> torch.Tensor:
        # 优先使用head_n参数避免多余计算。
        outputs = model(images, head_n=self.target_head_index)
        if isinstance(outputs, (tuple, list)):
            # ArkSwinTransformer在head_n不为空时返回(features, logits)。
            if isinstance(outputs, tuple):
                return outputs[1]
            return outputs[self.target_head_index]
        return outputs


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def export_results(results: InferenceResults, output_dir: Path, prob_name: str, binary_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prob_header = ["Path", *results.class_names]
    prob_rows = (
        [path, *map(lambda x: f"{x:.6f}", probs)]
        for path, probs in zip(results.paths, results.probabilities)
    )
    write_csv(output_dir / prob_name, prob_header, prob_rows)

    binary_header = prob_header
    binary_rows = (
        [path, *binary.astype(int).tolist()]
        for path, binary in zip(results.paths, results.binary)
    )
    write_csv(output_dir / binary_name, binary_header, binary_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ark+ zero-shot 推理脚本：保留多任务头并输出CheXpert 14类预测",
    )
    parser.add_argument("data_root", help="影像数据根目录")
    parser.add_argument("csv_path", help="test.csv路径，仅读取第一列图像相对路径")
    parser.add_argument("weights", help="Ark+预训练或微调checkpoint路径")
    parser.add_argument(
        "--output-dir",
        default="zeroshot_outputs",
        help="结果保存目录，默认zeroshot_outputs",
    )
    parser.add_argument(
        "--model-name",
        default="swin_large_768",
        help="模型名称，对应预训练阶段使用的结构，如swin_large_768",
    )
    parser.add_argument(
        "--num-classes-list",
        default=None,
        help="多任务头的类别数列表，逗号分隔。默认与Ark+预训练一致(14,14,14,3,6,1)",
    )
    parser.add_argument(
        "--target-head",
        default="chexpert",
        help="需要导出的head名称（默认chexpert）。若自定义num-classes-list，则可用head索引(head_0等)",
    )
    parser.add_argument(
        "--class-names",
        nargs="*",
        default=None,
        help="可选：覆盖目标head的类别名称列表",
    )
    parser.add_argument(
        "--checkpoint-key",
        default="teacher",
        help="checkpoint中字典key，Ark+预训练默认teacher，如为state_dict请自行指定",
    )
    parser.add_argument(
        "--projector-features",
        type=int,
        default=1376,
        help="与预训练配置保持一致的projector特征维度，默认1376",
    )
    parser.add_argument(
        "--use-mlp",
        action="store_true",
        help="若预训练使用MLP projector，请开启该选项",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=768,
        help="图像缩放尺寸，需与预训练模型一致，默认768",
    )
    parser.add_argument(
        "--normalization",
        default="imagenet",
        help="图像归一化方案，默认imagenet",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="推理使用的设备",
    )
    parser.add_argument(
        "--thresholds-file",
        default=None,
        help="可选：自定义阈值文件(csv)，用于替换默认阈值",
    )
    parser.add_argument(
        "--default-threshold",
        type=float,
        default=0.5,
        help="当无专用阈值时使用的默认阈值，默认0.5",
    )
    parser.add_argument(
        "--prob-filename",
        default="out_inter.csv",
        help="概率输出文件名，默认out_inter.csv",
    )
    parser.add_argument(
        "--binary-filename",
        default="out_final.csv",
        help="阈值化结果文件名，默认out_final.csv",
    )

    args = parser.parse_args()
    args.num_classes_list = parse_num_classes_list(args.num_classes_list)
    return args


def main() -> None:
    args = parse_args()

    runner = ArkZeroShotRunner(args)
    results = runner.run()
    export_results(
        results,
        output_dir=Path(args.output_dir),
        prob_name=args.prob_filename,
        binary_name=args.binary_filename,
    )
    print(
        f"推理完成，共处理{len(results.paths)}张影像。概率结果保存在"
        f" {Path(args.output_dir) / args.prob_filename}，阈值化结果保存在"
        f" {Path(args.output_dir) / args.binary_filename}。"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
