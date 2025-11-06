"""Inference utilities for Ark+ classification models.

This module defines an `ArkPlusInference` helper that can be reused both as a
Python API and as a CLI script (``python Finetuning/inference.py``).
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataloader import build_transform_classification
from models import build_classification_model


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


class CheXpertInferenceDataset(Dataset):
    """Dataset that reads only image paths from a CSV file for inference."""

    def __init__(
        self,
        root_dir: str,
        csv_path: str,
        transform,
    ) -> None:
        self.root_dir = root_dir
        self.csv_path = csv_path
        self.transform = transform
        self.samples: List[tuple[str, str]] = []

        with open(csv_path, "r", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue
                rel_path = row[0].strip()
                if not rel_path:
                    continue
                full_path = os.path.join(root_dir, rel_path)
                if not os.path.isfile(full_path):
                    # Treat the very first non-empty row as a header if the path
                    # does not exist on disk. Otherwise raise an informative
                    # error so the user can fix the CSV.
                    if not self.samples:
                        continue
                    raise FileNotFoundError(
                        f"无法在磁盘上找到图像文件: {full_path}. 请确认test.csv中的路径与数据集根目录是否匹配。"
                    )
                self.samples.append((rel_path, full_path))

        if not self.samples:
            raise ValueError(
                "在提供的CSV中未找到有效的图像路径，请检查test.csv第一列是否包含相对路径。"
            )

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
        return [rel_path for rel_path, _ in self.samples]


@dataclass
class InferenceOutputs:
    probabilities: np.ndarray
    binary: np.ndarray
    paths: List[str]


class ArkPlusInference:
    """封装Ark+模型推理流程的帮助类。"""

    def __init__(
        self,
        data_root: str,
        csv_path: str,
        weights: str,
        output_dir: str,
        model_name: str = "swin_large",
        num_classes: Optional[int] = None,
        class_names: Optional[Sequence[str]] = None,
        init: str = "Ark6",
        checkpoint_key: Optional[str] = "state_dict",
        scale_up: bool = False,
        keep_head: bool = True,
        normalization: str = "imagenet",
        input_size: int = 224,
        resize: int = 256,
        batch_size: int = 8,
        num_workers: int = 4,
        device: Optional[str] = None,
        test_augment: bool = False,
        thresholds: Optional[Sequence[float]] = None,
        default_threshold: float = 0.5,
        data_set: str = "CheXpert",
    ) -> None:
        self.data_root = data_root
        self.csv_path = csv_path
        self.weights = weights
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        if class_names is None:
            class_names = CHEXPERT_CLASS_NAMES
        self.class_names = list(class_names)

        if num_classes is None:
            num_classes = len(self.class_names)
        self.num_classes = num_classes

        if len(self.class_names) != self.num_classes:
            raise ValueError(
                "类别数量与class_names长度不一致，请检查参数。"
            )

        if thresholds is None:
            self.thresholds = np.full(self.num_classes, default_threshold, dtype=np.float32)
        else:
            if len(thresholds) != self.num_classes:
                raise ValueError("阈值数量与类别数不一致。")
            self.thresholds = np.asarray(thresholds, dtype=np.float32)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.args = SimpleNamespace(
            model_name=model_name,
            num_class=self.num_classes,
            init=init,
            pretrained_weights=weights,
            keep_head=keep_head,
            key=checkpoint_key,
            scale_up=scale_up,
            normalization=normalization,
            input_size=input_size,
            img_size=resize,
            data_set=data_set,
            test_augment=test_augment,
        )

        self.batch_size = batch_size
        self.num_workers = num_workers

    def _build_dataloader(self) -> tuple[DataLoader, List[str]]:
        transform = build_transform_classification(
            normalize=self.args.normalization,
            crop_size=self.args.input_size,
            resize=self.args.img_size,
            mode="test",
            test_augment=self.args.test_augment,
        )
        dataset = CheXpertInferenceDataset(
            root_dir=self.data_root,
            csv_path=self.csv_path,
            transform=transform,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        return loader, dataset.relative_paths

    def _build_model(self) -> torch.nn.Module:
        model = build_classification_model(self.args)
        model.to(self.device)
        model.eval()
        return model

    def _predict(self, model: torch.nn.Module, loader: DataLoader) -> np.ndarray:
        all_probs: List[torch.Tensor] = []
        with torch.no_grad():
            for images, _ in tqdm(loader, desc="Running inference"):
                if images.dim() == 5:
                    bs, crops, c, h, w = images.shape
                    images = images.view(-1, c, h, w)
                    is_tencrop = True
                elif images.dim() == 4:
                    is_tencrop = False
                else:
                    raise ValueError("不支持的输入张量维度，请检查增广配置。")

                images = images.to(self.device, non_blocking=True)
                outputs = model(images)

                if is_tencrop:
                    outputs = outputs.view(bs, crops, -1).mean(dim=1)

                if self.args.data_set in ["RSNAPneumonia", "COVIDx"]:
                    probs = torch.softmax(outputs, dim=1)
                else:
                    probs = torch.sigmoid(outputs)

                all_probs.append(probs.cpu())
        return torch.cat(all_probs, dim=0).numpy()

    def run(self) -> InferenceOutputs:
        loader, paths = self._build_dataloader()
        model = self._build_model()
        probabilities = self._predict(model, loader)
        binary = (probabilities >= self.thresholds.reshape(1, -1)).astype(np.int32)
        return InferenceOutputs(probabilities=probabilities, binary=binary, paths=paths)

    def save_outputs(
        self,
        outputs: InferenceOutputs,
        inter_path: Optional[str] = None,
        final_path: Optional[str] = None,
    ) -> tuple[str, str]:
        if inter_path is None:
            inter_path = os.path.join(self.output_dir, "out_inter.csv")
        if final_path is None:
            final_path = os.path.join(self.output_dir, "out_final.csv")

        header = ["Path"] + list(self.class_names)

        with open(inter_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for rel_path, row in zip(outputs.paths, outputs.probabilities):
                writer.writerow([rel_path] + [f"{float(x):.6f}" for x in row])

        with open(final_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for rel_path, row in zip(outputs.paths, outputs.binary):
                writer.writerow([rel_path] + [int(x) for x in row])

        return inter_path, final_path


def _parse_class_names(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    if os.path.isfile(value):
        with open(value, "r", encoding="utf-8") as handle:
            names = [line.strip() for line in handle if line.strip()]
            return names
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_thresholds(path: Optional[str], num_classes: int) -> Optional[List[float]]:
    if path is None:
        return None
    values: List[float] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            cleaned = line.strip()
            if not cleaned:
                continue
            cleaned = cleaned.replace(",", " ")
            for token in cleaned.split():
                values.append(float(token))
    if len(values) != num_classes:
        raise ValueError(
            f"阈值文件中包含{len(values)}个值，但期望{num_classes}个。"
        )
    return values


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Ark+ 模型推理脚本")
    parser.add_argument("data_root", help="数据集根目录")
    parser.add_argument("csv_path", help="test.csv路径，仅读取第一列图像相对路径")
    parser.add_argument("weights", help="预训练或微调模型权重路径")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="保存out_inter.csv与out_final.csv的目录",
    )
    parser.add_argument("--model-name", default="swin_large", help="模型名称，例如swin_large")
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="类别数量，不指定时与class-names长度一致",
    )
    parser.add_argument(
        "--class-names",
        default=None,
        help="类别名称列表，可为逗号分隔字符串或文本文件路径",
    )
    parser.add_argument("--init", default="Ark6", help="预训练初始化名称，用于加载权重")
    parser.add_argument(
        "--checkpoint-key",
        default="state_dict",
        help="在权重文件中用于提取state_dict的键，微调模型可自行指定",
    )
    parser.add_argument(
        "--no-keep-head",
        dest="keep_head",
        action="store_false",
        help="不保留checkpoint中的分类头",
    )
    parser.set_defaults(keep_head=True)
    parser.add_argument(
        "--scale-up",
        action="store_true",
        help="加载权重时移除attn_mask（与训练脚本保持一致）",
    )
    parser.add_argument(
        "--normalization",
        default="imagenet",
        help="归一化方式，默认imagenet",
    )
    parser.add_argument("--input-size", type=int, default=224, help="中心裁剪尺寸")
    parser.add_argument("--resize", type=int, default=256, help="缩放尺寸")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device",
        default=None,
        help="推理设备，例如cuda或cpu，默认自动检测",
    )
    parser.add_argument(
        "--test-augment",
        action="store_true",
        help="启用TenCrop测试增广",
    )
    parser.add_argument(
        "--default-threshold",
        type=float,
        default=0.5,
        help="未提供阈值文件时使用的默认阈值",
    )
    parser.add_argument(
        "--thresholds-file",
        default=None,
        help="包含逐类阈值的文本文件（可选）",
    )
    parser.add_argument(
        "--data-set",
        default="CheXpert",
        help="用于兼容softmax/sigmoid决策的dataset名称",
    )
    parser.add_argument(
        "--inter-path",
        default=None,
        help="自定义out_inter.csv保存路径",
    )
    parser.add_argument(
        "--final-path",
        default=None,
        help="自定义out_final.csv保存路径",
    )

    args = parser.parse_args(argv)

    class_names = _parse_class_names(args.class_names)
    thresholds = None

    if args.thresholds_file is not None:
        num_classes = args.num_classes or (
            len(class_names) if class_names is not None else len(CHEXPERT_CLASS_NAMES)
        )
        thresholds = _load_thresholds(args.thresholds_file, num_classes)

    runner = ArkPlusInference(
        data_root=args.data_root,
        csv_path=args.csv_path,
        weights=args.weights,
        output_dir=args.output_dir,
        model_name=args.model_name,
        num_classes=args.num_classes,
        class_names=class_names,
        init=args.init,
        checkpoint_key=args.checkpoint_key,
        scale_up=args.scale_up,
        keep_head=args.keep_head,
        normalization=args.normalization,
        input_size=args.input_size,
        resize=args.resize,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        test_augment=args.test_augment,
        thresholds=thresholds,
        default_threshold=args.default_threshold,
        data_set=args.data_set,
    )

    outputs = runner.run()
    inter_path, final_path = runner.save_outputs(outputs, args.inter_path, args.final_path)

    print(f"概率结果已保存至: {inter_path}")
    print(f"阈值二值化结果已保存至: {final_path}")


if __name__ == "__main__":
    main()
