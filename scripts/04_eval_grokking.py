from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from fizzbuzz.config import ExperimentConfig, load_config
from fizzbuzz.data import build_train_loader
from fizzbuzz.evaluator import evaluate_extrapolation
from fizzbuzz.model import build_model, count_parameters
from fizzbuzz.trainer import resolve_device
from fizzbuzz.utils import JsonDict, ensure_dir, save_json, seed_everything


DEFAULT_CONFIG_PATHS: tuple[Path, ...] = (
    Path("configs/small.yaml"),
    Path("configs/medium.yaml"),
    Path("configs/large.yaml"),
)

DEFAULT_EPOCH_CHECKPOINTS: tuple[int, ...] = (
    10,
    50,
    100,
    500,
    1000,
    5000,
    10000,
)

RANGE_ORDER: tuple[str, ...] = (
    "train",
    "test_6digit",
    "test_7digit",
    "test_8digit",
)

RANGE_DISPLAY_NAMES: Mapping[str, str] = {
    "train": "train",
    "test_6digit": "test-6digit",
    "test_7digit": "test-7digit",
    "test_8digit": "test-8digit",
}

RANGE_COLORS: Mapping[str, str] = {
    "train": "tab:blue",
    "test_6digit": "tab:orange",
    "test_7digit": "tab:green",
    "test_8digit": "tab:red",
}

NUM_CLASSES = 4


@dataclass(frozen=True)
class MetricSnapshot:
    """1つの評価範囲に対する最小分類指標です。

    Parameters
    ----------
    accuracy : float
        Accuracyです。
    macro_f1 : float
        Macro F1です。
    num_samples : int
        評価サンプル数です。
    elapsed_sec : float
        評価に要した秒数です。
    """

    accuracy: float
    macro_f1: float
    num_samples: int
    elapsed_sec: float


@dataclass(frozen=True)
class EpochEvalRecord:
    """1つのepoch checkpointに対する評価結果です。

    Parameters
    ----------
    epoch : int
        評価対象の学習epoch数です。
    weight_path : str
        評価した重みファイルのプロジェクト相対パスです。
    metrics : dict[str, MetricSnapshot]
        評価範囲名から指標への対応です。
    elapsed_sec : float
        train/testを含むcheckpoint全体の評価時間です。
    """

    epoch: int
    weight_path: str
    metrics: dict[str, MetricSnapshot]
    elapsed_sec: float


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析します。

    Returns
    -------
    argparse.Namespace
        解析済み引数。
    """
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate grokking checkpoints for the FizzBuzz "
            "experiment and plot train/test metric trajectories."
        )
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config",
        type=Path,
        help="Path to a YAML config file.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all default configs: small, medium, and large.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=list(DEFAULT_EPOCH_CHECKPOINTS),
        help=(
            "Epoch checkpoints to evaluate. "
            f"Default: {list(DEFAULT_EPOCH_CHECKPOINTS)}."
        ),
    )
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=Path("runs/grokking"),
        help=(
            "Root directory containing <model_name>/milestones/epoch_XXXXXX.pt "
            "weights. Default: runs/grokking."
        ),
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=None,
        help=(
            "Batch size for train-range evaluation. "
            "If omitted, cfg.training.batch_size is used."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Reuse existing per-checkpoint eval_grokking.json files when present. "
            "Plots are still regenerated from the loaded records."
        ),
    )

    return parser.parse_args()


def evaluate_grokking_from_config_path(
    config_path: Path,
    *,
    epoch_checkpoints: tuple[int, ...],
    sweep_root: Path,
    train_batch_size: int | None,
    skip_existing: bool,
) -> JsonDict:
    """指定configのepoch sweep checkpoint列を評価します。

    Parameters
    ----------
    config_path : pathlib.Path
        評価に使うYAML configのパス。
    epoch_checkpoints : tuple[int, ...]
        評価対象のepoch checkpoint列。
    sweep_root : pathlib.Path
        checkpointを格納したrootディレクトリ。
    train_batch_size : int | None
        train評価用batch size。Noneの場合はconfigのtraining.batch_sizeを使います。
    skip_existing : bool
        Trueの場合、既存のper-checkpoint評価JSONを再利用します。

    Returns
    -------
    dict[str, JsonValue]
        モデル単位の評価結果payload。

    Raises
    ------
    FileNotFoundError
        configファイルまたはcheckpoint重みが存在しない場合。
    ValueError
        epoch checkpoint列または引数が不正な場合。
    """
    resolved_config_path = resolve_project_path(config_path)

    if not resolved_config_path.exists():
        raise FileNotFoundError(f"Config file not found: {resolved_config_path}")

    validated_epochs = validate_epoch_checkpoints(epoch_checkpoints)
    resolved_sweep_root = resolve_project_path(sweep_root)

    cfg = load_config(resolved_config_path)
    return evaluate_one_grokking_sweep(
        cfg,
        config_path=resolved_config_path,
        epoch_checkpoints=validated_epochs,
        sweep_root=resolved_sweep_root,
        train_batch_size=train_batch_size,
        skip_existing=skip_existing,
    )


def evaluate_one_grokking_sweep(
    cfg: ExperimentConfig,
    *,
    config_path: Path,
    epoch_checkpoints: tuple[int, ...],
    sweep_root: Path,
    train_batch_size: int | None,
    skip_existing: bool,
) -> JsonDict:
    """1つのモデルサイズについてgrokking評価を実行します。

    Parameters
    ----------
    cfg : ExperimentConfig
        検証済みの実験設定。
    config_path : pathlib.Path
        元configファイルのパス。
    epoch_checkpoints : tuple[int, ...]
        評価対象のepoch checkpoint列。
    sweep_root : pathlib.Path
        checkpointを格納したrootディレクトリ。
    train_batch_size : int | None
        train評価用batch size。Noneの場合はconfigのtraining.batch_sizeを使います。
    skip_existing : bool
        Trueの場合、既存のper-checkpoint評価JSONを再利用します。

    Returns
    -------
    dict[str, JsonValue]
        モデル単位の評価結果payload。
    """
    sweep_dir = ensure_dir(sweep_root / cfg.name)

    print(f"\n[grokking_eval] name={cfg.name}")
    print(f"[grokking_eval] config={config_path}")
    print(f"[grokking_eval] checkpoints={list(epoch_checkpoints)}")
    print(f"[grokking_eval] sweep_dir={sweep_dir}")

    records: list[EpochEvalRecord] = []

    for epoch in epoch_checkpoints:
        record = evaluate_one_epoch_checkpoint(
            cfg,
            epoch=epoch,
            sweep_dir=sweep_dir,
            train_batch_size=train_batch_size,
            skip_existing=skip_existing,
        )
        records.append(record)

    num_parameters = infer_num_parameters(cfg)
    records_payload = [epoch_record_to_dict(record) for record in records]

    payload: JsonDict = {
        "name": cfg.name,
        "seed": cfg.seed,
        "config_path": to_project_relative_str(config_path),
        "sweep_dir": to_project_relative_str(sweep_dir),
        "epoch_checkpoints": list(epoch_checkpoints),
        "num_parameters": num_parameters,
        "ranges": list(RANGE_ORDER),
        "records": records_payload, # type: ignore
    }

    summary_path = sweep_dir / "grokking_eval_results.json"
    save_json(payload, summary_path)

    plot_path = save_grokking_metric_plot(
        model_name=cfg.name,
        records=records,
        sweep_dir=sweep_dir,
    )
    payload["plot_path"] = to_project_relative_str(plot_path)
    save_json(payload, summary_path)

    print(f"[grokking_eval] saved={summary_path}")
    print(f"[grokking_eval] plot={plot_path}")

    return payload


def evaluate_one_epoch_checkpoint(
    cfg: ExperimentConfig,
    *,
    epoch: int,
    sweep_dir: Path,
    train_batch_size: int | None,
    skip_existing: bool,
) -> EpochEvalRecord:
    """1つのepoch checkpointをtrain/test範囲で評価します。

    Parameters
    ----------
    cfg : ExperimentConfig
        検証済みの実験設定。
    epoch : int
        評価対象のepoch数。
    sweep_dir : pathlib.Path
        epoch sweep出力ディレクトリ。
    train_batch_size : int | None
        train評価用batch size。Noneの場合はconfigのtraining.batch_sizeを使います。
    skip_existing : bool
        Trueの場合、既存のper-checkpoint評価JSONを再利用します。

    Returns
    -------
    EpochEvalRecord
        1 checkpoint分の評価結果。

    Raises
    ------
    FileNotFoundError
        checkpoint重みが存在しない場合。
    """
    weight_path = sweep_dir / "milestones" / f"epoch_{epoch:06d}.pt"
    eval_dir = ensure_dir(sweep_dir / "evals")
    eval_path = eval_dir / f"epoch_{epoch:06d}.json"

    if skip_existing and eval_path.exists():
        print(f"\n[grokking_eval] reuse epoch={epoch} eval={eval_path}")
        return load_epoch_eval_record(eval_path)

    if not weight_path.exists():
        raise FileNotFoundError(
            f"Checkpoint weight not found: {weight_path}. "
            "Run scripts/03_epoch_sweep.py first."
        )

    started_at = time.perf_counter()

    print(f"\n[grokking_eval] start epoch={epoch}")
    print(f"[grokking_eval] weight={weight_path}")

    seed_everything(cfg.seed)

    device = resolve_device(cfg.training.device)
    model = build_model(cfg.model)
    load_model_state(model, weight_path)
    model.to(device)
    model.eval()

    train_metrics = evaluate_train_metrics(
        model,
        cfg=cfg,
        device=device,
        train_batch_size=train_batch_size,
    )

    test_result = evaluate_extrapolation(
        model,
        data_config=cfg.data,
        eval_config=cfg.eval,
        device=device,
    )

    metrics: dict[str, MetricSnapshot] = {"train": train_metrics}

    for range_name, range_result in test_result.results.items():
        metrics[range_name] = MetricSnapshot(
            accuracy=float(range_result.metrics.accuracy),
            macro_f1=float(range_result.metrics.macro_f1),
            num_samples=int(range_result.metrics.num_samples),
            elapsed_sec=float(range_result.elapsed_sec),
        )

    elapsed_sec = time.perf_counter() - started_at

    record = EpochEvalRecord(
        epoch=epoch,
        weight_path=to_project_relative_str(weight_path),
        metrics=metrics,
        elapsed_sec=float(elapsed_sec),
    )

    save_json(epoch_record_to_dict(record), eval_path)

    for range_name in RANGE_ORDER:
        metric = metrics[range_name]
        print(
            "[grokking_eval] "
            f"epoch={epoch} "
            f"{format_range_name(range_name)} "
            f"acc={metric.accuracy:.6f} "
            f"macro_f1={metric.macro_f1:.6f} "
            f"n={metric.num_samples:,} "
            f"sec={metric.elapsed_sec:.2f}"
        )

    print(f"[grokking_eval] checkpoint saved={eval_path}")
    return record


def evaluate_train_metrics(
    model: nn.Module,
    *,
    cfg: ExperimentConfig,
    device: torch.device,
    train_batch_size: int | None,
) -> MetricSnapshot:
    """train範囲に対するAccuracyとMacro F1を算出します。

    Parameters
    ----------
    model : torch.nn.Module
        評価対象モデル。
    cfg : ExperimentConfig
        実験設定。
    device : torch.device
        評価に使うdevice。
    train_batch_size : int | None
        train評価用batch size。Noneの場合はconfigのtraining.batch_sizeを使います。

    Returns
    -------
    MetricSnapshot
        train範囲に対するAccuracy, Macro F1, サンプル数, 評価時間。

    Raises
    ------
    ValueError
        train_batch_sizeが正でない場合。
    RuntimeError
        モデル出力またはbatch形式が不正な場合。
    """
    batch_size = cfg.training.batch_size if train_batch_size is None else train_batch_size

    if batch_size <= 0:
        raise ValueError(f"train_batch_size must be positive: {batch_size}")

    started_at = time.perf_counter()

    generator = torch.Generator()
    generator.manual_seed(cfg.seed)

    train_loader = build_train_loader(
        cfg.data,
        batch_size=batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=False,
        generator=generator,
    )

    confusion_matrix = evaluate_loader_confusion_matrix(
        model,
        train_loader=train_loader,
        device=device,
        num_classes=NUM_CLASSES,
    )
    accuracy, macro_f1, num_samples = summarize_confusion_matrix(confusion_matrix)

    return MetricSnapshot(
        accuracy=accuracy,
        macro_f1=macro_f1,
        num_samples=num_samples,
        elapsed_sec=float(time.perf_counter() - started_at),
    )


def evaluate_loader_confusion_matrix(
    model: nn.Module,
    *,
    train_loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> np.ndarray:
    """DataLoader全体のconfusion matrixを作成します。

    Parameters
    ----------
    model : torch.nn.Module
        評価対象モデル。
    train_loader : torch.utils.data.DataLoader
        評価対象DataLoader。
    device : torch.device
        評価に使うdevice。
    num_classes : int
        クラス数。

    Returns
    -------
    numpy.ndarray
        shape ``(num_classes, num_classes)`` のconfusion matrix。
        行が真ラベル、列が予測ラベルです。

    Raises
    ------
    ValueError
        num_classesが正でない場合。
    RuntimeError
        batchまたはモデル出力の形式が不正な場合。
    """
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive: {num_classes}")

    confusion_matrix = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.int64,
        device="cpu",
    )

    model.eval()

    with torch.inference_mode():
        for batch in train_loader:
            digits, lengths, labels = unpack_digit_batch(batch)
            digits = digits.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            lengths_on_device = (
                lengths.to(device, non_blocking=True) if lengths is not None else None
            )

            logits = forward_model(model, digits=digits, lengths=lengths_on_device)
            predictions = torch.argmax(logits, dim=1)

            confusion_matrix += batch_confusion_matrix(
                labels=labels,
                predictions=predictions,
                num_classes=num_classes,
            )

    return confusion_matrix.numpy()


def unpack_digit_batch(batch: object) -> tuple[Tensor, Tensor | None, Tensor]:
    """FizzBuzz用batchからdigits, lengths, labelsを取り出します。

    Parameters
    ----------
    batch : object
        ``DigitBatch`` またはtuple/list形式のbatch。

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]
        digits, lengths, labels。

    Raises
    ------
    RuntimeError
        batch形式が未対応の場合。
    """
    if hasattr(batch, "digits") and hasattr(batch, "labels"):
        digits = getattr(batch, "digits")
        lengths = getattr(batch, "lengths", None)
        labels = getattr(batch, "labels")
        return validate_unpacked_batch(digits, lengths, labels)

    if isinstance(batch, Sequence):
        if len(batch) == 3:
            digits, lengths, labels = batch
            return validate_unpacked_batch(digits, lengths, labels)
        if len(batch) == 2:
            digits, labels = batch
            return validate_unpacked_batch(digits, None, labels)

    raise RuntimeError(
        "Unsupported batch format. Expected DigitBatch, "
        "(digits, lengths, labels), or (digits, labels)."
    )


def validate_unpacked_batch(
    digits: object,
    lengths: object | None,
    labels: object,
) -> tuple[Tensor, Tensor | None, Tensor]:
    """unpack後のbatch要素がTensorであることを検証します。

    Parameters
    ----------
    digits : object
        digit ID列。
    lengths : object | None
        有効系列長。
    labels : object
        正解ラベル。

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]
        検証済みのdigits, lengths, labels。

    Raises
    ------
    RuntimeError
        Tensorでない要素が含まれる場合。
    """
    if not isinstance(digits, Tensor):
        raise RuntimeError(f"digits must be a torch.Tensor, got {type(digits).__name__}.")
    if lengths is not None and not isinstance(lengths, Tensor):
        raise RuntimeError(
            f"lengths must be a torch.Tensor or None, got {type(lengths).__name__}."
        )
    if not isinstance(labels, Tensor):
        raise RuntimeError(f"labels must be a torch.Tensor, got {type(labels).__name__}.")

    return digits, lengths, labels


def forward_model(
    model: nn.Module,
    *,
    digits: Tensor,
    lengths: Tensor | None,
) -> Tensor:
    """モデルを呼び出してlogitsを取得します。

    Parameters
    ----------
    model : torch.nn.Module
        評価対象モデル。
    digits : torch.Tensor
        digit ID列。
    lengths : torch.Tensor | None
        有効系列長。Noneの場合はdigitsのみでforwardします。

    Returns
    -------
    torch.Tensor
        shape ``(batch_size, num_classes)`` のlogits。

    Raises
    ------
    RuntimeError
        モデル出力がTensorでない場合、またはshapeが不正な場合。
    TypeError
        モデル呼び出しに失敗した場合。
    """
    if lengths is None:
        output = model(digits)
    else:
        try:
            output = model(digits, lengths)
        except TypeError as error_with_lengths:
            try:
                output = model(digits)
            except TypeError:
                raise error_with_lengths

    if isinstance(output, tuple | list):
        if len(output) == 0:
            raise RuntimeError("Model output tuple/list must not be empty.")
        output = output[0]

    if not isinstance(output, Tensor):
        raise RuntimeError(
            f"Model output must be a torch.Tensor, got {type(output).__name__}."
        )
    if output.ndim != 2:
        raise RuntimeError(f"Model logits must be 2D, got shape={tuple(output.shape)}.")

    return output


def batch_confusion_matrix(
    *,
    labels: Tensor,
    predictions: Tensor,
    num_classes: int,
) -> Tensor:
    """1 batch分のconfusion matrixを作成します。

    Parameters
    ----------
    labels : torch.Tensor
        真ラベル。shape ``(batch_size,)``。
    predictions : torch.Tensor
        予測ラベル。shape ``(batch_size,)``。
    num_classes : int
        クラス数。

    Returns
    -------
    torch.Tensor
        CPU上のconfusion matrix。

    Raises
    ------
    ValueError
        labels/predictionsのshapeまたは値域が不正な場合。
    """
    labels_cpu = labels.detach().to("cpu", dtype=torch.long).reshape(-1)
    predictions_cpu = predictions.detach().to("cpu", dtype=torch.long).reshape(-1)

    if labels_cpu.shape != predictions_cpu.shape:
        raise ValueError(
            "labels and predictions must have the same shape: "
            f"labels={tuple(labels_cpu.shape)}, "
            f"predictions={tuple(predictions_cpu.shape)}."
        )
    if labels_cpu.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.int64)
    if int(labels_cpu.min()) < 0 or int(labels_cpu.max()) >= num_classes:
        raise ValueError("labels contain out-of-range class ids.")
    if int(predictions_cpu.min()) < 0 or int(predictions_cpu.max()) >= num_classes:
        raise ValueError("predictions contain out-of-range class ids.")

    indices = labels_cpu * num_classes + predictions_cpu
    counts = torch.bincount(indices, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def summarize_confusion_matrix(confusion_matrix: np.ndarray) -> tuple[float, float, int]:
    """confusion matrixからAccuracyとMacro F1を計算します。

    Parameters
    ----------
    confusion_matrix : numpy.ndarray
        行が真ラベル、列が予測ラベルのconfusion matrix。

    Returns
    -------
    tuple[float, float, int]
        accuracy, macro_f1, num_samples。

    Raises
    ------
    ValueError
        confusion matrixが不正な場合。
    """
    if confusion_matrix.ndim != 2:
        raise ValueError(
            f"confusion_matrix must be 2D, got ndim={confusion_matrix.ndim}."
        )
    if confusion_matrix.shape[0] != confusion_matrix.shape[1]:
        raise ValueError(
            "confusion_matrix must be square, "
            f"got shape={confusion_matrix.shape}."
        )
    if np.any(confusion_matrix < 0):
        raise ValueError("confusion_matrix must not contain negative values.")

    matrix = confusion_matrix.astype(np.float64, copy=False)
    num_samples = int(matrix.sum())

    if num_samples == 0:
        raise ValueError("confusion_matrix must contain at least one sample.")

    true_positive = np.diag(matrix)
    predicted_positive = matrix.sum(axis=0)
    actual_positive = matrix.sum(axis=1)

    precision = np.divide(
        true_positive,
        predicted_positive,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=predicted_positive > 0,
    )
    recall = np.divide(
        true_positive,
        actual_positive,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=actual_positive > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision, dtype=np.float64),
        where=(precision + recall) > 0,
    )

    accuracy = float(true_positive.sum() / num_samples)
    macro_f1 = float(np.mean(f1))

    return accuracy, macro_f1, num_samples


def load_model_state(model: nn.Module, weight_path: Path) -> None:
    """モデル重みを読み込みます。

    Parameters
    ----------
    model : torch.nn.Module
        重みを読み込むモデル。
    weight_path : pathlib.Path
        ``state_dict`` が保存された ``.pt`` ファイル。

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        重みファイルが存在しない場合。
    TypeError
        読み込んだオブジェクトがMappingでない場合。
    RuntimeError
        ``state_dict`` の読み込みに失敗した場合。
    """
    if not weight_path.exists():
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

    state = torch.load(
        weight_path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(state, Mapping):
        raise TypeError(
            "weight file must contain a model state_dict mapping, "
            f"got {type(state).__name__}."
        )

    model.load_state_dict(state)


def save_grokking_metric_plot(
    *,
    model_name: str,
    records: Sequence[EpochEvalRecord],
    sweep_dir: Path,
) -> Path:
    """1モデルサイズ分のgrokking推移図を保存します。

    Parameters
    ----------
    model_name : str
        モデル名。
    records : Sequence[EpochEvalRecord]
        epoch checkpointごとの評価結果。
    sweep_dir : pathlib.Path
        grokking checkpoint root for one model size.
        root直下の ``history.json`` からtrain accuracyのper-epoch系列を読みます。

    Returns
    -------
    pathlib.Path
        保存先画像パス。

    Raises
    ------
    ValueError
        recordsが空、または必要な評価範囲を含まない場合。
    """
    if len(records) == 0:
        raise ValueError("records must not be empty.")

    sorted_records = sorted(records, key=lambda record: record.epoch)
    epochs = np.asarray([record.epoch for record in sorted_records], dtype=np.float64)
    train_accuracy_curve = load_train_accuracy_curve(
        sweep_dir=sweep_dir,
        records=sorted_records,
    )

    if train_accuracy_curve is None:
        print(
            "[grokking_eval] train history was not found or could not be parsed; "
            "train Acc is plotted only at checkpoint epochs."
        )

    image_dir = ensure_dir(PROJECT_ROOT / "runs" / "images" / "grokking")
    output_path = image_dir / f"grokking_{slugify(model_name)}.png"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    metric_specs = (
        ("accuracy", "Acc"),
        ("macro_f1", "Macro F1"),
    )

    for ax, (metric_name, y_label) in zip(axes, metric_specs, strict=True):
        for range_name in RANGE_ORDER:
            if (
                metric_name == "accuracy"
                and range_name == "train"
                and train_accuracy_curve is not None
            ):
                train_epochs, train_acc = train_accuracy_curve
                ax.plot(
                    train_epochs,
                    train_acc,
                    marker=None,
                    linewidth=1.8,
                    label=RANGE_DISPLAY_NAMES[range_name],
                    color=RANGE_COLORS[range_name],
                )
                continue

            values = np.asarray(
                [get_metric_value(record, range_name, metric_name) for record in sorted_records],
                dtype=np.float64,
            )
            ax.plot(
                epochs,
                values,
                marker="o",
                linewidth=1.8,
                markersize=4.5,
                label=RANGE_DISPLAY_NAMES[range_name],
                color=RANGE_COLORS[range_name],
            )

        # A pure log scale cannot include x=0.  A symlog scale keeps the requested
        # displayed range [0, 10^4] while remaining logarithmic outside the small
        # linear region near zero.
        ax.set_xscale("symlog", linthresh=10, linscale=1.0, base=10)
        ax.set_xlim(0, 10_000)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks([0, 10, 100, 1000, 10000])
        ax.set_xticklabels(["0", "10", "$10^2$", "$10^3$", "$10^4$"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel(y_label)
        ax.set_title(y_label)
        ax.grid(True, which="both", alpha=0.35)
        ax.legend(loc="lower right", fontsize=9)

    fig.suptitle(f"Grokking metric trajectories: {model_name}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path



def load_train_accuracy_curve(
    *,
    sweep_dir: Path,
    records: Sequence[EpochEvalRecord],
) -> tuple[np.ndarray, np.ndarray] | None:
    """root直下の ``history.json`` からtrain accuracy系列を読みます。

    Parameters
    ----------
    sweep_dir : pathlib.Path
        grokking checkpoint root for one model size.
    records : Sequence[EpochEvalRecord]
        epoch checkpointごとの評価結果。空の場合はNoneを返します。

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray] | None
        ``(epochs, accuracies)``。履歴が見つからない、またはparseできない場合はNone。

    Notes
    -----
    03側は各checkpointを独立再学習せず、最大epochまで1回だけ学習します。
    そのため、train Acc曲線は ``runs/grokking/<model_name>/history.json``
    から読みます。
    """
    if len(records) == 0:
        return None

    history_path = sweep_dir / "history.json"
    if not history_path.exists():
        return None

    try:
        points = parse_train_accuracy_history(history_path)
    except (OSError, TypeError, ValueError, KeyError) as error:
        print(f"[grokking_eval] failed to parse history={history_path}: {error}")
        return None

    if len(points) == 0:
        return None

    sorted_points = sorted(points, key=lambda item: item[0])
    epochs = np.asarray([epoch for epoch, _ in sorted_points], dtype=np.float64)
    accuracies = np.asarray([accuracy for _, accuracy in sorted_points], dtype=np.float64)
    return epochs, accuracies


def parse_train_accuracy_history(history_path: Path) -> list[tuple[int, float]]:
    """``history.json`` から ``(epoch, accuracy)`` のリストを取り出します。

    Parameters
    ----------
    history_path : pathlib.Path
        Trainerが保存した ``history.json`` のパス。

    Returns
    -------
    list[tuple[int, float]]
        epoch番号とtrain accuracyの系列。

    Raises
    ------
    TypeError
        JSON構造が未対応の場合。
    ValueError
        accuracyが数値でない場合。
    """
    import json

    with history_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, Mapping):
        raw_records = payload.get("records", payload.get("history"))
    elif isinstance(payload, list):
        raw_records = payload
    else:
        raise TypeError(
            "history JSON must contain a mapping or list, "
            f"got {type(payload).__name__}."
        )

    if not isinstance(raw_records, list):
        raise TypeError("history JSON must contain a list under 'records' or 'history'.")

    points: list[tuple[int, float]] = []

    for index, raw_record in enumerate(raw_records, start=1):
        if not isinstance(raw_record, Mapping):
            raise TypeError(
                "each history record must be a mapping, "
                f"got {type(raw_record).__name__}."
            )

        epoch = parse_history_epoch(raw_record, fallback_epoch=index)
        accuracy = parse_history_accuracy(raw_record)
        points.append((epoch, accuracy))

    return points


def parse_history_epoch(
    raw_record: Mapping[object, object],
    *,
    fallback_epoch: int,
) -> int:
    """history recordからepoch番号を取り出します。

    Parameters
    ----------
    raw_record : Mapping[object, object]
        1 epoch分の履歴record。
    fallback_epoch : int
        ``epoch`` キーが存在しない場合に使う1始まりのepoch番号。

    Returns
    -------
    int
        epoch番号。

    Raises
    ------
    ValueError
        epoch番号が正でない場合。
    """
    raw_epoch = raw_record.get("epoch", fallback_epoch)
    epoch = int(raw_epoch) # type: ignore

    if epoch <= 0:
        raise ValueError(f"history epoch must be positive: {epoch}")

    return epoch


def parse_history_accuracy(raw_record: Mapping[object, object]) -> float:
    """history recordからtrain accuracyを取り出します。

    Parameters
    ----------
    raw_record : Mapping[object, object]
        1 epoch分の履歴record。

    Returns
    -------
    float
        train accuracy。

    Raises
    ------
    KeyError
        accuracyに対応するキーが存在しない場合。
    ValueError
        accuracyが有限値でない場合。
    """
    accuracy_keys = ("accuracy", "acc", "train_accuracy", "train_acc")

    for key in accuracy_keys:
        if key not in raw_record:
            continue

        accuracy = float(raw_record[key]) # type: ignore
        if not np.isfinite(accuracy):
            raise ValueError(f"history accuracy must be finite: {accuracy}")
        return accuracy

    raise KeyError(f"history record does not contain accuracy keys: {accuracy_keys}")

def get_metric_value(
    record: EpochEvalRecord,
    range_name: str,
    metric_name: str,
) -> float:
    """recordから指定範囲・指定指標の値を取得します。

    Parameters
    ----------
    record : EpochEvalRecord
        1 checkpoint分の評価結果。
    range_name : str
        評価範囲名。
    metric_name : str
        ``accuracy`` または ``macro_f1``。

    Returns
    -------
    float
        指標値。

    Raises
    ------
    KeyError
        range_nameが存在しない場合。
    ValueError
        metric_nameが未対応の場合。
    """
    if range_name not in record.metrics:
        raise KeyError(
            f"Missing range={range_name!r} in epoch={record.epoch}. "
            f"Available ranges={sorted(record.metrics)}."
        )

    metric = record.metrics[range_name]

    if metric_name == "accuracy":
        return metric.accuracy
    if metric_name == "macro_f1":
        return metric.macro_f1

    raise ValueError(f"Unsupported metric_name: {metric_name}")


def infer_num_parameters(cfg: ExperimentConfig) -> int:
    """configからモデルを構築してパラメータ数を返します。

    Parameters
    ----------
    cfg : ExperimentConfig
        実験設定。

    Returns
    -------
    int
        学習対象パラメータ数。
    """
    model = build_model(cfg.model)
    return count_parameters(model)


def epoch_record_to_dict(record: EpochEvalRecord) -> JsonDict:
    """EpochEvalRecordをJSON保存可能なdictに変換します。

    Parameters
    ----------
    record : EpochEvalRecord
        変換対象。

    Returns
    -------
    dict[str, JsonValue]
        JSON保存可能なdict。
    """
    return {
        "epoch": record.epoch,
        "weight_path": record.weight_path,
        "metrics": {
            range_name: asdict(metric)
            for range_name, metric in record.metrics.items()
        },
        "elapsed_sec": record.elapsed_sec,
    }


def load_epoch_eval_record(eval_path: Path) -> EpochEvalRecord:
    """保存済みのper-checkpoint評価JSONを読み込みます。

    Parameters
    ----------
    eval_path : pathlib.Path
        ``eval_grokking.json`` のパス。

    Returns
    -------
    EpochEvalRecord
        復元した評価結果。

    Raises
    ------
    FileNotFoundError
        eval_pathが存在しない場合。
    TypeError
        JSONの構造が不正な場合。
    """
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval JSON not found: {eval_path}")

    import json

    with eval_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, Mapping):
        raise TypeError(f"Eval JSON must contain a mapping: {eval_path}")

    raw_metrics = payload.get("metrics")
    if not isinstance(raw_metrics, Mapping):
        raise TypeError(f"Eval JSON must contain a metrics mapping: {eval_path}")

    metrics: dict[str, MetricSnapshot] = {}
    for range_name, metric_payload in raw_metrics.items():
        if not isinstance(range_name, str):
            raise TypeError("range names in metrics must be strings.")
        if not isinstance(metric_payload, Mapping):
            raise TypeError(f"Metric payload must be a mapping: {range_name}")

        metrics[range_name] = MetricSnapshot(
            accuracy=float(metric_payload["accuracy"]),
            macro_f1=float(metric_payload["macro_f1"]),
            num_samples=int(metric_payload["num_samples"]),
            elapsed_sec=float(metric_payload["elapsed_sec"]),
        )

    return EpochEvalRecord(
        epoch=int(payload["epoch"]),
        weight_path=str(payload["weight_path"]),
        metrics=metrics,
        elapsed_sec=float(payload["elapsed_sec"]),
    )


def save_merged_summary(summaries: Mapping[str, JsonDict]) -> Path:
    """全モデルのgrokking評価summary JSONを保存します。

    Parameters
    ----------
    summaries : Mapping[str, JsonDict]
        モデル名からモデル単位payloadへの対応。

    Returns
    -------
    pathlib.Path
        保存先パス。
    """
    summary_dir = ensure_dir(PROJECT_ROOT / "runs" / "grokking")
    summary_path = summary_dir / "grokking_eval_summary.json"
    save_json({"by_model": summaries}, summary_path)  # type: ignore[arg-type]
    return summary_path


def validate_epoch_checkpoints(epoch_checkpoints: tuple[int, ...]) -> tuple[int, ...]:
    """epoch checkpoint列を検証します。

    Parameters
    ----------
    epoch_checkpoints : tuple[int, ...]
        検証対象のepoch checkpoint列。

    Returns
    -------
    tuple[int, ...]
        昇順に整列した一意なepoch checkpoint列。

    Raises
    ------
    ValueError
        checkpoint列が空、または正でない値を含む場合。
    """
    if len(epoch_checkpoints) == 0:
        raise ValueError("epoch_checkpoints must not be empty.")

    invalid_epochs = [epoch for epoch in epoch_checkpoints if epoch <= 0]
    if len(invalid_epochs) > 0:
        raise ValueError(f"epoch_checkpoints must be positive: {invalid_epochs}")

    return tuple(sorted(set(epoch_checkpoints)))


def resolve_project_path(path: Path) -> Path:
    """プロジェクトルート基準でパスを解決します。

    Parameters
    ----------
    path : pathlib.Path
        解決対象のパス。

    Returns
    -------
    pathlib.Path
        絶対パス。
    """
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def to_project_relative_str(path: Path) -> str:
    """プロジェクトルートからの相対パス文字列を返します。

    Parameters
    ----------
    path : pathlib.Path
        変換対象のパス。

    Returns
    -------
    str
        相対パス文字列。相対化できない場合は絶対パス文字列。
    """
    resolved = path.resolve()

    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def format_range_name(range_name: str) -> str:
    """評価範囲名を表示用に整形します。

    Parameters
    ----------
    range_name : str
        評価範囲名。

    Returns
    -------
    str
        表示用ラベル。
    """
    return RANGE_DISPLAY_NAMES.get(range_name, range_name)


def slugify(value: str) -> str:
    """ファイル名用に文字列を簡易正規化します。

    Parameters
    ----------
    value : str
        変換対象。

    Returns
    -------
    str
        ファイル名に使いやすい文字列。
    """
    normalized = value.strip().replace(" ", "_")
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in normalized)


def main() -> None:
    """grokking評価スクリプトのエントリーポイントです。

    Returns
    -------
    None
    """
    args = parse_args()
    epoch_checkpoints = validate_epoch_checkpoints(tuple(args.epochs))

    if args.all:
        summaries: dict[str, JsonDict] = {}

        for config_path in DEFAULT_CONFIG_PATHS:
            summary = evaluate_grokking_from_config_path(
                config_path,
                epoch_checkpoints=epoch_checkpoints,
                sweep_root=args.sweep_root,
                train_batch_size=args.train_batch_size,
                skip_existing=args.skip_existing,
            )
            model_name = str(summary["name"])
            summaries[model_name] = summary

        summary_path = save_merged_summary(summaries)
        print(f"\n[grokking_eval] all done summary={summary_path}")
        return

    evaluate_grokking_from_config_path(
        args.config,
        epoch_checkpoints=epoch_checkpoints,
        sweep_root=args.sweep_root,
        train_batch_size=args.train_batch_size,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()