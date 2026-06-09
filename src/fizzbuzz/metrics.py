from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, cast

import torch

from fizzbuzz.data import CLASS_NAMES


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonDict: TypeAlias = dict[str, JsonValue]


@dataclass(frozen=True)
class ClassMetrics:
    """1クラス分の分類指標です。

    Parameters
    ----------
    precision : float
        そのクラスと予測したサンプルのうち、実際にそのクラスだった割合です。
    recall : float
        実際にそのクラスであるサンプルのうち、正しくそのクラスと予測できた割合です。
    f1 : float
        Precision と Recall のバランスを表す指標です。
    support : int
        そのクラスに属する正解サンプル数です。
    """

    precision: float
    recall: float
    f1: float
    support: int

    def to_dict(self) -> JsonDict:
        """JSON保存可能な辞書へ変換します。

        Returns
        -------
        dict[str, JsonValue]
            JSON保存可能な辞書。
        """
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "support": self.support,
        }


@dataclass(frozen=True)
class ClassificationMetrics:
    """4クラス分類の評価結果です。

    Parameters
    ----------
    accuracy : float
        全体の正解率です。
    macro_precision : float
        クラス別Precisionの単純平均です。
    macro_recall : float
        クラス別Recallの単純平均です。
    macro_f1 : float
        クラス別F1-scoreの単純平均です。
    classwise : dict[str, ClassMetrics]
        クラス名ごとの評価指標です。
    confusion_matrix : list[list[int]]
        confusion matrixです。行が正解ラベル、列が予測ラベルです。
    num_samples : int
        評価サンプル数です。
    """

    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    classwise: dict[str, ClassMetrics]
    confusion_matrix: list[list[int]]
    num_samples: int

    def to_dict(self) -> JsonDict:
        """JSON保存可能な辞書へ変換します。

        Returns
        -------
        dict[str, JsonValue]
            JSON保存可能な辞書。
        """
        return {
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "classwise": {
                class_name: metrics.to_dict()
                for class_name, metrics in self.classwise.items()
            },
            "confusion_matrix": self.confusion_matrix, # type: ignore
            "num_samples": self.num_samples,
        }


def init_confusion_matrix(
    *,
    num_classes: int = len(CLASS_NAMES),
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """空のconfusion matrixを作成します。

    Parameters
    ----------
    num_classes : int, default=len(CLASS_NAMES)
        クラス数です。
    device : torch.device or str, default="cpu"
        confusion matrixを作成するデバイスです。

    Returns
    -------
    torch.Tensor
        形状 ``(num_classes, num_classes)`` のconfusion matrixです。

    Raises
    ------
    ValueError
        ``num_classes`` が2未満の場合。
    """
    if num_classes < 2:
        raise ValueError(f"num_classes must be greater than 1, got {num_classes}.")

    return torch.zeros(
        (num_classes, num_classes),
        dtype=torch.long,
        device=device,
    )


def update_confusion_matrix(
    confusion_matrix: torch.Tensor,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> torch.Tensor:
    """confusion matrixを更新します。

    Parameters
    ----------
    confusion_matrix : torch.Tensor
        現在のconfusion matrixです。行が正解ラベル、列が予測ラベルです。
    y_true : torch.Tensor
        正解ラベルです。
    y_pred : torch.Tensor
        予測ラベルです。

    Returns
    -------
    torch.Tensor
        更新後のconfusion matrixです。

    Raises
    ------
    ValueError
        入力テンソルの形状、dtype、ラベル範囲が不正な場合。
    """
    _validate_confusion_matrix(confusion_matrix)
    _validate_label_tensors(y_true=y_true, y_pred=y_pred)

    num_classes = confusion_matrix.size(0)
    y_true_flat = y_true.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    y_pred_flat = y_pred.detach().to(device="cpu", dtype=torch.long).reshape(-1)

    _validate_label_range(y_true_flat, num_classes=num_classes, name="y_true")
    _validate_label_range(y_pred_flat, num_classes=num_classes, name="y_pred")

    indices = y_true_flat * num_classes + y_pred_flat
    counts = torch.bincount(
        indices,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)

    counts = counts.to(
        device=confusion_matrix.device,
        dtype=confusion_matrix.dtype,
    )

    return confusion_matrix + counts


def compute_metrics_from_confusion_matrix(
    confusion_matrix: torch.Tensor,
    *,
    class_names: tuple[str, ...] = CLASS_NAMES,
    zero_division: float = 0.0,
) -> ClassificationMetrics:
    """confusion matrixから分類指標を計算します。

    Parameters
    ----------
    confusion_matrix : torch.Tensor
        confusion matrixです。行が正解ラベル、列が予測ラベルです。
    class_names : tuple[str, ...], default=CLASS_NAMES
        クラス名です。
    zero_division : float, default=0.0
        PrecisionやRecallの分母が0の場合に使う値です。

    Returns
    -------
    ClassificationMetrics
        Accuracy、class-wise指標、Macro平均、confusion matrixを含む評価結果。

    Raises
    ------
    ValueError
        confusion matrixやクラス名の形式が不正な場合。
    """
    _validate_confusion_matrix(confusion_matrix)

    num_classes = confusion_matrix.size(0)
    if len(class_names) != num_classes:
        raise ValueError(
            "class_names length must match num_classes, "
            f"got len(class_names)={len(class_names)}, num_classes={num_classes}."
        )

    if zero_division < 0.0 or zero_division > 1.0:
        raise ValueError(
            f"zero_division must be in [0, 1], got {zero_division}."
        )

    cm = confusion_matrix.detach().to(device="cpu", dtype=torch.float64)
    num_samples = int(cm.sum().item())

    if num_samples <= 0:
        raise ValueError("confusion_matrix must contain at least one sample.")

    true_positive = torch.diag(cm)
    support = cm.sum(dim=1)
    predicted = cm.sum(dim=0)

    precision = _safe_divide(
        numerator=true_positive,
        denominator=predicted,
        fill_value=zero_division,
    )
    recall = _safe_divide(
        numerator=true_positive,
        denominator=support,
        fill_value=zero_division,
    )
    f1 = _compute_f1(
        precision=precision,
        recall=recall,
        fill_value=zero_division,
    )

    accuracy = float(true_positive.sum().item() / num_samples)

    classwise: dict[str, ClassMetrics] = {}
    for class_idx, class_name in enumerate(class_names):
        classwise[class_name] = ClassMetrics(
            precision=float(precision[class_idx].item()),
            recall=float(recall[class_idx].item()),
            f1=float(f1[class_idx].item()),
            support=int(support[class_idx].item()),
        )

    confusion_as_list = cast(
        list[list[int]],
        confusion_matrix.detach().to(device="cpu", dtype=torch.long).tolist(),
    )

    return ClassificationMetrics(
        accuracy=accuracy,
        macro_precision=float(precision.mean().item()),
        macro_recall=float(recall.mean().item()),
        macro_f1=float(f1.mean().item()),
        classwise=classwise,
        confusion_matrix=confusion_as_list,
        num_samples=num_samples,
    )


def accuracy_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """logitsと正解ラベルからAccuracyを計算します。

    Parameters
    ----------
    logits : torch.Tensor
        モデルの出力logitsです。形状は ``(batch_size, num_classes)`` です。
    labels : torch.Tensor
        正解ラベルです。形状は ``(batch_size,)`` です。

    Returns
    -------
    float
        Accuracy。

    Raises
    ------
    ValueError
        入力テンソルの形状やdtypeが不正な場合。
    """
    if logits.ndim != 2:
        raise ValueError(
            f"logits must be a 2D tensor, got shape={tuple(logits.shape)}."
        )
    if labels.ndim != 1:
        raise ValueError(
            f"labels must be a 1D tensor, got shape={tuple(labels.shape)}."
        )
    if logits.size(0) != labels.size(0):
        raise ValueError(
            "batch size mismatch between logits and labels, "
            f"got logits batch={logits.size(0)}, labels batch={labels.size(0)}."
        )
    if labels.dtype != torch.long:
        raise ValueError(f"labels dtype must be torch.long, got {labels.dtype}.")
    if labels.numel() == 0:
        raise ValueError("labels must not be empty.")

    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()

    return float(correct / labels.numel())


def predictions_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """logitsから予測ラベルを返します。

    Parameters
    ----------
    logits : torch.Tensor
        モデルの出力logitsです。形状は ``(batch_size, num_classes)`` です。

    Returns
    -------
    torch.Tensor
        予測ラベルです。形状は ``(batch_size,)`` です。

    Raises
    ------
    ValueError
        ``logits`` の形状が不正な場合。
    """
    if logits.ndim != 2:
        raise ValueError(
            f"logits must be a 2D tensor, got shape={tuple(logits.shape)}."
        )
    if logits.numel() == 0:
        raise ValueError("logits must not be empty.")

    return logits.argmax(dim=1)


def _safe_divide(
    *,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    fill_value: float,
) -> torch.Tensor:
    """ゼロ除算を避けてテンソル同士を割ります。

    Parameters
    ----------
    numerator : torch.Tensor
        分子です。
    denominator : torch.Tensor
        分母です。
    fill_value : float
        分母が0の場合に使う値です。

    Returns
    -------
    torch.Tensor
        割り算の結果です。
    """
    if numerator.shape != denominator.shape:
        raise ValueError(
            "numerator and denominator must have the same shape, "
            f"got {tuple(numerator.shape)} and {tuple(denominator.shape)}."
        )

    output = torch.full_like(numerator, fill_value=fill_value)
    valid = denominator != 0

    output[valid] = numerator[valid] / denominator[valid]
    return output


def _compute_f1(
    *,
    precision: torch.Tensor,
    recall: torch.Tensor,
    fill_value: float,
) -> torch.Tensor:
    """PrecisionとRecallからF1-scoreを計算します。

    Parameters
    ----------
    precision : torch.Tensor
        クラス別Precisionです。
    recall : torch.Tensor
        クラス別Recallです。
    fill_value : float
        分母が0の場合に使う値です。

    Returns
    -------
    torch.Tensor
        クラス別F1-scoreです。
    """
    if precision.shape != recall.shape:
        raise ValueError(
            "precision and recall must have the same shape, "
            f"got {tuple(precision.shape)} and {tuple(recall.shape)}."
        )

    denominator = precision + recall
    numerator = 2.0 * precision * recall

    return _safe_divide(
        numerator=numerator,
        denominator=denominator,
        fill_value=fill_value,
    )


def _validate_confusion_matrix(confusion_matrix: torch.Tensor) -> None:
    """confusion matrixの形式を検証します。

    Parameters
    ----------
    confusion_matrix : torch.Tensor
        検証対象のconfusion matrix。

    Returns
    -------
    None

    Raises
    ------
    ValueError
        confusion matrixの形状、dtype、値が不正な場合。
    """
    if confusion_matrix.ndim != 2:
        raise ValueError(
            "confusion_matrix must be a 2D tensor, "
            f"got shape={tuple(confusion_matrix.shape)}."
        )
    if confusion_matrix.size(0) != confusion_matrix.size(1):
        raise ValueError(
            "confusion_matrix must be square, "
            f"got shape={tuple(confusion_matrix.shape)}."
        )
    if confusion_matrix.size(0) < 2:
        raise ValueError(
            "confusion_matrix must have at least 2 classes, "
            f"got {confusion_matrix.size(0)}."
        )
    if confusion_matrix.dtype not in (torch.int32, torch.int64, torch.long):
        raise ValueError(
            "confusion_matrix dtype must be an integer dtype, "
            f"got {confusion_matrix.dtype}."
        )
    if confusion_matrix.numel() == 0:
        raise ValueError("confusion_matrix must not be empty.")
    if int(confusion_matrix.min().item()) < 0:
        raise ValueError("confusion_matrix must not contain negative values.")


def _validate_label_tensors(
    *,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> None:
    """正解ラベルと予測ラベルの形式を検証します。

    Parameters
    ----------
    y_true : torch.Tensor
        正解ラベルです。
    y_pred : torch.Tensor
        予測ラベルです。

    Returns
    -------
    None

    Raises
    ------
    ValueError
        ラベルテンソルの形状やdtypeが不正な場合。
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape, "
            f"got {tuple(y_true.shape)} and {tuple(y_pred.shape)}."
        )
    if y_true.numel() == 0:
        raise ValueError("y_true and y_pred must not be empty.")
    if y_true.dtype != torch.long:
        raise ValueError(f"y_true dtype must be torch.long, got {y_true.dtype}.")
    if y_pred.dtype != torch.long:
        raise ValueError(f"y_pred dtype must be torch.long, got {y_pred.dtype}.")


def _validate_label_range(
    labels: torch.Tensor,
    *,
    num_classes: int,
    name: str,
) -> None:
    """ラベル値がクラス範囲内にあるか検証します。

    Parameters
    ----------
    labels : torch.Tensor
        検証対象のラベルテンソル。
    num_classes : int
        クラス数。
    name : str
        エラーメッセージに表示する変数名。

    Returns
    -------
    None

    Raises
    ------
    ValueError
        ラベルが範囲外の場合。
    """
    min_label = int(labels.min().item())
    max_label = int(labels.max().item())

    if min_label < 0:
        raise ValueError(f"{name} must be non-negative, got min={min_label}.")
    if max_label >= num_classes:
        raise ValueError(
            f"{name} must be smaller than num_classes={num_classes}, "
            f"got max={max_label}."
        )