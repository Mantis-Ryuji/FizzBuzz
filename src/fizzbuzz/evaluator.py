from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import ceil
from time import perf_counter
from typing import TypeAlias

import torch
from torch import nn
from tqdm.auto import tqdm

from fizzbuzz.config import DataConfig, EvalConfig
from fizzbuzz.data import DigitBatch, iter_integer_batches, make_batch_from_range
from fizzbuzz.metrics import (
    ClassificationMetrics,
    compute_metrics_from_confusion_matrix,
    init_confusion_matrix,
    predictions_from_logits,
    update_confusion_matrix,
)


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonDict: TypeAlias = dict[str, JsonValue]


@dataclass(frozen=True)
class RangeEvalResult:
    """1つの整数範囲に対する評価結果です。

    Parameters
    ----------
    name : str
        評価範囲名です。例として ``test_6digit`` などを想定します。
    start : int
        評価範囲の開始値です。
    end : int
        評価範囲の終了値です。
    batch_size : int
        評価時のbatch sizeです。
    metrics : ClassificationMetrics
        confusion matrixから計算された分類指標です。
    elapsed_sec : float
        評価にかかった秒数です。

    Notes
    -----
    評価対象の整数全体はメモリに保持せず、batch単位で生成します。
    """

    name: str
    start: int
    end: int
    batch_size: int
    metrics: ClassificationMetrics
    elapsed_sec: float

    @property
    def num_samples(self) -> int:
        """評価サンプル数を返します。

        Returns
        -------
        int
            評価サンプル数。
        """
        return self.end - self.start + 1

    def to_dict(self) -> JsonDict:
        """JSON保存可能な辞書へ変換します。

        Returns
        -------
        dict[str, JsonValue]
            JSON保存可能な辞書。
        """
        return {
            "name": self.name,
            "start": self.start,
            "end": self.end,
            "batch_size": self.batch_size,
            "num_samples": self.num_samples,
            "elapsed_sec": self.elapsed_sec,
            "metrics": self.metrics.to_dict(),
        }


@dataclass(frozen=True)
class ExtrapolationEvalResult:
    """外挿評価全体の結果です。

    Parameters
    ----------
    results : dict[str, RangeEvalResult]
        評価範囲名ごとの評価結果です。

    Notes
    -----
    通常は ``test_6digit``、``test_7digit``、``test_8digit`` の3つを持ちます。
    """

    results: dict[str, RangeEvalResult]

    def to_dict(self) -> JsonDict:
        """JSON保存可能な辞書へ変換します。

        Returns
        -------
        dict[str, JsonValue]
            JSON保存可能な辞書。
        """
        return {
            name: result.to_dict()
            for name, result in self.results.items()
        }

    def summary_dict(self) -> JsonDict:
        """記事用の集計に使いやすい辞書へ変換します。

        Returns
        -------
        dict[str, JsonValue]
            AccuracyとMacro F1を中心にした要約辞書。
        """
        return {
            name: {
                "accuracy": result.metrics.accuracy,
                "macro_precision": result.metrics.macro_precision,
                "macro_recall": result.metrics.macro_recall,
                "macro_f1": result.metrics.macro_f1,
                "num_samples": result.metrics.num_samples,
                "elapsed_sec": result.elapsed_sec,
            }
            for name, result in self.results.items()
        }


def evaluate_range(
    model: nn.Module,
    *,
    data_config: DataConfig,
    name: str,
    start: int,
    end: int,
    batch_size: int,
    device: torch.device | str,
) -> RangeEvalResult:
    """指定された整数範囲でモデルを評価します。

    Parameters
    ----------
    model : torch.nn.Module
        評価対象のモデル。
    data_config : DataConfig
        データ設定。
    name : str
        評価範囲名。
    start : int
        評価範囲の開始値。
    end : int
        評価範囲の終了値。
    batch_size : int
        評価時のbatch size。
    device : torch.device or str
        評価に使うデバイス。

    Returns
    -------
    RangeEvalResult
        指定範囲に対する評価結果。

    Raises
    ------
    TypeError
        ``model`` が ``torch.nn.Module`` でない場合。
    ValueError
        評価範囲、batch size、評価名が不正な場合。
    RuntimeError
        指定デバイスが利用できない場合。
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be nn.Module, got {type(model).__name__}.")

    _validate_range_args(
        name=name,
        start=start,
        end=end,
        batch_size=batch_size,
    )

    device_obj = _resolve_device(device)

    confusion_matrix = init_confusion_matrix(
        num_classes=data_config.num_classes,
        device="cpu",
    )

    model = model.to(device_obj)
    was_training = model.training
    model.eval()

    total_samples = end - start + 1
    total_batches = ceil(total_samples / batch_size)
    start_time = perf_counter()

    progress_bar = tqdm(
        iter_integer_batches(
            start=start,
            end=end,
            batch_size=batch_size,
        ),
        total=total_batches,
        desc=f"Eval {name}",
        unit="batch",
        dynamic_ncols=True,
    )

    try:
        with torch.inference_mode():
            for number_batch in progress_bar:
                batch = make_batch_from_range(
                    number_batch,
                    pad_token_id=data_config.pad_token_id,
                )
                batch_on_device = move_batch_to_device(batch, device=device_obj)

                logits = model(batch_on_device.digits, batch_on_device.lengths)
                preds = predictions_from_logits(logits).to(device="cpu")

                confusion_matrix = update_confusion_matrix(
                    confusion_matrix,
                    y_true=batch.labels,
                    y_pred=preds,
                )

                seen_samples = int(confusion_matrix.sum().item())
                progress_bar.set_postfix(
                    {
                        "seen": f"{seen_samples:,}/{total_samples:,}",
                    }
                )
    finally:
        if was_training:
            model.train()

    elapsed_sec = perf_counter() - start_time
    metrics = compute_metrics_from_confusion_matrix(confusion_matrix)

    return RangeEvalResult(
        name=name,
        start=start,
        end=end,
        batch_size=batch_size,
        metrics=metrics,
        elapsed_sec=elapsed_sec,
    )


def evaluate_extrapolation(
    model: nn.Module,
    *,
    data_config: DataConfig,
    eval_config: EvalConfig,
    device: torch.device | str,
) -> ExtrapolationEvalResult:
    """設定された全ての桁数範囲で外挿評価を行います。

    Parameters
    ----------
    model : torch.nn.Module
        評価対象のモデル。
    data_config : DataConfig
        データ設定。
    eval_config : EvalConfig
        外挿評価設定。
    device : torch.device or str
        評価に使うデバイス。

    Returns
    -------
    ExtrapolationEvalResult
        外挿評価全体の結果。

    Raises
    ------
    TypeError
        ``model`` が ``torch.nn.Module`` でない場合。
    ValueError
        評価設定が不正な場合。
    RuntimeError
        指定デバイスが利用できない場合。
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be nn.Module, got {type(model).__name__}.")

    if len(eval_config.digit_ranges) == 0:
        raise ValueError("eval_config.digit_ranges must not be empty.")

    results: dict[str, RangeEvalResult] = {}

    range_items = list(eval_config.digit_ranges.items())
    outer_bar = tqdm(
        range_items,
        desc="Extrapolation eval",
        unit="range",
        dynamic_ncols=True,
    )

    for range_name, (start, end) in outer_bar:
        outer_bar.set_postfix({"range": range_name})

        results[range_name] = evaluate_range(
            model,
            data_config=data_config,
            name=range_name,
            start=start,
            end=end,
            batch_size=eval_config.batch_size,
            device=device,
        )

    return ExtrapolationEvalResult(results=results)


def move_batch_to_device(
    batch: DigitBatch,
    *,
    device: torch.device | str,
) -> DigitBatch:
    """DigitBatchを指定デバイスへ転送します。

    Parameters
    ----------
    batch : DigitBatch
        転送対象のバッチ。
    device : torch.device or str
        転送先デバイス。

    Returns
    -------
    DigitBatch
        指定デバイスへ転送されたバッチ。

    Raises
    ------
    TypeError
        ``batch`` が ``DigitBatch`` でない場合。
    RuntimeError
        指定デバイスが利用できない場合。
    """
    if not isinstance(batch, DigitBatch):
        raise TypeError(f"batch must be DigitBatch, got {type(batch).__name__}.")

    device_obj = _resolve_device(device)

    return DigitBatch(
        digits=batch.digits.to(device_obj, non_blocking=True),
        lengths=batch.lengths.to(device_obj, non_blocking=True),
        labels=batch.labels.to(device_obj, non_blocking=True),
    )


def collect_summary_rows(
    model_name: str,
    result: ExtrapolationEvalResult,
) -> list[JsonDict]:
    """外挿評価結果を表形式に近い行リストへ変換します。

    Parameters
    ----------
    model_name : str
        モデル名。例として ``small``、``medium``、``large`` を想定します。
    result : ExtrapolationEvalResult
        外挿評価結果。

    Returns
    -------
    list[dict[str, JsonValue]]
        各評価範囲の要約行。

    Raises
    ------
    ValueError
        ``model_name`` が空の場合。
    """
    cleaned_model_name = model_name.strip()
    if cleaned_model_name == "":
        raise ValueError("model_name must not be empty.")

    rows: list[JsonDict] = []

    for range_name, range_result in result.results.items():
        rows.append(
            {
                "model": cleaned_model_name,
                "range": range_name,
                "start": range_result.start,
                "end": range_result.end,
                "num_samples": range_result.metrics.num_samples,
                "accuracy": range_result.metrics.accuracy,
                "macro_precision": range_result.metrics.macro_precision,
                "macro_recall": range_result.metrics.macro_recall,
                "macro_f1": range_result.metrics.macro_f1,
                "elapsed_sec": range_result.elapsed_sec,
            }
        )

    return rows


def merge_extrapolation_summaries(
    results: Mapping[str, ExtrapolationEvalResult],
) -> JsonDict:
    """複数モデルの外挿評価結果をsummary形式へ統合します。

    Parameters
    ----------
    results : Mapping[str, ExtrapolationEvalResult]
        モデル名から外挿評価結果への対応です。

    Returns
    -------
    dict[str, JsonValue]
        JSON保存可能なsummary辞書。

    Raises
    ------
    ValueError
        ``results`` が空の場合。
    """
    if len(results) == 0:
        raise ValueError("results must not be empty.")

    rows: list[JsonValue] = []
    by_model: dict[str, JsonValue] = {}

    for model_name, result in results.items():
        rows.extend(collect_summary_rows(model_name, result))
        by_model[model_name] = result.summary_dict()

    return {
        "rows": rows,
        "by_model": by_model,
    }


def _validate_range_args(
    *,
    name: str,
    start: int,
    end: int,
    batch_size: int,
) -> None:
    """評価範囲引数を検証します。

    Parameters
    ----------
    name : str
        評価範囲名。
    start : int
        評価開始値。
    end : int
        評価終了値。
    batch_size : int
        batch size。

    Returns
    -------
    None

    Raises
    ------
    ValueError
        引数が不正な場合。
    """
    if name.strip() == "":
        raise ValueError("name must not be empty.")
    if start <= 0:
        raise ValueError(f"start must be positive, got {start}.")
    if end <= 0:
        raise ValueError(f"end must be positive, got {end}.")
    if start > end:
        raise ValueError(
            f"start must be less than or equal to end, got {start} > {end}."
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")


def _resolve_device(device: torch.device | str) -> torch.device:
    """デバイス指定をtorch.deviceへ変換し、利用可能性を確認します。

    Parameters
    ----------
    device : torch.device or str
        デバイス指定。

    Returns
    -------
    torch.device
        解決されたデバイス。

    Raises
    ------
    RuntimeError
        指定デバイスが利用できない場合。
    """
    device_obj = torch.device(device)

    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device was requested, but CUDA is not available.")

    if device_obj.type == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not torch.backends.mps.is_available():
            raise RuntimeError("MPS device was requested, but MPS is not available.")

    return device_obj