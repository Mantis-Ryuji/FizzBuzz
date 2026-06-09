from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, TypeAlias

import matplotlib.pyplot as plt
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from fizzbuzz.config import ExperimentConfig, load_config
from fizzbuzz.evaluator import (
    ExtrapolationEvalResult,
    evaluate_extrapolation,
    merge_extrapolation_summaries,
)
from fizzbuzz.metrics import ClassificationMetrics
from fizzbuzz.model import build_model, count_parameters
from fizzbuzz.trainer import resolve_device
from fizzbuzz.utils import JsonDict, JsonValue, ensure_dir, save_json, seed_everything


MetricName: TypeAlias = Literal["accuracy", "macro_f1"]


DEFAULT_CONFIG_PATHS: tuple[Path, ...] = (
    Path("configs/small.yaml"),
    Path("configs/medium.yaml"),
    Path("configs/large.yaml"),
)


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析します。

    Returns
    -------
    argparse.Namespace
        解析済み引数。
    """
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate trained Digit GRU models on 6/7/8-digit "
            "FizzBuzz extrapolation ranges."
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
        "--weight",
        type=Path,
        default=None,
        help=(
            "Path to a model weight file. "
            "If omitted, runs/weights/<config.name>/model.pt is used."
        ),
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot generation.",
    )

    return parser.parse_args()


def evaluate_from_config_path(
    config_path: Path,
    *,
    weight_path: Path | None,
) -> tuple[str, ExtrapolationEvalResult]:
    """指定configの学習済みモデルを外挿評価します。

    Parameters
    ----------
    config_path : pathlib.Path
        評価に使うYAML configのパス。
    weight_path : pathlib.Path | None
        学習済み重みのパス。Noneの場合はconfigから標準パスを推定します。

    Returns
    -------
    tuple[str, ExtrapolationEvalResult]
        モデル名と外挿評価結果。

    Raises
    ------
    FileNotFoundError
        configまたは重みファイルが存在しない場合。
    """
    resolved_config_path = resolve_project_path(config_path)

    if not resolved_config_path.exists():
        raise FileNotFoundError(f"Config file not found: {resolved_config_path}")

    cfg = load_config(resolved_config_path)

    resolved_weight_path = (
        resolve_project_path(weight_path)
        if weight_path is not None
        else get_default_weight_path(cfg)
    )

    if not resolved_weight_path.exists():
        raise FileNotFoundError(f"Weight file not found: {resolved_weight_path}")

    result = evaluate_one_experiment(
        cfg,
        config_path=resolved_config_path,
        weight_path=resolved_weight_path,
    )

    return cfg.name, result


def evaluate_one_experiment(
    cfg: ExperimentConfig,
    *,
    config_path: Path,
    weight_path: Path,
) -> ExtrapolationEvalResult:
    """1つの学習済みモデルを6/7/8桁範囲で評価します。

    Parameters
    ----------
    cfg : ExperimentConfig
        検証済みの実験設定。
    config_path : pathlib.Path
        元configファイルのパス。
    weight_path : pathlib.Path
        学習済み重みファイルのパス。

    Returns
    -------
    ExtrapolationEvalResult
        外挿評価結果。
    """
    print(f"\n[eval] name={cfg.name}")
    print(f"[eval] config={config_path}")
    print(f"[eval] weight={weight_path}")

    seed_everything(cfg.seed)

    device = resolve_device(cfg.training.device)
    model = build_model(cfg.model)
    load_model_state(model, weight_path)

    num_parameters = count_parameters(model)
    print(f"[eval] device={device}")
    print(f"[eval] num_parameters={num_parameters:,}")

    result = evaluate_extrapolation(
        model,
        data_config=cfg.data,
        eval_config=cfg.eval,
        device=device,
    )

    output_dir = ensure_dir(resolve_project_path(cfg.output.weight_dir) / cfg.name)
    eval_path = output_dir / "eval_results.json"

    payload = build_eval_payload(
        cfg=cfg,
        config_path=config_path,
        weight_path=weight_path,
        result=result,
        num_parameters=num_parameters,
    )
    save_json(payload, eval_path)

    print(f"[eval] saved={eval_path}")

    for range_name, range_result in result.results.items():
        metrics = range_result.metrics
        print(
            "[eval] "
            f"{range_name} "
            f"acc={metrics.accuracy:.6f} "
            f"macro_f1={metrics.macro_f1:.6f} "
            f"n={metrics.num_samples:,} "
            f"sec={range_result.elapsed_sec:.2f}"
        )

    return result


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


def build_eval_payload(
    *,
    cfg: ExperimentConfig,
    config_path: Path,
    weight_path: Path,
    result: ExtrapolationEvalResult,
    num_parameters: int,
) -> JsonDict:
    """評価結果保存用のJSON payloadを作成します。

    Parameters
    ----------
    cfg : ExperimentConfig
        実験設定。
    config_path : pathlib.Path
        configファイルのパス。
    weight_path : pathlib.Path
        重みファイルのパス。
    result : ExtrapolationEvalResult
        外挿評価結果。
    num_parameters : int
        学習対象パラメータ数。

    Returns
    -------
    dict[str, JsonValue]
        JSON保存可能な評価結果。
    """
    return {
        "name": cfg.name,
        "seed": cfg.seed,
        "config_path": to_project_relative_str(config_path),
        "weight_path": to_project_relative_str(weight_path),
        "num_parameters": num_parameters,
        "results": result.to_dict(),
        "summary": result.summary_dict(),
    }


def save_merged_summary(
    results: Mapping[str, ExtrapolationEvalResult],
) -> Path:
    """複数モデルの評価結果をsummary JSONとして保存します。

    Parameters
    ----------
    results : Mapping[str, ExtrapolationEvalResult]
        モデル名から評価結果への対応。

    Returns
    -------
    pathlib.Path
        保存先パス。
    """
    summary = merge_extrapolation_summaries(results)
    summary_path = PROJECT_ROOT / "runs" / "eval_summary.json"
    save_json(summary, summary_path)
    return summary_path


def plot_metric_by_range(
    results: Mapping[str, ExtrapolationEvalResult],
    *,
    metric_name: MetricName,
    output_path: Path,
) -> None:
    """モデルサイズ別・桁数別の評価指標を折れ線グラフで保存します。

    Parameters
    ----------
    results : Mapping[str, ExtrapolationEvalResult]
        モデル名から外挿評価結果への対応。
    metric_name : {"accuracy", "macro_f1"}
        可視化する指標名。
    output_path : pathlib.Path
        保存先画像パス。

    Returns
    -------
    None

    Raises
    ------
    ValueError
        ``results`` が空の場合、または未対応の指標名の場合。
    """
    if len(results) == 0:
        raise ValueError("results must not be empty.")

    if metric_name not in {"accuracy", "macro_f1"}:
        raise ValueError(
            f"metric_name must be 'accuracy' or 'macro_f1', got {metric_name!r}."
        )

    first_result = next(iter(results.values()))
    range_names = list(first_result.results.keys())

    x_values = list(range(len(range_names)))

    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name, result in results.items():
        y_values = [
            get_metric_value(result.results[range_name].metrics, metric_name)
            for range_name in range_names
        ]
        ax.plot(
            x_values,
            y_values,
            marker="o",
            label=model_name,
        )

    ax.set_xticks(x_values)
    ax.set_xticklabels([format_range_name(name) for name in range_names])
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Evaluation range")
    ax.set_ylabel(format_metric_name(metric_name))
    ax.set_title(f"{format_metric_name(metric_name)} by extrapolation range")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(
    metrics: ClassificationMetrics,
    *,
    output_path: Path,
    title: str,
) -> None:
    """confusion matrixを画像として保存します。

    Parameters
    ----------
    metrics : ClassificationMetrics
        可視化対象の分類指標。
    output_path : pathlib.Path
        保存先画像パス。
    title : str
        図タイトル。

    Returns
    -------
    None
    """
    class_names = list(metrics.classwise.keys())
    cm = torch.tensor(metrics.confusion_matrix, dtype=torch.float64)

    row_sums = cm.sum(dim=1, keepdim=True)
    normalized = torch.zeros_like(cm)
    valid_rows = row_sums.squeeze(1) > 0
    normalized[valid_rows] = cm[valid_rows] / row_sums[valid_rows]

    values = normalized.numpy()

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(values, vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            ax.text(
                col_idx,
                row_idx,
                f"{values[row_idx, col_idx]:.2f}",
                ha="center",
                va="center",
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def make_plots(results: Mapping[str, ExtrapolationEvalResult]) -> None:
    """評価結果から記事用の図を作成します。

    Parameters
    ----------
    results : Mapping[str, ExtrapolationEvalResult]
        モデル名から外挿評価結果への対応。

    Returns
    -------
    None
    """
    image_dir = ensure_dir(PROJECT_ROOT / "runs" / "images")

    plot_metric_by_range(
        results,
        metric_name="accuracy",
        output_path=image_dir / "accuracy_by_digits.png",
    )
    plot_metric_by_range(
        results,
        metric_name="macro_f1",
        output_path=image_dir / "macro_f1_by_digits.png",
    )

    if "large" in results and "test_8digit" in results["large"].results:
        plot_confusion_matrix(
            results["large"].results["test_8digit"].metrics,
            output_path=image_dir / "confusion_matrix_large_8digit.png",
            title="Large model confusion matrix on 8-digit test",
        )

    print(f"[eval] plots saved={image_dir}")


def get_metric_value(
    metrics: ClassificationMetrics,
    metric_name: MetricName,
) -> float:
    """ClassificationMetricsから指定指標を取得します。

    Parameters
    ----------
    metrics : ClassificationMetrics
        評価指標。
    metric_name : {"accuracy", "macro_f1"}
        取得する指標名。

    Returns
    -------
    float
        指標値。

    Raises
    ------
    ValueError
        未対応の指標名の場合。
    """
    if metric_name == "accuracy":
        return metrics.accuracy
    if metric_name == "macro_f1":
        return metrics.macro_f1

    raise ValueError(f"Unsupported metric_name: {metric_name!r}")


def format_metric_name(metric_name: MetricName) -> str:
    """図表示用に指標名を整形します。

    Parameters
    ----------
    metric_name : {"accuracy", "macro_f1"}
        指標名。

    Returns
    -------
    str
        表示用の指標名。
    """
    if metric_name == "accuracy":
        return "Accuracy"
    if metric_name == "macro_f1":
        return "Macro F1"

    raise ValueError(f"Unsupported metric_name: {metric_name!r}")


def format_range_name(range_name: str) -> str:
    """評価範囲名を図表示用に整形します。

    Parameters
    ----------
    range_name : str
        評価範囲名。

    Returns
    -------
    str
        表示用ラベル。
    """
    replacements = {
        "test_6digit": "6-digit",
        "test_7digit": "7-digit",
        "test_8digit": "8-digit",
    }

    return replacements.get(range_name, range_name)


def get_default_weight_path(cfg: ExperimentConfig) -> Path:
    """configから標準の重みパスを返します。

    Parameters
    ----------
    cfg : ExperimentConfig
        実験設定。

    Returns
    -------
    pathlib.Path
        標準の重みパス。
    """
    return resolve_project_path(cfg.output.weight_dir) / cfg.name / "model.pt"


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


def main() -> None:
    """外挿評価スクリプトのエントリーポイントです。

    Returns
    -------
    None
    """
    args = parse_args()

    if args.all:
        if args.weight is not None:
            raise ValueError("--weight cannot be used with --all.")

        results: dict[str, ExtrapolationEvalResult] = {}

        for config_path in DEFAULT_CONFIG_PATHS:
            model_name, result = evaluate_from_config_path(
                config_path,
                weight_path=None,
            )
            results[model_name] = result

        summary_path = save_merged_summary(results)
        print(f"\n[eval] all done summary={summary_path}")

        if not args.skip_plots:
            make_plots(results)

        return

    model_name, result = evaluate_from_config_path(
        args.config,
        weight_path=args.weight,
    )

    single_results = {model_name: result}

    if not args.skip_plots:
        make_plots(single_results)


if __name__ == "__main__":
    main()