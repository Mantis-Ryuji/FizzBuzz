from __future__ import annotations

import argparse
import json
import shutil
import sys

import torch

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import is_dataclass, replace
from pathlib import Path
from typing import TypeVar, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from fizzbuzz.config import ExperimentConfig, load_config, save_config
from fizzbuzz.data import build_train_loader
from fizzbuzz.model import build_model, count_parameters
from fizzbuzz.trainer import Trainer, TrainHistory
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

DEFAULT_CHECKPOINT_EVERY_EPOCHS = 10

ConfigT = TypeVar("ConfigT")


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析します。

    Returns
    -------
    argparse.Namespace
        解析済み引数。
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train one grokking trajectory for each FizzBuzz model and save "
            "milestone weights at selected epochs."
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
        help="Train all default configs: small, medium, and large.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=list(DEFAULT_EPOCH_CHECKPOINTS),
        help=(
            "Milestone epochs to save. "
            f"Default: {list(DEFAULT_EPOCH_CHECKPOINTS)}."
        ),
    )
    parser.add_argument(
        "--checkpoint-every-epochs",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY_EPOCHS,
        help=(
            "Interval for saving resume checkpoint last.pt. "
            f"Default: {DEFAULT_CHECKPOINT_EVERY_EPOCHS}."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the existing grokking run directory and start from scratch.",
    )

    return parser.parse_args()


def train_sweep_from_config_path(
    config_path: Path,
    *,
    epoch_checkpoints: tuple[int, ...],
    checkpoint_every_epochs: int,
    overwrite: bool,
) -> JsonDict:
    """指定configからgrokking checkpoint学習を実行します。

    Parameters
    ----------
    config_path : pathlib.Path
        学習に使うYAML configのパス。
    epoch_checkpoints : tuple[int, ...]
        保存するmilestone epoch列。
    checkpoint_every_epochs : int
        resume用checkpointの保存間隔。
    overwrite : bool
        Trueの場合、既存のgrokking runを削除して最初から実行します。

    Returns
    -------
    dict[str, JsonValue]
        学習結果の要約。

    Raises
    ------
    FileNotFoundError
        configファイルが存在しない場合。
    ValueError
        ``epoch_checkpoints`` または ``checkpoint_every_epochs`` が不正な場合。
    """
    resolved_config_path = resolve_project_path(config_path)

    if not resolved_config_path.exists():
        raise FileNotFoundError(f"Config file not found: {resolved_config_path}")

    validated_epochs = validate_epoch_checkpoints(epoch_checkpoints)
    validate_checkpoint_every_epochs(checkpoint_every_epochs)

    cfg = load_config(resolved_config_path)
    return train_one_sweep(
        cfg,
        config_path=resolved_config_path,
        epoch_checkpoints=validated_epochs,
        checkpoint_every_epochs=checkpoint_every_epochs,
        overwrite=overwrite,
    )


def train_one_sweep(
    cfg: ExperimentConfig,
    *,
    config_path: Path,
    epoch_checkpoints: tuple[int, ...],
    checkpoint_every_epochs: int,
    overwrite: bool,
) -> JsonDict:
    """1つの実験設定でgrokking checkpoint学習を実行します。

    Parameters
    ----------
    cfg : ExperimentConfig
        検証済みの実験設定。
    config_path : pathlib.Path
        元configファイルのパス。
    epoch_checkpoints : tuple[int, ...]
        保存するmilestone epoch列。
    checkpoint_every_epochs : int
        resume用checkpointの保存間隔。
    overwrite : bool
        Trueの場合、既存のgrokking runを削除して最初から実行します。

    Returns
    -------
    dict[str, JsonValue]
        学習結果の要約。

    Raises
    ------
    ValueError
        学習履歴が空の場合。
    """
    max_epoch = max(epoch_checkpoints)
    cfg_for_run = copy_experiment_config_with_epochs(cfg, max_epoch)
    sweep_dir = PROJECT_ROOT / "runs" / "grokking" / cfg.name

    if overwrite and sweep_dir.exists():
        shutil.rmtree(sweep_dir)

    sweep_dir = ensure_dir(sweep_dir)
    model_path = sweep_dir / "model.pt"
    history_path = sweep_dir / "history.json"
    checkpoint_path = sweep_dir / "last.pt"
    milestone_dir = sweep_dir / "milestones"
    copied_config_path = sweep_dir / "config.yaml"
    summary_path = sweep_dir / "grokking_train_summary.json"

    print(f"\n[grokking_train] name={cfg.name}")
    print(f"[grokking_train] config={config_path}")
    print(f"[grokking_train] milestones={list(epoch_checkpoints)}")
    print(f"[grokking_train] max_epoch={max_epoch}")
    print(f"[grokking_train] output_dir={sweep_dir}")

    if model_path.exists() and not overwrite:
        print(f"[grokking_train] skip existing model={model_path}")
        return build_skipped_summary(
            cfg=cfg_for_run,
            config_path=config_path,
            sweep_dir=sweep_dir,
            model_path=model_path,
            checkpoint_path=checkpoint_path,
            history_path=history_path,
            milestone_dir=milestone_dir,
            epoch_checkpoints=epoch_checkpoints,
            summary_path=summary_path,
        )

    seed_everything(cfg_for_run.seed)

    generator = torch.Generator()
    generator.manual_seed(cfg_for_run.seed)

    train_loader = build_train_loader(
        cfg_for_run.data,
        batch_size=cfg_for_run.training.batch_size,
        num_workers=cfg_for_run.training.num_workers,
        shuffle=True,
        generator=generator,
    )

    model = build_model(cfg_for_run.model)
    num_parameters = count_parameters(model)

    print(f"[grokking_train] num_parameters={num_parameters:,}")
    print(f"[grokking_train] epochs={cfg_for_run.training.epochs}")
    print(f"[grokking_train] batch_size={cfg_for_run.training.batch_size}")
    if checkpoint_path.exists():
        print(f"[grokking_train] resume checkpoint={checkpoint_path}")

    trainer = Trainer(
        model=model,
        config=cfg_for_run.training,
        output_dir=sweep_dir,
    )

    history = trainer.fit(
        train_loader,
        save_model=True,
        model_filename=model_path.name,
        history_filename=history_path.name,
        resume=True,
        checkpoint_filename=checkpoint_path.name,
        checkpoint_every_epochs=checkpoint_every_epochs,
        milestone_epochs=epoch_checkpoints,
        milestone_dirname=milestone_dir.name,
    )

    save_config(cfg_for_run, copied_config_path)

    summary = build_train_summary(
        cfg=cfg_for_run,
        config_path=config_path,
        sweep_dir=sweep_dir,
        model_path=model_path,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        milestone_dir=milestone_dir,
        history=history,
        num_parameters=num_parameters,
        epoch_checkpoints=epoch_checkpoints,
        checkpoint_every_epochs=checkpoint_every_epochs,
        status="completed",
    )
    save_json(summary, summary_path)

    final_record = history.records[-1]
    print(
        "[grokking_train] done "
        f"name={cfg_for_run.name} "
        f"epoch={final_record.epoch} "
        f"loss={final_record.loss:.6f} "
        f"acc={final_record.accuracy:.6f} "
        f"model={model_path}"
    )

    return summary


def build_train_summary(
    *,
    cfg: ExperimentConfig,
    config_path: Path,
    sweep_dir: Path,
    model_path: Path,
    checkpoint_path: Path,
    history_path: Path,
    milestone_dir: Path,
    history: TrainHistory,
    num_parameters: int,
    epoch_checkpoints: tuple[int, ...],
    checkpoint_every_epochs: int,
    status: str,
) -> JsonDict:
    """学習結果の要約辞書を作成します。

    Parameters
    ----------
    cfg : ExperimentConfig
        実験設定。
    config_path : pathlib.Path
        元configファイルのパス。
    sweep_dir : pathlib.Path
        grokking runの出力ディレクトリ。
    model_path : pathlib.Path
        保存された最終重みファイルのパス。
    checkpoint_path : pathlib.Path
        resume用checkpointのパス。
    history_path : pathlib.Path
        学習履歴JSONのパス。
    milestone_dir : pathlib.Path
        milestone重みディレクトリ。
    history : TrainHistory
        学習履歴。
    num_parameters : int
        学習対象パラメータ数。
    epoch_checkpoints : tuple[int, ...]
        保存対象のmilestone epoch列。
    checkpoint_every_epochs : int
        resume用checkpointの保存間隔。
    status : str
        実行状態。

    Returns
    -------
    dict[str, JsonValue]
        JSON保存可能な学習要約。

    Raises
    ------
    ValueError
        学習履歴が空の場合。
    """
    if len(history.records) == 0:
        raise ValueError("history must contain at least one record.")

    final_record = history.records[-1]

    return {
        "name": cfg.name,
        "status": status,
        "seed": cfg.seed,
        "config_path": to_project_relative_str(config_path),
        "output_dir": to_project_relative_str(sweep_dir),
        "model_path": to_project_relative_str(model_path),
        "checkpoint_path": to_project_relative_str(checkpoint_path),
        "history_path": to_project_relative_str(history_path),
        "milestone_dir": to_project_relative_str(milestone_dir),
        "milestone_epochs": list(epoch_checkpoints),
        "checkpoint_every_epochs": checkpoint_every_epochs,
        "num_parameters": num_parameters,
        "epochs": cfg.training.epochs,
        "batch_size": cfg.training.batch_size,
        "lr": cfg.training.lr,
        "weight_decay": cfg.training.weight_decay,
        "final_epoch": final_record.epoch,
        "final_loss": final_record.loss,
        "final_accuracy": final_record.accuracy,
        "final_lr": final_record.lr,
        "total_elapsed_sec": sum(record.elapsed_sec for record in history.records),
    }


def build_skipped_summary(
    *,
    cfg: ExperimentConfig,
    config_path: Path,
    sweep_dir: Path,
    model_path: Path,
    checkpoint_path: Path,
    history_path: Path,
    milestone_dir: Path,
    epoch_checkpoints: tuple[int, ...],
    summary_path: Path,
) -> JsonDict:
    """既存モデルをskipした場合のsummaryを返します。

    Parameters
    ----------
    cfg : ExperimentConfig
        実験設定。
    config_path : pathlib.Path
        元configファイルのパス。
    sweep_dir : pathlib.Path
        grokking runの出力ディレクトリ。
    model_path : pathlib.Path
        既存の最終重みファイルのパス。
    checkpoint_path : pathlib.Path
        resume用checkpointのパス。
    history_path : pathlib.Path
        学習履歴JSONのパス。
    milestone_dir : pathlib.Path
        milestone重みディレクトリ。
    epoch_checkpoints : tuple[int, ...]
        保存対象のmilestone epoch列。
    summary_path : pathlib.Path
        既存summaryのパス。

    Returns
    -------
    dict[str, JsonValue]
        JSON保存可能なskip summary。
    """
    existing_summary = load_mapping_json(summary_path)
    if existing_summary is not None:
        existing_summary["status"] = "skipped_existing_model"
        return dict(existing_summary)  # type: ignore[return-value]

    return {
        "name": cfg.name,
        "status": "skipped_existing_model",
        "seed": cfg.seed,
        "config_path": to_project_relative_str(config_path),
        "output_dir": to_project_relative_str(sweep_dir),
        "model_path": to_project_relative_str(model_path),
        "checkpoint_path": to_project_relative_str(checkpoint_path),
        "history_path": to_project_relative_str(history_path),
        "milestone_dir": to_project_relative_str(milestone_dir),
        "milestone_epochs": list(epoch_checkpoints),
        "epochs": cfg.training.epochs,
        "batch_size": cfg.training.batch_size,
        "lr": cfg.training.lr,
        "weight_decay": cfg.training.weight_decay,
    }


def copy_experiment_config_with_epochs(
    cfg: ExperimentConfig,
    epochs: int,
) -> ExperimentConfig:
    """``training.epochs`` だけを差し替えたconfigを作成します。

    Parameters
    ----------
    cfg : ExperimentConfig
        元の実験設定。
    epochs : int
        差し替えるepoch数。

    Returns
    -------
    ExperimentConfig
        ``training.epochs`` を更新した実験設定。

    Raises
    ------
    ValueError
        ``epochs`` が正でない場合。
    AttributeError
        configが ``training`` または ``epochs`` を持たない場合。
    """
    if epochs <= 0:
        raise ValueError(f"epochs must be positive: {epochs}")

    training = copy_config_with_field(cfg.training, "epochs", epochs)
    return copy_config_with_field(cfg, "training", training)


def copy_config_with_field(
    cfg: ConfigT,
    field_name: str,
    value: object,
) -> ConfigT:
    """dataclass/Pydantic/通常オブジェクトのフィールドをコピー更新します。

    Parameters
    ----------
    cfg : ConfigT
        更新対象の設定オブジェクト。
    field_name : str
        更新するフィールド名。
    value : object
        新しいフィールド値。

    Returns
    -------
    ConfigT
        指定フィールドを更新したコピー。

    Raises
    ------
    AttributeError
        ``cfg`` が指定フィールドを持たない場合。
    """
    if not hasattr(cfg, field_name):
        raise AttributeError(f"Config object does not have field: {field_name}")

    cfg_copy = deepcopy(cfg)

    if is_dataclass(cfg_copy):
        return cast(ConfigT, replace(cfg_copy, **{field_name: value}))  # type: ignore[arg-type]

    model_copy = getattr(cfg_copy, "model_copy", None)
    if callable(model_copy):
        return cast(ConfigT, model_copy(update={field_name: value}))

    copy_method = getattr(cfg_copy, "copy", None)
    if callable(copy_method):
        return cast(ConfigT, copy_method(update={field_name: value}))

    setattr(cfg_copy, field_name, value)
    return cfg_copy


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


def validate_checkpoint_every_epochs(checkpoint_every_epochs: int) -> None:
    """resume checkpoint保存間隔を検証します。

    Parameters
    ----------
    checkpoint_every_epochs : int
        resume checkpoint保存間隔。

    Returns
    -------
    None

    Raises
    ------
    ValueError
        ``checkpoint_every_epochs`` が正でない場合。
    """
    if checkpoint_every_epochs <= 0:
        raise ValueError(
            "checkpoint_every_epochs must be positive, "
            f"got {checkpoint_every_epochs}."
        )


def load_mapping_json(path: Path) -> dict[str, object] | None:
    """JSONファイルをMappingとして読み込みます。

    Parameters
    ----------
    path : pathlib.Path
        読み込み対象JSONのパス。

    Returns
    -------
    dict[str, object] | None
        読み込みに成功したMapping。存在しない場合はNone。

    Raises
    ------
    ValueError
        JSONがMappingでない場合。
    """
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON must contain a mapping: {path}")

    return dict(payload)


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
    """epoch sweepスクリプトのエントリーポイントです。

    Returns
    -------
    None
    """
    args = parse_args()
    epoch_checkpoints = validate_epoch_checkpoints(tuple(args.epochs))
    validate_checkpoint_every_epochs(args.checkpoint_every_epochs)

    if args.all:
        summaries: dict[str, JsonDict] = {}

        for config_path in DEFAULT_CONFIG_PATHS:
            summary = train_sweep_from_config_path(
                config_path,
                epoch_checkpoints=epoch_checkpoints,
                checkpoint_every_epochs=args.checkpoint_every_epochs,
                overwrite=args.overwrite,
            )
            model_name = str(summary["name"])
            summaries[model_name] = summary

        summary_dir = ensure_dir(PROJECT_ROOT / "runs" / "grokking")
        summary_path = summary_dir / "grokking_train_summary.json"
        save_json({"by_model": summaries}, summary_path)  # type: ignore[arg-type]

        print(f"\n[grokking_train] all done summary={summary_path}")
        return

    train_sweep_from_config_path(
        args.config,
        epoch_checkpoints=epoch_checkpoints,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()