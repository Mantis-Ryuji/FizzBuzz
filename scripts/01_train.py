from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

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


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析します。

    Returns
    -------
    argparse.Namespace
        解析済み引数。
    """
    parser = argparse.ArgumentParser(
        description="Train Digit GRU models for the FizzBuzz extrapolation experiment."
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
        "--overwrite",
        action="store_true",
        help="Overwrite existing model.pt if it already exists.",
    )

    return parser.parse_args()


def train_from_config_path(
    config_path: Path,
    *,
    overwrite: bool,
) -> JsonDict:
    """指定configから1モデルを学習します。

    Parameters
    ----------
    config_path : pathlib.Path
        学習に使うYAML configのパス。
    overwrite : bool
        Trueの場合、既存の ``model.pt`` を上書きします。

    Returns
    -------
    dict[str, JsonValue]
        学習結果の要約。

    Raises
    ------
    FileNotFoundError
        configファイルが存在しない場合。
    FileExistsError
        ``model.pt`` が既に存在し、かつ ``overwrite=False`` の場合。
    """
    resolved_config_path = resolve_project_path(config_path)

    if not resolved_config_path.exists():
        raise FileNotFoundError(f"Config file not found: {resolved_config_path}")

    cfg = load_config(resolved_config_path)
    return train_one_experiment(
        cfg,
        config_path=resolved_config_path,
        overwrite=overwrite,
    )


def train_one_experiment(
    cfg: ExperimentConfig,
    *,
    config_path: Path,
    overwrite: bool,
) -> JsonDict:
    """1つの実験設定でモデルを学習します。

    Parameters
    ----------
    cfg : ExperimentConfig
        検証済みの実験設定。
    config_path : pathlib.Path
        元configファイルのパス。
    overwrite : bool
        Trueの場合、既存の ``model.pt`` を上書きします。

    Returns
    -------
    dict[str, JsonValue]
        学習結果の要約。

    Raises
    ------
    FileExistsError
        ``model.pt`` が既に存在し、かつ ``overwrite=False`` の場合。
    ValueError
        学習履歴が空の場合。
    """
    output_dir = ensure_dir(resolve_project_path(cfg.output.weight_dir) / cfg.name)
    model_path = output_dir / "model.pt"
    history_path = output_dir / "history.json"
    copied_config_path = output_dir / "config.yaml"
    summary_path = output_dir / "train_summary.json"

    if model_path.exists() and not overwrite:
        raise FileExistsError(
            f"Model already exists: {model_path}. "
            "Use --overwrite to overwrite it."
        )

    print(f"\n[train] name={cfg.name}")
    print(f"[train] config={config_path}")
    print(f"[train] output_dir={output_dir}")

    seed_everything(cfg.seed)

    generator = torch.Generator()
    generator.manual_seed(cfg.seed)

    train_loader = build_train_loader(
        cfg.data,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=True,
        generator=generator,
    )

    model = build_model(cfg.model)
    num_parameters = count_parameters(model)

    print(f"[train] num_parameters={num_parameters:,}")
    print(f"[train] epochs={cfg.training.epochs}")
    print(f"[train] batch_size={cfg.training.batch_size}")

    trainer = Trainer(
        model=model,
        config=cfg.training,
        output_dir=output_dir,
    )

    history = trainer.fit(
        train_loader,
        save_model=True,
        model_filename=model_path.name,
        history_filename=history_path.name,
    )

    save_config(cfg, copied_config_path)

    summary = build_train_summary(
        cfg=cfg,
        config_path=config_path,
        model_path=model_path,
        history=history,
        num_parameters=num_parameters,
    )
    save_json(summary, summary_path)

    final_record = history.records[-1]
    print(
        "[train] done "
        f"name={cfg.name} "
        f"loss={final_record.loss:.6f} "
        f"acc={final_record.accuracy:.6f} "
        f"model={model_path}"
    )

    return summary


def build_train_summary(
    *,
    cfg: ExperimentConfig,
    config_path: Path,
    model_path: Path,
    history: TrainHistory,
    num_parameters: int,
) -> JsonDict:
    """学習結果の要約辞書を作成します。

    Parameters
    ----------
    cfg : ExperimentConfig
        実験設定。
    config_path : pathlib.Path
        元configファイルのパス。
    model_path : pathlib.Path
        保存された重みファイルのパス。
    history : TrainHistory
        学習履歴。
    num_parameters : int
        学習対象パラメータ数。

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
        "seed": cfg.seed,
        "config_path": str(config_path.relative_to(PROJECT_ROOT)),
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "num_parameters": num_parameters,
        "epochs": cfg.training.epochs,
        "batch_size": cfg.training.batch_size,
        "lr": cfg.training.lr,
        "weight_decay": cfg.training.weight_decay,
        "final_loss": final_record.loss,
        "final_accuracy": final_record.accuracy,
        "final_lr": final_record.lr,
        "total_elapsed_sec": sum(record.elapsed_sec for record in history.records),
    }


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


def main() -> None:
    """学習スクリプトのエントリーポイントです。

    Returns
    -------
    None
    """
    args = parse_args()

    if args.all:
        summaries: dict[str, JsonDict] = {}

        for config_path in DEFAULT_CONFIG_PATHS:
            summary = train_from_config_path(
                config_path,
                overwrite=args.overwrite,
            )
            model_name = str(summary["name"])
            summaries[model_name] = summary

        summary_path = PROJECT_ROOT / "runs" / "train_summary.json"
        save_json({"by_model": summaries}, summary_path) # type: ignore

        print(f"\n[train] all done summary={summary_path}")
        return

    train_from_config_path(
        args.config,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()