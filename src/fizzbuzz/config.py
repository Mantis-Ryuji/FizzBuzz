from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


DeviceName = Literal["auto", "cpu", "cuda", "mps"]


class DataConfig(BaseModel):
    """データ生成に関する設定です。

    Parameters
    ----------
    train_start : int, default=1
        学習に使う整数範囲の開始値。
    train_end : int, default=99999
        学習に使う整数範囲の終了値。
    max_digits : int, default=5
        学習データで想定する最大桁数。
    pad_token_id : int, default=10
        digit列をpaddingするときに使うトークンID。
    num_classes : int, default=4
        FizzBuzz分類のクラス数。

    Notes
    -----
    入力特徴量として剰余は使わず、整数を10進数字列に変換して扱います。
    """

    model_config = ConfigDict(extra="forbid")

    train_start: int = Field(default=1, ge=1)
    train_end: int = Field(default=99_999, ge=1)
    max_digits: int = Field(default=5, ge=1)
    pad_token_id: int = Field(default=10, ge=0)
    num_classes: int = Field(default=4, ge=2)

    @model_validator(mode="after")
    def validate_train_range(self) -> DataConfig:
        """学習範囲の整合性を検証します。"""
        if self.train_start > self.train_end:
            raise ValueError(
                "train_start must be less than or equal to train_end, "
                f"got train_start={self.train_start}, train_end={self.train_end}."
            )

        if len(str(self.train_end)) > self.max_digits:
            raise ValueError(
                "train_end exceeds max_digits, "
                f"got train_end={self.train_end}, max_digits={self.max_digits}."
            )

        if self.num_classes != 4:
            raise ValueError(
                "FizzBuzz classification must have exactly 4 classes, "
                f"got num_classes={self.num_classes}."
            )

        return self


class ModelConfig(BaseModel):
    """モデル構造に関する設定です。

    Parameters
    ----------
    vocab_size : int, default=11
        digit ID の語彙サイズ。0〜9の数字とpadding tokenを含めます。
    embedding_dim : int
        digit embedding の次元数。
    hidden_dim : int
        GRU の隠れ状態次元数。
    num_layers : int, default=1
        GRU の層数。
    dropout : float, default=0.0
        GRU層間のdropout率。
    bidirectional : bool, default=False
        双方向GRUを使うかどうか。
    padding_idx : int, default=10
        embedding層でpaddingとして扱うID。

    Notes
    -----
    本実験では Digit Embedding + GRU + Linear classifier を基本モデルとします。
    """

    model_config = ConfigDict(extra="forbid")

    vocab_size: int = Field(default=11, ge=2)
    embedding_dim: int = Field(gt=0)
    hidden_dim: int = Field(gt=0)
    num_layers: int = Field(default=1, ge=1)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    bidirectional: bool = False
    padding_idx: int = Field(default=10, ge=0)

    @model_validator(mode="after")
    def validate_token_ids(self) -> ModelConfig:
        """語彙サイズとpadding IDの整合性を検証します。"""
        if self.padding_idx >= self.vocab_size:
            raise ValueError(
                "padding_idx must be smaller than vocab_size, "
                f"got padding_idx={self.padding_idx}, vocab_size={self.vocab_size}."
            )

        return self


class TrainingConfig(BaseModel):
    """学習に関する設定です。

    Parameters
    ----------
    epochs : int
        学習エポック数。
    batch_size : int
        学習時のミニバッチサイズ。
    lr : float
        AdamW の学習率。
    weight_decay : float, default=0.0
        AdamW の weight decay。
    num_workers : int, default=0
        DataLoader の worker 数。
    grad_clip_norm : float, default=1.0
        勾配クリッピングの最大ノルム。
    device : {"auto", "cpu", "cuda", "mps"}, default="auto"
        学習に使うデバイス。
    """

    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    lr: float = Field(gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    num_workers: int = Field(default=0, ge=0)
    grad_clip_norm: float = Field(default=1.0, gt=0.0)
    device: DeviceName = "auto"


class EvalConfig(BaseModel):
    """外挿評価に関する設定です。

    Parameters
    ----------
    batch_size : int
        評価時のミニバッチサイズ。
    digit_ranges : dict[str, tuple[int, int]]
        評価対象の整数範囲。キーは評価名、値は開始値と終了値です。

    Notes
    -----
    巨大な評価データは全件保持せず、batch単位でstreaming評価します。
    """

    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(gt=0)
    digit_ranges: dict[str, tuple[int, int]] = Field(
        default_factory=lambda: {
            "test_6digit": (100_000, 999_999),
            "test_7digit": (1_000_000, 9_999_999),
            "test_8digit": (10_000_000, 99_999_999),
        }
    )

    @field_validator("digit_ranges")
    @classmethod
    def validate_digit_ranges(
        cls,
        value: dict[str, tuple[int, int]],
    ) -> dict[str, tuple[int, int]]:
        """評価範囲の形式と大小関係を検証します。"""
        if len(value) == 0:
            raise ValueError("digit_ranges must not be empty.")

        for name, range_pair in value.items():
            if name.strip() == "":
                raise ValueError("digit range name must not be empty.")

            start, end = range_pair
            if start <= 0:
                raise ValueError(
                    f"range start must be positive for {name}, got {start}."
                )
            if start > end:
                raise ValueError(
                    f"range start must be less than or equal to end for {name}, "
                    f"got start={start}, end={end}."
                )

        return value


class OutputConfig(BaseModel):
    """出力先に関する設定です。

    Parameters
    ----------
    weight_dir : pathlib.Path, default="runs/weights"
        学習済み重みや評価結果を保存するディレクトリ。
    image_dir : pathlib.Path, default="runs/images"
        可視化画像を保存するディレクトリ。
    """

    model_config = ConfigDict(extra="forbid")

    weight_dir: Path = Path("runs/weights")
    image_dir: Path = Path("runs/images")

    @field_validator("weight_dir", "image_dir")
    @classmethod
    def validate_path(cls, value: Path) -> Path:
        """空のパスを禁止します。"""
        if str(value).strip() == "":
            raise ValueError("output path must not be empty.")

        return value


class ExperimentConfig(BaseModel):
    """FizzBuzz外挿実験全体の設定です。

    Parameters
    ----------
    name : str
        実験名。通常は small, medium, large のいずれかです。
    seed : int
        乱数シード。
    data : DataConfig
        データ生成設定。
    model : ModelConfig
        モデル設定。
    training : TrainingConfig
        学習設定。
    eval : EvalConfig
        外挿評価設定。
    output : OutputConfig
        出力先設定。
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    seed: int = Field(default=42, ge=0)
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    eval: EvalConfig
    output: OutputConfig = Field(default_factory=OutputConfig)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """実験名を検証します。"""
        cleaned = value.strip()

        if cleaned == "":
            raise ValueError("name must not be empty.")

        if re.fullmatch(r"[A-Za-z0-9_-]+", cleaned) is None:
            raise ValueError(
                "name must contain only letters, digits, underscores, or hyphens, "
                f"got {value!r}."
            )

        return cleaned

    @model_validator(mode="after")
    def validate_cross_section_consistency(self) -> ExperimentConfig:
        """セクション間の整合性を検証します。"""
        if self.data.pad_token_id != self.model.padding_idx:
            raise ValueError(
                "data.pad_token_id and model.padding_idx must be the same, "
                f"got {self.data.pad_token_id} and {self.model.padding_idx}."
            )

        if self.model.vocab_size <= self.data.pad_token_id:
            raise ValueError(
                "model.vocab_size must be greater than data.pad_token_id, "
                f"got vocab_size={self.model.vocab_size}, "
                f"pad_token_id={self.data.pad_token_id}."
            )

        for range_name, (start, _end) in self.eval.digit_ranges.items():
            if start <= self.data.train_end:
                raise ValueError(
                    "evaluation ranges must start after the training range, "
                    f"but {range_name} starts at {start} and "
                    f"train_end={self.data.train_end}."
                )

        return self


def load_config(path: str | Path) -> ExperimentConfig:
    """YAML設定ファイルを読み込み、検証済み設定を返します。

    Parameters
    ----------
    path : str or pathlib.Path
        読み込むYAMLファイルのパス。

    Returns
    -------
    ExperimentConfig
        検証済みの実験設定。

    Raises
    ------
    FileNotFoundError
        指定された設定ファイルが存在しない場合。
    ValueError
        YAMLのトップレベルが辞書でない場合、またはキーが文字列でない場合。
    yaml.YAMLError
        YAMLとして不正な場合。
    pydantic.ValidationError
        設定値がスキーマに適合しない場合。
    """
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")

    for key in raw:
        if not isinstance(key, str):
            raise ValueError(
                f"Config keys must be strings, got {type(key).__name__}."
            )

    raw_mapping = cast(dict[str, object], raw)
    return ExperimentConfig.model_validate(raw_mapping)


def config_to_dict(config: ExperimentConfig) -> dict[str, object]:
    """設定オブジェクトをJSON/YAML互換の辞書に変換します。

    Parameters
    ----------
    config : ExperimentConfig
        変換する実験設定。

    Returns
    -------
    dict[str, object]
        JSON/YAML互換の辞書。
    """
    return cast(dict[str, object], config.model_dump(mode="json"))


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    """検証済み設定をYAMLとして保存します。

    Parameters
    ----------
    config : ExperimentConfig
        保存する実験設定。
    path : str or pathlib.Path
        保存先パス。

    Returns
    -------
    None

    Raises
    ------
    ValueError
        保存先パスが空の場合。
    OSError
        ファイル保存に失敗した場合。
    yaml.YAMLError
        YAMLへの変換に失敗した場合。
    """
    save_path = Path(path)

    if str(save_path).strip() == "":
        raise ValueError("path must not be empty.")

    if save_path.parent != Path("."):
        save_path.parent.mkdir(parents=True, exist_ok=True)

    with save_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            config_to_dict(config),
            f,
            allow_unicode=True,
            sort_keys=False,
        )