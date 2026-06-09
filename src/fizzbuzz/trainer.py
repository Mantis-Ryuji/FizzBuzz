from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import TypeAlias

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from fizzbuzz.config import TrainingConfig
from fizzbuzz.data import DigitBatch
from fizzbuzz.metrics import accuracy_from_logits
from fizzbuzz.utils import JsonDict, ensure_dir, save_json


CheckpointDict: TypeAlias = dict[str, object]


@dataclass(frozen=True)
class TrainEpochResult:
    """1 epoch分の学習結果です。

    Parameters
    ----------
    epoch : int
        epoch番号です。1始まりです。
    loss : float
        epoch平均のCrossEntropyLossです。
    accuracy : float
        epoch平均のAccuracyです。
    lr : float
        epoch終了時点の学習率です。
    elapsed_sec : float
        epochにかかった秒数です。
    """

    epoch: int
    loss: float
    accuracy: float
    lr: float
    elapsed_sec: float

    def to_dict(self) -> JsonDict:
        """JSON保存可能な辞書へ変換します。

        Returns
        -------
        dict[str, JsonValue]
            JSON保存可能な辞書。
        """
        return {
            "epoch": self.epoch,
            "loss": self.loss,
            "accuracy": self.accuracy,
            "lr": self.lr,
            "elapsed_sec": self.elapsed_sec,
        }


@dataclass
class TrainHistory:
    """学習履歴です。

    Parameters
    ----------
    records : list[TrainEpochResult]
        epochごとの学習結果です。
    """

    records: list[TrainEpochResult] = field(default_factory=list)

    def append(self, result: TrainEpochResult) -> None:
        """学習結果を追加します。

        Parameters
        ----------
        result : TrainEpochResult
            追加するepoch結果。

        Returns
        -------
        None
        """
        self.records.append(result)

    def to_dict(self) -> JsonDict:
        """JSON保存可能な辞書へ変換します。

        Returns
        -------
        dict[str, JsonValue]
            JSON保存可能な辞書。
        """
        return {
            "records": [record.to_dict() for record in self.records],
            "epoch": [record.epoch for record in self.records],
            "loss": [record.loss for record in self.records],
            "accuracy": [record.accuracy for record in self.records],
            "lr": [record.lr for record in self.records],
            "elapsed_sec": [record.elapsed_sec for record in self.records],
        }


class Trainer:
    """FizzBuzz分類モデルの学習器です。

    Parameters
    ----------
    model : torch.nn.Module
        学習対象モデルです。
    config : TrainingConfig
        学習設定です。
    output_dir : str or pathlib.Path
        重みや履歴を保存するディレクトリです。
    criterion : torch.nn.Module | None, default=None
        損失関数です。Noneの場合はCrossEntropyLossを使います。

    Attributes
    ----------
    model : torch.nn.Module
        学習対象モデルです。
    config : TrainingConfig
        学習設定です。
    output_dir : pathlib.Path
        出力ディレクトリです。
    optimizer : torch.optim.Optimizer
        AdamW optimizerです。
    criterion : torch.nn.Module
        損失関数です。
    history : TrainHistory
        学習履歴です。

    Notes
    -----
    本実験では validation による checkpoint selection は行いません。
    固定epochで学習し、最後の重みを正式な学習済み重みとして保存します。
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        config: TrainingConfig,
        output_dir: str | Path,
        criterion: nn.Module | None = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be nn.Module, got {type(model).__name__}.")

        self.model = model
        self.config = config
        self.output_dir = ensure_dir(output_dir)
        self.device = resolve_device(config.device)

        self.model.to(self.device)

        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.criterion.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        self.history = TrainHistory()

    def fit(
        self,
        train_loader: DataLoader[DigitBatch],
        *,
        save_model: bool = True,
        model_filename: str = "model.pt",
        history_filename: str = "history.json",
    ) -> TrainHistory:
        """モデルを固定epochで学習します。

        Parameters
        ----------
        train_loader : DataLoader[DigitBatch]
            学習用DataLoaderです。
        save_model : bool, default=True
            Trueの場合、学習終了後に最終重みを保存します。
        model_filename : str, default="model.pt"
            保存する重みファイル名です。
        history_filename : str, default="history.json"
            保存する学習履歴ファイル名です。

        Returns
        -------
        TrainHistory
            学習履歴。

        Raises
        ------
        ValueError
            ファイル名が空の場合。
        """
        if model_filename.strip() == "":
            raise ValueError("model_filename must not be empty.")
        if history_filename.strip() == "":
            raise ValueError("history_filename must not be empty.")

        epoch_bar = tqdm(
            range(1, self.config.epochs + 1),
            desc="Training",
            unit="epoch",
            dynamic_ncols=True,
        )

        for epoch in epoch_bar:
            result = self.train_one_epoch(
                train_loader=train_loader,
                epoch=epoch,
            )
            self.history.append(result)

            epoch_bar.set_postfix(
                {
                    "loss": f"{result.loss:.4f}",
                    "acc": f"{result.accuracy:.4f}",
                    "lr": f"{result.lr:.2e}",
                }
            )

        save_json(self.history.to_dict(), self.output_dir / history_filename)

        if save_model:
            self.save_model(self.output_dir / model_filename)

        return self.history

    def train_one_epoch(
        self,
        *,
        train_loader: DataLoader[DigitBatch],
        epoch: int,
    ) -> TrainEpochResult:
        """1 epoch分の学習を実行します。

        Parameters
        ----------
        train_loader : DataLoader[DigitBatch]
            学習用DataLoaderです。
        epoch : int
            epoch番号です。1始まりです。

        Returns
        -------
        TrainEpochResult
            1 epoch分の学習結果。

        Raises
        ------
        ValueError
            ``epoch`` が正でない場合。
        """
        if epoch <= 0:
            raise ValueError(f"epoch must be positive, got {epoch}.")

        self.model.train()

        start_time = perf_counter()
        total_loss = 0.0
        total_correct = 0.0
        total_samples = 0

        batch_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:03d}",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )

        for batch in batch_bar:
            batch_on_device = move_batch_to_device(batch, device=self.device)

            self.optimizer.zero_grad(set_to_none=True)

            logits = self.model(
                batch_on_device.digits,
                batch_on_device.lengths,
            )
            loss = self.criterion(logits, batch_on_device.labels)

            loss.backward()

            if self.config.grad_clip_norm > 0.0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.grad_clip_norm,
                )

            self.optimizer.step()

            batch_size = batch_on_device.labels.size(0)
            batch_accuracy = accuracy_from_logits(
                logits.detach(),
                batch_on_device.labels,
            )

            total_loss += float(loss.item()) * batch_size
            total_correct += batch_accuracy * batch_size
            total_samples += batch_size

            running_loss = total_loss / total_samples
            running_accuracy = total_correct / total_samples

            batch_bar.set_postfix(
                {
                    "loss": f"{running_loss:.4f}",
                    "acc": f"{running_accuracy:.4f}",
                }
            )

        if total_samples <= 0:
            raise ValueError("train_loader produced no samples.")

        elapsed_sec = perf_counter() - start_time

        return TrainEpochResult(
            epoch=epoch,
            loss=total_loss / total_samples,
            accuracy=total_correct / total_samples,
            lr=self.get_current_lr(),
            elapsed_sec=elapsed_sec,
        )

    def save_model(self, path: str | Path) -> None:
        """モデル重みを保存します。

        Parameters
        ----------
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
            保存に失敗した場合。
        """
        save_path = Path(path)

        if str(save_path).strip() == "":
            raise ValueError("path must not be empty.")

        if save_path.parent != Path("."):
            save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), save_path)

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        epoch: int,
    ) -> None:
        """完全checkpointを保存します。

        Parameters
        ----------
        path : str or pathlib.Path
            保存先パス。
        epoch : int
            保存時点のepoch番号。

        Returns
        -------
        None

        Raises
        ------
        ValueError
            ``epoch`` が負の場合、または保存先パスが空の場合。
        OSError
            保存に失敗した場合。
        """
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}.")

        save_path = Path(path)

        if str(save_path).strip() == "":
            raise ValueError("path must not be empty.")

        if save_path.parent != Path("."):
            save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint: CheckpointDict = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history.to_dict(),
            "config": self.config.model_dump(mode="json"),
        }

        torch.save(checkpoint, save_path)

    def load_checkpoint(self, path: str | Path) -> int:
        """完全checkpointを読み込みます。

        Parameters
        ----------
        path : str or pathlib.Path
            読み込むcheckpointのパス。

        Returns
        -------
        int
            checkpointに保存されていたepoch番号。

        Raises
        ------
        FileNotFoundError
            checkpointファイルが存在しない場合。
        KeyError
            checkpointに必要なキーが存在しない場合。
        ValueError
            checkpoint内のepochが不正な場合。
        RuntimeError
            checkpointの読み込みに失敗した場合。
        """
        load_path = Path(path)

        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")

        checkpoint = torch.load(
            load_path,
            map_location=self.device,
            weights_only=False,
        )

        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint must be a dictionary.")

        required_keys = {
            "epoch",
            "model_state_dict",
            "optimizer_state_dict",
        }
        missing_keys = required_keys - set(checkpoint.keys())
        if len(missing_keys) > 0:
            raise KeyError(f"checkpoint is missing keys: {sorted(missing_keys)}")

        epoch = checkpoint["epoch"]
        if not isinstance(epoch, int):
            raise ValueError(f"checkpoint epoch must be int, got {type(epoch).__name__}.")
        if epoch < 0:
            raise ValueError(f"checkpoint epoch must be non-negative, got {epoch}.")

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return epoch

    def get_current_lr(self) -> float:
        """現在の学習率を返します。

        Returns
        -------
        float
            現在の学習率。
        """
        return float(self.optimizer.param_groups[0]["lr"])


def move_batch_to_device(
    batch: DigitBatch,
    *,
    device: torch.device | str,
) -> DigitBatch:
    """DigitBatchを指定デバイスへ転送します。

    Parameters
    ----------
    batch : DigitBatch
        転送対象のバッチです。
    device : torch.device or str
        転送先デバイスです。

    Returns
    -------
    DigitBatch
        指定デバイスに転送されたバッチ。

    Raises
    ------
    TypeError
        ``batch`` が ``DigitBatch`` でない場合。
    RuntimeError
        指定デバイスが利用できない場合。
    """
    if not isinstance(batch, DigitBatch):
        raise TypeError(f"batch must be DigitBatch, got {type(batch).__name__}.")

    device_obj = torch.device(device)

    return DigitBatch(
        digits=batch.digits.to(device_obj, non_blocking=True),
        lengths=batch.lengths.to(device_obj, non_blocking=True),
        labels=batch.labels.to(device_obj, non_blocking=True),
    )


def resolve_device(device: str | torch.device) -> torch.device:
    """デバイス指定を解決します。

    Parameters
    ----------
    device : str or torch.device
        デバイス指定です。``"auto"``, ``"cpu"``, ``"cuda"``, ``"mps"`` を想定します。

    Returns
    -------
    torch.device
        解決されたデバイス。

    Raises
    ------
    ValueError
        未対応のデバイス指定の場合。
    RuntimeError
        指定されたデバイスが利用できない場合。
    """
    if isinstance(device, torch.device):
        return device

    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError(
            f"device must be one of 'auto', 'cpu', 'cuda', or 'mps', got {device!r}."
        )

    device_obj = torch.device(device)

    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device was requested, but CUDA is not available.")

    if device_obj.type == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not torch.backends.mps.is_available():
            raise RuntimeError("MPS device was requested, but MPS is not available.")

    return device_obj