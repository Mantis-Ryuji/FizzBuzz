from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
import warnings
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


def _as_int(value: object, *, name: str) -> int:
    """objectを厳密にintへ変換します。

    Parameters
    ----------
    value : object
        変換対象の値です。
    name : str
        エラーメッセージに使う値の名前です。

    Returns
    -------
    int
        変換後の整数です。

    Raises
    ------
    ValueError
        value が int でない場合。
    """
    if type(value) is not int:
        raise ValueError(f"{name} must be int, got {type(value).__name__}.")
    return value


def _as_float(value: object, *, name: str) -> float:
    """objectをfloatへ変換します。

    Parameters
    ----------
    value : object
        変換対象の値です。
    name : str
        エラーメッセージに使う値の名前です。

    Returns
    -------
    float
        変換後の浮動小数点数です。

    Raises
    ------
    ValueError
        value が数値でない場合。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric, got {type(value).__name__}.")
    return float(value)


def _to_cpu_byte_tensor_or_none(
    value: object,
    *,
    name: str,
) -> torch.Tensor | None:
    """RNG stateをCPU上のByteTensorへ正規化します。

    Parameters
    ----------
    value : object
        正規化対象のRNG stateです。
    name : str
        warning messageに使う名前です。

    Returns
    -------
    torch.Tensor | None
        CPU上の1次元ByteTensorです。復元できない場合はNoneです。
    """
    if not isinstance(value, torch.Tensor):
        warnings.warn(
            f"Skip restoring {name}: expected torch.Tensor, "
            f"got {type(value).__name__}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    tensor = value.detach().to(device="cpu", dtype=torch.uint8).contiguous()

    if tensor.ndim != 1:
        warnings.warn(
            f"Skip restoring {name}: expected 1D RNG state tensor, "
            f"got shape={tuple(tensor.shape)}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    return tensor


def _restore_rng_state_from_checkpoint(rng_state: object) -> None:
    """checkpoint内のRNG stateを可能な範囲で復元します。

    Parameters
    ----------
    rng_state : object
        checkpointに保存された ``rng_state`` です。

    Returns
    -------
    None

    Notes
    -----
    CUDA RNG stateはPyTorchのversionや ``map_location`` によって
    GPU tensorとして復元されることがあります。``torch.cuda.set_rng_state_all`` は
    CPU上のByteTensorを要求するため、ここでCPU ByteTensorへ正規化します。
    RNG stateは再現性用の補助情報なので、形式が不正な場合でもmodel/optimizerの
    resume自体は止めずにwarningを出してスキップします。
    """
    if not isinstance(rng_state, Mapping):
        warnings.warn(
            f"Skip restoring RNG state: expected mapping, got {type(rng_state).__name__}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    torch_rng_state = rng_state.get("torch")
    if torch_rng_state is not None:
        torch_state = _to_cpu_byte_tensor_or_none(
            torch_rng_state,
            name="rng_state['torch']",
        )
        if torch_state is not None:
            torch.set_rng_state(torch_state)

    cuda_rng_state = rng_state.get("cuda")
    if not torch.cuda.is_available() or cuda_rng_state is None:
        return

    if isinstance(cuda_rng_state, torch.Tensor):
        raw_cuda_states: list[object] = [cuda_rng_state]
    elif isinstance(cuda_rng_state, (list, tuple)):
        raw_cuda_states = list(cuda_rng_state)
    else:
        warnings.warn(
            "Skip restoring rng_state['cuda']: expected tensor, list, or tuple, "
            f"got {type(cuda_rng_state).__name__}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    cuda_states: list[torch.Tensor] = []
    for index, raw_state in enumerate(raw_cuda_states):
        cuda_state = _to_cpu_byte_tensor_or_none(
            raw_state,
            name=f"rng_state['cuda'][{index}]",
        )
        if cuda_state is not None:
            cuda_states.append(cuda_state)

    if len(cuda_states) == 0:
        return

    torch.cuda.set_rng_state_all(cuda_states[: torch.cuda.device_count()])


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

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> TrainEpochResult:
        """辞書から1 epoch分の学習結果を復元します。

        Parameters
        ----------
        data : collections.abc.Mapping[str, object]
            ``to_dict`` で保存された辞書です。

        Returns
        -------
        TrainEpochResult
            復元されたepoch結果。

        Raises
        ------
        KeyError
            必要なキーが存在しない場合。
        ValueError
            値の型が不正な場合。
        """
        required_keys = {"epoch", "loss", "accuracy", "lr", "elapsed_sec"}
        missing_keys = required_keys - set(data.keys())
        if len(missing_keys) > 0:
            raise KeyError(f"epoch result is missing keys: {sorted(missing_keys)}")

        epoch = _as_int(data["epoch"], name="epoch")
        if epoch <= 0:
            raise ValueError(f"epoch must be positive, got {epoch}.")

        return cls(
            epoch=epoch,
            loss=_as_float(data["loss"], name="loss"),
            accuracy=_as_float(data["accuracy"], name="accuracy"),
            lr=_as_float(data["lr"], name="lr"),
            elapsed_sec=_as_float(data["elapsed_sec"], name="elapsed_sec"),
        )


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

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> TrainHistory:
        """辞書から学習履歴を復元します。

        Parameters
        ----------
        data : collections.abc.Mapping[str, object]
            ``to_dict`` で保存された辞書です。

        Returns
        -------
        TrainHistory
            復元された学習履歴。

        Raises
        ------
        KeyError
            ``records`` キーが存在しない場合。
        ValueError
            ``records`` の形式が不正な場合。
        """
        if "records" not in data:
            raise KeyError("history is missing key: 'records'")

        records_obj = data["records"]
        if not isinstance(records_obj, list):
            raise ValueError(
                f"history records must be list, got {type(records_obj).__name__}."
            )

        records: list[TrainEpochResult] = []
        for index, record_obj in enumerate(records_obj):
            if not isinstance(record_obj, Mapping):
                raise ValueError(
                    f"history record at index {index} must be a mapping, "
                    f"got {type(record_obj).__name__}."
                )
            records.append(TrainEpochResult.from_dict(record_obj))

        return cls(records=records)


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
        resume: bool = False,
        checkpoint_filename: str = "last.pt",
        checkpoint_every_epochs: int = 100,
        milestone_epochs: tuple[int, ...] = (),
        milestone_dirname: str = "milestones",
    ) -> TrainHistory:
        """モデルを固定epochで学習します。

        Parameters
        ----------
        train_loader : DataLoader[DigitBatch]
            学習用DataLoaderです。
        save_model : bool, default=True
            Trueの場合、学習終了後に最終重みを保存します。
        model_filename : str, default="model.pt"
            保存する最終重みファイル名です。
        history_filename : str, default="history.json"
            保存する学習履歴ファイル名です。
        resume : bool, default=False
            Trueの場合、``checkpoint_filename`` が存在すれば学習状態を復元します。
        checkpoint_filename : str, default="last.pt"
            resume用の完全checkpointファイル名です。
        checkpoint_every_epochs : int, default=100
            完全checkpointを保存するepoch間隔です。
        milestone_epochs : tuple[int, ...], default=()
            評価用の重みsnapshotを保存するepoch番号です。
        milestone_dirname : str, default="milestones"
            評価用の重みsnapshotを保存するディレクトリ名です。

        Returns
        -------
        TrainHistory
            学習履歴。

        Raises
        ------
        ValueError
            ファイル名、保存間隔、または milestone epoch が不正な場合。
        """
        if model_filename.strip() == "":
            raise ValueError("model_filename must not be empty.")
        if history_filename.strip() == "":
            raise ValueError("history_filename must not be empty.")
        if checkpoint_filename.strip() == "":
            raise ValueError("checkpoint_filename must not be empty.")
        if milestone_dirname.strip() == "":
            raise ValueError("milestone_dirname must not be empty.")
        if checkpoint_every_epochs <= 0:
            raise ValueError(
                "checkpoint_every_epochs must be positive, "
                f"got {checkpoint_every_epochs}."
            )
        if any(epoch <= 0 for epoch in milestone_epochs):
            raise ValueError(
                "milestone_epochs must contain positive integers, "
                f"got {milestone_epochs}."
            )

        checkpoint_path = self.output_dir / checkpoint_filename
        history_path = self.output_dir / history_filename
        milestone_dir = self.output_dir / milestone_dirname
        milestone_set = set(milestone_epochs)

        start_epoch = 1
        if resume and checkpoint_path.exists():
            completed_epoch = self.load_checkpoint(checkpoint_path)
            start_epoch = completed_epoch + 1

        if len(milestone_set) > 0:
            milestone_dir.mkdir(parents=True, exist_ok=True)

        if start_epoch > self.config.epochs:
            save_json(self.history.to_dict(), history_path)
            if save_model:
                self.save_model(self.output_dir / model_filename)
            return self.history

        epoch_bar = tqdm(
            range(start_epoch, self.config.epochs + 1),
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

            if epoch in milestone_set:
                self.save_model(milestone_dir / f"epoch_{epoch:06d}.pt")

            should_save_checkpoint = (
                epoch % checkpoint_every_epochs == 0
                or epoch == self.config.epochs
            )
            if should_save_checkpoint:
                self.save_checkpoint(checkpoint_path, epoch=epoch)
                save_json(self.history.to_dict(), history_path)

        save_json(self.history.to_dict(), history_path)

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
            "version": 1,
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history.to_dict(),
            "config": self.config.model_dump(mode="json"),
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
            },
        }

        tmp_path = save_path.with_name(f"{save_path.name}.tmp")
        torch.save(checkpoint, tmp_path)
        tmp_path.replace(save_path)

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
            checkpoint内の値が不正な場合。
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
        if type(epoch) is not int:
            raise ValueError(f"checkpoint epoch must be int, got {type(epoch).__name__}.")
        if epoch < 0:
            raise ValueError(f"checkpoint epoch must be non-negative, got {epoch}.")

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "history" in checkpoint:
            history_obj = checkpoint["history"]
            if not isinstance(history_obj, Mapping):
                raise ValueError(
                    f"checkpoint history must be a mapping, "
                    f"got {type(history_obj).__name__}."
                )
            self.history = TrainHistory.from_dict(history_obj)

        if "rng_state" in checkpoint:
            _restore_rng_state_from_checkpoint(checkpoint["rng_state"])

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