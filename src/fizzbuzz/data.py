from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset

from fizzbuzz.config import DataConfig


CLASS_NAMES: tuple[str, str, str, str] = (
    "Number",
    "Fizz",
    "Buzz",
    "FizzBuzz",
)

NUMBER_LABEL = 0
FIZZ_LABEL = 1
BUZZ_LABEL = 2
FIZZBUZZ_LABEL = 3


@dataclass(frozen=True)
class DigitBatch:
    """digit列バッチを表すコンテナです。

    Parameters
    ----------
    digits : torch.Tensor
        padding済みのdigit ID列です。形状は ``(batch_size, max_seq_len)`` です。
    lengths : torch.Tensor
        各サンプルの有効系列長です。形状は ``(batch_size,)`` です。
    labels : torch.Tensor
        FizzBuzzの正解ラベルです。形状は ``(batch_size,)`` です。

    Notes
    -----
    ``digits`` には0〜9の数字IDとpadding IDのみを含めます。
    モデル入力には剰余特徴量を含めません。
    """

    digits: torch.Tensor
    lengths: torch.Tensor
    labels: torch.Tensor


def fizzbuzz_label(n: int) -> int:
    """整数に対応するFizzBuzzラベルを返します。

    Parameters
    ----------
    n : int
        ラベルを計算する正の整数。

    Returns
    -------
    int
        FizzBuzzラベル。0=Number, 1=Fizz, 2=Buzz, 3=FizzBuzzです。

    Raises
    ------
    TypeError
        ``n`` が整数でない場合。
    ValueError
        ``n`` が正の整数でない場合。
    """
    if not isinstance(n, int):
        raise TypeError(f"n must be an int, got {type(n).__name__}.")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}.")

    if n % 15 == 0:
        return FIZZBUZZ_LABEL
    if n % 3 == 0:
        return FIZZ_LABEL
    if n % 5 == 0:
        return BUZZ_LABEL
    return NUMBER_LABEL


def int_to_digits(n: int) -> list[int]:
    """整数を10進digit ID列に変換します。

    Parameters
    ----------
    n : int
        変換する正の整数。

    Returns
    -------
    list[int]
        10進表記の各桁を左から並べたdigit ID列。

    Raises
    ------
    TypeError
        ``n`` が整数でない場合。
    ValueError
        ``n`` が正の整数でない場合。
    """
    if not isinstance(n, int):
        raise TypeError(f"n must be an int, got {type(n).__name__}.")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}.")

    return [int(char) for char in str(n)]


def validate_digit_sequence(sequence: Sequence[int]) -> None:
    """digit ID列の妥当性を検証します。

    Parameters
    ----------
    sequence : Sequence[int]
        検証対象のdigit ID列。

    Returns
    -------
    None

    Raises
    ------
    ValueError
        ``sequence`` が空、または0〜9以外の値を含む場合。
    TypeError
        digit IDが整数でない場合。
    """
    if len(sequence) == 0:
        raise ValueError("digit sequence must not be empty.")

    for digit in sequence:
        if not isinstance(digit, int):
            raise TypeError(
                f"digit must be an int, got {type(digit).__name__}."
            )
        if digit < 0 or digit > 9:
            raise ValueError(f"digit must be in [0, 9], got {digit}.")


def pad_digit_sequences(
    sequences: Sequence[Sequence[int]],
    *,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """digit ID列をpaddingしてテンソル化します。

    Parameters
    ----------
    sequences : Sequence[Sequence[int]]
        padding対象のdigit ID列群。
    pad_token_id : int
        paddingに用いるトークンID。

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        padding済みdigitテンソルと系列長テンソルのタプルです。

    Raises
    ------
    ValueError
        ``sequences`` が空の場合、または ``pad_token_id`` がdigit IDと衝突する場合。
    TypeError
        ``pad_token_id`` が整数でない場合。
    """
    if len(sequences) == 0:
        raise ValueError("sequences must not be empty.")

    if not isinstance(pad_token_id, int):
        raise TypeError(
            f"pad_token_id must be an int, got {type(pad_token_id).__name__}."
        )
    if pad_token_id <= 9:
        raise ValueError(
            "pad_token_id must be greater than 9 to avoid collision with digits, "
            f"got {pad_token_id}."
        )

    lengths_list: list[int] = []
    for sequence in sequences:
        validate_digit_sequence(sequence)
        lengths_list.append(len(sequence))

    batch_size = len(sequences)
    max_length = max(lengths_list)

    digits = torch.full(
        size=(batch_size, max_length),
        fill_value=pad_token_id,
        dtype=torch.long,
    )

    for row_idx, sequence in enumerate(sequences):
        digits[row_idx, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)

    lengths = torch.tensor(lengths_list, dtype=torch.long)
    return digits, lengths


class FizzBuzzDataset(Dataset[tuple[list[int], int]]):
    """FizzBuzz分類用のDatasetです。

    Parameters
    ----------
    start : int
        整数範囲の開始値。
    end : int
        整数範囲の終了値。
    max_digits : int | None, default=None
        許容する最大桁数。Noneの場合は検証しません。

    Notes
    -----
    整数列そのものは保持せず、indexから整数を復元します。
    これにより、学習データ範囲が多少大きくなってもメモリ使用量を抑えられます。
    """

    def __init__(
        self,
        *,
        start: int,
        end: int,
        max_digits: int | None = None,
    ) -> None:
        if not isinstance(start, int):
            raise TypeError(f"start must be an int, got {type(start).__name__}.")
        if not isinstance(end, int):
            raise TypeError(f"end must be an int, got {type(end).__name__}.")

        if start <= 0:
            raise ValueError(f"start must be positive, got {start}.")
        if end <= 0:
            raise ValueError(f"end must be positive, got {end}.")
        if start > end:
            raise ValueError(
                f"start must be less than or equal to end, got {start} > {end}."
            )

        if max_digits is not None:
            if max_digits <= 0:
                raise ValueError(f"max_digits must be positive, got {max_digits}.")
            if len(str(end)) > max_digits:
                raise ValueError(
                    "end exceeds max_digits, "
                    f"got end={end}, max_digits={max_digits}."
                )

        self.start = start
        self.end = end
        self.max_digits = max_digits

    def __len__(self) -> int:
        """Datasetのサンプル数を返します。

        Returns
        -------
        int
            サンプル数。
        """
        return self.end - self.start + 1

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        """指定indexのdigit列とラベルを返します。

        Parameters
        ----------
        index : int
            取得するサンプルのindex。

        Returns
        -------
        tuple[list[int], int]
            digit ID列とFizzBuzzラベル。

        Raises
        ------
        IndexError
            ``index`` が範囲外の場合。
        TypeError
            ``index`` が整数でない場合。
        """
        if not isinstance(index, int):
            raise TypeError(f"index must be an int, got {type(index).__name__}.")
        if index < 0 or index >= len(self):
            raise IndexError(f"index out of range: {index}")

        n = self.start + index
        return int_to_digits(n), fizzbuzz_label(n)


@dataclass(frozen=True)
class DigitCollator:
    """DataLoader用のcollate callableです。

    Parameters
    ----------
    pad_token_id : int
        paddingに用いるトークンID。

    Notes
    -----
    Windows の multiprocessing DataLoader では、collate関数がpickle可能である必要があります。
    そのため、ローカル関数ではなくトップレベルのcallable classとして定義します。
    """

    pad_token_id: int

    def __post_init__(self) -> None:
        """初期化後にpadding token IDを検証します。

        Raises
        ------
        TypeError
            ``pad_token_id`` が整数でない場合。
        ValueError
            ``pad_token_id`` がdigit IDと衝突する場合。
        """
        if not isinstance(self.pad_token_id, int):
            raise TypeError(
                "pad_token_id must be an int, "
                f"got {type(self.pad_token_id).__name__}."
            )
        if self.pad_token_id <= 9:
            raise ValueError(
                "pad_token_id must be greater than 9 to avoid collision with digits, "
                f"got {self.pad_token_id}."
            )

    def __call__(self, batch: Sequence[tuple[list[int], int]]) -> DigitBatch:
        """サンプル列をDigitBatchへ変換します。

        Parameters
        ----------
        batch : Sequence[tuple[list[int], int]]
            Datasetから取得された ``(digit列, label)`` の列。

        Returns
        -------
        DigitBatch
            padding済みdigit列、系列長、ラベルを持つbatch。

        Raises
        ------
        ValueError
            ``batch`` が空、またはラベル範囲が不正な場合。
        """
        if len(batch) == 0:
            raise ValueError("batch must not be empty.")

        sequences = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        for label in labels:
            if label < 0 or label >= len(CLASS_NAMES):
                raise ValueError(
                    f"label must be in [0, {len(CLASS_NAMES) - 1}], got {label}."
                )

        digits, lengths = pad_digit_sequences(
            sequences,
            pad_token_id=self.pad_token_id,
        )
        label_tensor = torch.tensor(labels, dtype=torch.long)

        return DigitBatch(
            digits=digits,
            lengths=lengths,
            labels=label_tensor,
        )


def make_collate_fn(*, pad_token_id: int) -> DigitCollator:
    """DataLoader用のcollate callableを作成します。

    Parameters
    ----------
    pad_token_id : int
        paddingに用いるトークンID。

    Returns
    -------
    DigitCollator
        DataLoaderに渡すpickle可能なcollate callable。
    """
    return DigitCollator(pad_token_id=pad_token_id)


def build_train_dataset(config: DataConfig) -> FizzBuzzDataset:
    """学習用Datasetを構築します。

    Parameters
    ----------
    config : DataConfig
        データ設定。

    Returns
    -------
    FizzBuzzDataset
        学習用Dataset。
    """
    return FizzBuzzDataset(
        start=config.train_start,
        end=config.train_end,
        max_digits=config.max_digits,
    )


def build_train_loader(
    config: DataConfig,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    generator: torch.Generator | None = None,
) -> DataLoader[DigitBatch]:
    """学習用DataLoaderを構築します。

    Parameters
    ----------
    config : DataConfig
        データ設定。
    batch_size : int
        ミニバッチサイズ。
    num_workers : int
        DataLoaderのworker数。
    shuffle : bool, default=True
        サンプル順をシャッフルするかどうか。
    generator : torch.Generator | None, default=None
        shuffleに用いる乱数生成器。

    Returns
    -------
    DataLoader[DigitBatch]
        学習用DataLoader。

    Raises
    ------
    ValueError
        ``batch_size`` が正でない場合、または ``num_workers`` が負の場合。
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {num_workers}.")

    dataset = build_train_dataset(config)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=make_collate_fn(pad_token_id=config.pad_token_id),
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    ) # type: ignore


def iter_integer_batches(
    *,
    start: int,
    end: int,
    batch_size: int,
) -> Iterator[range]:
    """整数範囲をbatch単位のrangeとして逐次生成します。

    Parameters
    ----------
    start : int
        整数範囲の開始値。
    end : int
        整数範囲の終了値。
    batch_size : int
        1 batchあたりの整数数。

    Yields
    ------
    range
        batch単位の整数range。

    Raises
    ------
    ValueError
        範囲指定またはbatch sizeが不正な場合。
    """
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

    for batch_start in range(start, end + 1, batch_size):
        batch_end = min(batch_start + batch_size - 1, end)
        yield range(batch_start, batch_end + 1)


def make_batch_from_numbers(
    numbers: Sequence[int],
    *,
    pad_token_id: int,
) -> DigitBatch:
    """整数列から評価用DigitBatchを作成します。

    Parameters
    ----------
    numbers : Sequence[int]
        batch化する正の整数列。
    pad_token_id : int
        paddingに用いるトークンID。

    Returns
    -------
    DigitBatch
        digit ID列、系列長、ラベルを持つバッチ。

    Raises
    ------
    ValueError
        ``numbers`` が空の場合。
    """
    if len(numbers) == 0:
        raise ValueError("numbers must not be empty.")

    sequences = [int_to_digits(n) for n in numbers]
    labels = [fizzbuzz_label(n) for n in numbers]

    digits, lengths = pad_digit_sequences(
        sequences,
        pad_token_id=pad_token_id,
    )

    return DigitBatch(
        digits=digits,
        lengths=lengths,
        labels=torch.tensor(labels, dtype=torch.long),
    )


def make_batch_from_range(
    numbers: range,
    *,
    pad_token_id: int,
) -> DigitBatch:
    """整数rangeから評価用DigitBatchを作成します。

    Parameters
    ----------
    numbers : range
        batch化する整数range。
    pad_token_id : int
        paddingに用いるトークンID。

    Returns
    -------
    DigitBatch
        digit ID列、系列長、ラベルを持つバッチ。

    Raises
    ------
    ValueError
        ``numbers`` が空の場合。
    """
    if len(numbers) == 0:
        raise ValueError("numbers must not be empty.")

    return make_batch_from_numbers(
        list(numbers),
        pad_token_id=pad_token_id,
    )