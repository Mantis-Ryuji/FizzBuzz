from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from fizzbuzz.config import ModelConfig


@dataclass(frozen=True)
class ModelOutput:
    """モデル出力を表すコンテナです。

    Parameters
    ----------
    logits : torch.Tensor
        4クラス分類のlogitsです。形状は ``(batch_size, num_classes)`` です。
    features : torch.Tensor
        最終時刻に対応する系列表現です。形状は ``(batch_size, feature_dim)`` です。

    Notes
    -----
    ``features`` は分類器に入力される直前の表現です。
    可視化や内部表現解析を行いたい場合に利用できます。
    """

    logits: torch.Tensor
    features: torch.Tensor


class DigitGRUClassifier(nn.Module):
    """10進digit列からFizzBuzzラベルを予測するGRU分類器です。

    Parameters
    ----------
    vocab_size : int
        digit ID の語彙サイズです。0〜9の数字とpadding tokenを含みます。
    embedding_dim : int
        digit embedding の次元数です。
    hidden_dim : int
        GRU の隠れ状態次元数です。
    num_layers : int, default=1
        GRU の層数です。
    dropout : float, default=0.0
        GRU層間のdropout率です。``num_layers=1`` の場合は内部的に0として扱います。
    bidirectional : bool, default=False
        双方向GRUを使うかどうかです。
    padding_idx : int, default=10
        embedding層でpaddingとして扱うトークンIDです。
    num_classes : int, default=4
        出力クラス数です。

    Attributes
    ----------
    embedding : torch.nn.Embedding
        digit IDを連続ベクトルへ変換する層です。
    gru : torch.nn.GRU
        digit系列を処理するGRUです。
    classifier : torch.nn.Linear
        GRU特徴量を4クラスlogitsへ変換する線形層です。

    Notes
    -----
    入力には剰余特徴量を含めません。
    モデルはdigit ID列だけを受け取り、系列表現からFizzBuzzラベルを予測します。
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        padding_idx: int = 10,
        num_classes: int = 4,
    ) -> None:
        super().__init__()

        self._validate_init_args(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            padding_idx=padding_idx,
            num_classes=num_classes,
        )

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.padding_idx = padding_idx
        self.num_classes = num_classes

        num_directions = 2 if bidirectional else 1
        effective_dropout = dropout if num_layers > 1 else 0.0

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
            bidirectional=bidirectional,
        )

        self.classifier = nn.Linear(
            in_features=hidden_dim * num_directions,
            out_features=num_classes,
        )

    @staticmethod
    def _validate_init_args(
        *,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        padding_idx: int,
        num_classes: int,
    ) -> None:
        """初期化引数を検証します。

        Parameters
        ----------
        vocab_size : int
            digit ID の語彙サイズです。
        embedding_dim : int
            digit embedding の次元数です。
        hidden_dim : int
            GRU の隠れ状態次元数です。
        num_layers : int
            GRU の層数です。
        dropout : float
            GRU層間のdropout率です。
        padding_idx : int
            padding token IDです。
        num_classes : int
            出力クラス数です。

        Returns
        -------
        None

        Raises
        ------
        ValueError
            引数の範囲や整合性が不正な場合。
        """
        if vocab_size <= 1:
            raise ValueError(f"vocab_size must be greater than 1, got {vocab_size}.")
        if embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim must be positive, got {embedding_dim}."
            )
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")
        if padding_idx < 0:
            raise ValueError(f"padding_idx must be non-negative, got {padding_idx}.")
        if padding_idx >= vocab_size:
            raise ValueError(
                "padding_idx must be smaller than vocab_size, "
                f"got padding_idx={padding_idx}, vocab_size={vocab_size}."
            )
        if num_classes <= 1:
            raise ValueError(
                f"num_classes must be greater than 1, got {num_classes}."
            )

    @property
    def feature_dim(self) -> int:
        """分類器直前の特徴量次元数を返します。

        Returns
        -------
        int
            特徴量次元数。
        """
        return self.hidden_dim * (2 if self.bidirectional else 1)

    def forward(
        self,
        digits: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """digit ID列から4クラスlogitsを計算します。

        Parameters
        ----------
        digits : torch.Tensor
            padding済みdigit ID列です。形状は ``(batch_size, seq_len)`` です。
        lengths : torch.Tensor
            各サンプルの有効系列長です。形状は ``(batch_size,)`` です。

        Returns
        -------
        torch.Tensor
            4クラス分類のlogitsです。形状は ``(batch_size, num_classes)`` です。

        Raises
        ------
        ValueError
            入力テンソルの形状、dtype、値が不正な場合。
        """
        output = self.forward_with_features(digits=digits, lengths=lengths)
        return output.logits

    def forward_with_features(
        self,
        digits: torch.Tensor,
        lengths: torch.Tensor,
    ) -> ModelOutput:
        """digit ID列からlogitsと系列特徴量を計算します。

        Parameters
        ----------
        digits : torch.Tensor
            padding済みdigit ID列です。形状は ``(batch_size, seq_len)`` です。
        lengths : torch.Tensor
            各サンプルの有効系列長です。形状は ``(batch_size,)`` です。

        Returns
        -------
        ModelOutput
            logitsと分類器直前の特徴量を持つ出力コンテナ。

        Raises
        ------
        ValueError
            入力テンソルの形状、dtype、値が不正な場合。
        """
        self._validate_forward_inputs(digits=digits, lengths=lengths)

        embedded = self.embedding(digits)

        packed = pack_padded_sequence(
            embedded,
            lengths=lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, hidden = self.gru(packed)
        features = self._extract_last_layer_hidden(hidden)
        logits = self.classifier(features)

        return ModelOutput(
            logits=logits,
            features=features,
        )

    def _validate_forward_inputs(
        self,
        *,
        digits: torch.Tensor,
        lengths: torch.Tensor,
    ) -> None:
        """forward入力を検証します。

        Parameters
        ----------
        digits : torch.Tensor
            padding済みdigit ID列です。
        lengths : torch.Tensor
            各サンプルの有効系列長です。

        Returns
        -------
        None

        Raises
        ------
        ValueError
            入力テンソルの形状、dtype、値が不正な場合。
        """
        if digits.ndim != 2:
            raise ValueError(
                f"digits must be a 2D tensor, got shape={tuple(digits.shape)}."
            )
        if lengths.ndim != 1:
            raise ValueError(
                f"lengths must be a 1D tensor, got shape={tuple(lengths.shape)}."
            )
        if digits.size(0) != lengths.size(0):
            raise ValueError(
                "batch size mismatch between digits and lengths, "
                f"got digits batch={digits.size(0)}, lengths batch={lengths.size(0)}."
            )
        if digits.dtype != torch.long:
            raise ValueError(f"digits dtype must be torch.long, got {digits.dtype}.")
        if lengths.dtype != torch.long:
            raise ValueError(f"lengths dtype must be torch.long, got {lengths.dtype}.")
        if digits.numel() == 0:
            raise ValueError("digits must not be empty.")
        if lengths.numel() == 0:
            raise ValueError("lengths must not be empty.")

        min_digit = int(digits.min().item())
        max_digit = int(digits.max().item())
        if min_digit < 0:
            raise ValueError(f"digits must be non-negative, got min={min_digit}.")
        if max_digit >= self.vocab_size:
            raise ValueError(
                f"digits must be smaller than vocab_size={self.vocab_size}, "
                f"got max={max_digit}."
            )

        min_length = int(lengths.min().item())
        max_length = int(lengths.max().item())
        seq_len = digits.size(1)

        if min_length <= 0:
            raise ValueError(f"all lengths must be positive, got min={min_length}.")
        if max_length > seq_len:
            raise ValueError(
                "lengths must be less than or equal to sequence length, "
                f"got max_length={max_length}, seq_len={seq_len}."
            )

    def _extract_last_layer_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """GRUの最終層hidden stateを抽出します。

        Parameters
        ----------
        hidden : torch.Tensor
            GRUから返されるhidden stateです。

        Returns
        -------
        torch.Tensor
            分類器に入力する系列特徴量です。

        Raises
        ------
        ValueError
            ``hidden`` の形状が不正な場合。
        """
        expected_layers = self.num_layers * (2 if self.bidirectional else 1)

        if hidden.ndim != 3:
            raise ValueError(
                f"hidden must be a 3D tensor, got shape={tuple(hidden.shape)}."
            )
        if hidden.size(0) != expected_layers:
            raise ValueError(
                "unexpected hidden layers, "
                f"got {hidden.size(0)}, expected {expected_layers}."
            )

        if not self.bidirectional:
            return hidden[-1]

        batch_size = hidden.size(1)
        reshaped = hidden.view(self.num_layers, 2, batch_size, self.hidden_dim)
        last_layer = reshaped[-1]

        forward_hidden = last_layer[0]
        backward_hidden = last_layer[1]

        return torch.cat([forward_hidden, backward_hidden], dim=1)


def build_model(config: ModelConfig) -> DigitGRUClassifier:
    """設定からDigitGRUClassifierを構築します。

    Parameters
    ----------
    config : ModelConfig
        モデル設定。

    Returns
    -------
    DigitGRUClassifier
        構築されたGRU分類器。
    """
    return DigitGRUClassifier(
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        bidirectional=config.bidirectional,
        padding_idx=config.padding_idx,
        num_classes=4,
    )


def count_parameters(model: nn.Module, *, trainable_only: bool = True) -> int:
    """モデルのパラメータ数を数えます。

    Parameters
    ----------
    model : torch.nn.Module
        パラメータ数を数えるモデル。
    trainable_only : bool, default=True
        Trueの場合、学習対象パラメータのみを数えます。

    Returns
    -------
    int
        パラメータ数。

    Raises
    ------
    TypeError
        ``model`` が ``torch.nn.Module`` でない場合。
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be nn.Module, got {type(model).__name__}.")

    if trainable_only:
        return sum(param.numel() for param in model.parameters() if param.requires_grad)

    return sum(param.numel() for param in model.parameters())