from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import cast

import numpy as np
import torch


JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonDict = dict[str, JsonValue]


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """乱数シードを固定します。

    Parameters
    ----------
    seed : int
        固定する乱数シード。
    deterministic : bool, default=True
        True の場合、可能な範囲で PyTorch の決定論的挙動を有効化します。

    Returns
    -------
    None

    Raises
    ------
    ValueError
        ``seed`` が負の整数の場合。
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}.")

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> Path:
    """ディレクトリを作成し、Path として返します。

    Parameters
    ----------
    path : str or pathlib.Path
        作成するディレクトリのパス。

    Returns
    -------
    pathlib.Path
        作成済みディレクトリのパス。

    Raises
    ------
    ValueError
        ``path`` が空文字の場合。
    """
    dir_path = Path(path)

    if str(dir_path).strip() == "":
        raise ValueError("path must not be empty.")

    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def save_json(data: JsonDict, path: str | Path, *, indent: int = 2) -> None:
    """辞書を JSON ファイルとして保存します。

    Parameters
    ----------
    data : dict[str, JsonValue]
        保存する JSON 互換の辞書。
    path : str or pathlib.Path
        保存先パス。
    indent : int, default=2
        JSON のインデント幅。

    Returns
    -------
    None

    Raises
    ------
    ValueError
        ``indent`` が負の場合。
    TypeError
        ``data`` が JSON serializable でない場合。
    OSError
        ファイル保存に失敗した場合。
    """
    if indent < 0:
        raise ValueError(f"indent must be non-negative, got {indent}.")

    save_path = Path(path)
    if str(save_path).strip() == "":
        raise ValueError("path must not be empty.")

    if save_path.parent != Path("."):
        save_path.parent.mkdir(parents=True, exist_ok=True)

    with save_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(path: str | Path) -> JsonDict:
    """JSON ファイルを読み込みます。

    Parameters
    ----------
    path : str or pathlib.Path
        読み込む JSON ファイルのパス。

    Returns
    -------
    dict[str, JsonValue]
        読み込まれた JSON 辞書。

    Raises
    ------
    FileNotFoundError
        指定されたファイルが存在しない場合。
    ValueError
        JSON のトップレベルが辞書でない場合。
    json.JSONDecodeError
        JSON として不正な場合。
    OSError
        ファイル読み込みに失敗した場合。
    """
    load_path = Path(path)

    if not load_path.exists():
        raise FileNotFoundError(f"JSON file not found: {load_path}")

    with load_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {load_path}")

    return cast(JsonDict, data)