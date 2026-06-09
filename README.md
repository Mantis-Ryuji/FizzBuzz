# FizzBuzz Digit Extrapolation Experiment

ニューラルネットワークが FizzBuzz の規則を、整数の10進数字列から学習し、学習時に見ていない桁数へ外挿できるかを検証する実験リポジトリです。

本実験では、入力特徴量として `n % 3` や `n % 5` は与えません。整数を10進数字列として入力し、Digit Embedding + GRU + Linear classifier により、以下の4クラス分類を行います。

| Label | Class    | Condition        |
| ----: | -------- | ---------------- |
|     0 | Number   | 3でも5でも割り切れない     |
|     1 | Fizz     | 3で割り切れ、5では割り切れない |
|     2 | Buzz     | 5で割り切れ、3では割り切れない |
|     3 | FizzBuzz | 15で割り切れる         |

## 1. Project Structure

```text
FizzBuzz/
├─ src/
│  └─ fizzbuzz/
│     ├─ __init__.py
│     ├─ config.py
│     ├─ data.py
│     ├─ model.py
│     ├─ metrics.py
│     ├─ trainer.py
│     ├─ evaluator.py
│     └─ utils.py
│
├─ scripts/
│  ├─ 01_train.py
│  └─ 02_eval_extrapolation.py
│
├─ configs/
│  ├─ small.yaml
│  ├─ medium.yaml
│  └─ large.yaml
│
├─ runs/
│  ├─ weights/
│  │  ├─ small/
│  │  ├─ medium/
│  │  └─ large/
│  └─ images/
│     └─ confusion_matrix/
│
├─ notebooks/
│  ├─ 01_analyze_training_history.ipynb
│  ├─ 02_analyze_extrapolation_results.ipynb
│  └─ 03_error_pattern_analysis.ipynb
│
├─ README.md
├─ requirements.txt
├─ .gitattributes
└─ .gitignore
```

各ディレクトリの役割は以下です。

```text
src/fizzbuzz/   再利用可能な実装本体
scripts/        実験実行スクリプト
configs/        モデルサイズ別の実験設定
runs/weights/   学習済み重み・学習履歴・評価結果
runs/images/    confusion matrix画像
```

## 2. Setup

Python環境を作成し、依存関係をインストールします。

```bash
pip install -r requirements.txt
```

`requirements.txt` の最小構成は以下です。

```txt
torch
numpy
pyyaml
pydantic>=2
tqdm
matplotlib
```

PyTorch は CUDA 環境に依存するため、必要に応じて公式手順に従ってインストールしてください。

## 3. Experiment Setting

学習データには1桁から5桁までの整数を使います。

```text
Train:
  1 <= n <= 99,999
```

外挿評価では、学習時に見ていない6桁・7桁・8桁の整数を評価します。

| Split       |                 Range | Purpose |
| ----------- | --------------------: | ------- |
| test_6digit |       100,000〜999,999 | 1段階外挿   |
| test_7digit |   1,000,000〜9,999,999 | 2段階外挿   |
| test_8digit | 10,000,000〜99,999,999 | 3段階外挿   |

8桁評価は9,000万件あります。全件をメモリに保持せず、batch単位で整数を生成して逐次評価します。

## 4. Model Configs

モデルサイズは以下の3種類です。

```text
configs/small.yaml
configs/medium.yaml
configs/large.yaml
```

基本構成は共通で、Digit Embedding + GRU + Linear classifier です。

| Config | Embedding dim | Hidden dim | GRU layers |
| ------ | ------------: | ---------: | ---------: |
| small  |            16 |         32 |          1 |
| medium |            64 |        128 |          2 |
| large  |           256 |        512 |          4 |

## 5. Train

### 5.1 Train a Single Model

例として small model を学習します。

```bash
python scripts/01_train.py --config configs/small.yaml
```

成功すると、以下が生成されます。

```text
runs/weights/small/
├─ model.pt
├─ history.json
├─ config.yaml
└─ train_summary.json
```

既存の `model.pt` が存在する場合は、誤上書きを防ぐためエラーになります。再学習して上書きする場合は `--overwrite` を付けます。

```bash
python scripts/01_train.py --config configs/small.yaml --overwrite
```

### 5.2 Train All Models

small / medium / large をまとめて学習します。

```bash
python scripts/01_train.py --all
```

既存重みを上書きする場合は以下です。

```bash
python scripts/01_train.py --all --overwrite
```

出力は以下です。

```text
runs/weights/
├─ small/
│  ├─ model.pt
│  ├─ history.json
│  ├─ config.yaml
│  └─ train_summary.json
├─ medium/
│  ├─ model.pt
│  ├─ history.json
│  ├─ config.yaml
│  └─ train_summary.json
└─ large/
   ├─ model.pt
   ├─ history.json
   ├─ config.yaml
   └─ train_summary.json
```

また、全体の学習要約として以下が保存されます。

```text
runs/train_summary.json
```

## 6. Evaluate Extrapolation

### 6.1 Evaluate a Single Model

small model を6桁・7桁・8桁で外挿評価します。

```bash
python scripts/02_eval_extrapolation.py --config configs/small.yaml
```

標準では、以下の重みを読み込みます。

```text
runs/weights/small/model.pt
```

別の重みを明示する場合は `--weight` を使います。

```bash
python scripts/02_eval_extrapolation.py \
  --config configs/small.yaml \
  --weight runs/weights/small/model.pt
```

評価結果は以下に保存されます。

```text
runs/weights/small/eval_results.json
```

confusion matrix 画像は以下に保存されます。

```text
runs/images/confusion_matrix/
├─ confusion_matrix_small_test_6digit.png
├─ confusion_matrix_small_test_7digit.png
└─ confusion_matrix_small_test_8digit.png
```

confusion matrix画像を作らない場合は、`--no-confmat` を付けます。

```bash
python scripts/02_eval_extrapolation.py \
  --config configs/small.yaml \
  --no-confmat
```

### 6.2 Evaluate All Models

small / medium / large をまとめて評価します。

```bash
python scripts/02_eval_extrapolation.py --all
```

出力は以下です。

```text
runs/weights/
├─ small/
│  └─ eval_results.json
├─ medium/
│  └─ eval_results.json
└─ large/
   └─ eval_results.json

runs/eval_summary.json
```

confusion matrix 画像は以下に保存されます。

```text
runs/images/confusion_matrix/
├─ confusion_matrix_small_test_6digit.png
├─ confusion_matrix_small_test_7digit.png
├─ confusion_matrix_small_test_8digit.png
├─ confusion_matrix_medium_test_6digit.png
├─ confusion_matrix_medium_test_7digit.png
├─ confusion_matrix_medium_test_8digit.png
├─ confusion_matrix_large_test_6digit.png
├─ confusion_matrix_large_test_7digit.png
└─ confusion_matrix_large_test_8digit.png
```

## 7. Output JSON Format

各モデルの評価結果は以下に保存されます。

```text
runs/weights/<model_name>/eval_results.json
```

主な構造は以下です。

```json
{
  "name": "small",
  "seed": 42,
  "config_path": "configs/small.yaml",
  "weight_path": "runs/weights/small/model.pt",
  "num_parameters": 5108,
  "results": {
    "test_6digit": {
      "metrics": {
        "accuracy": 0.99726,
        "macro_precision": 0.99648,
        "macro_recall": 0.99739,
        "macro_f1": 0.99693,
        "classwise": {
          "Number": {
            "precision": 0.99776,
            "recall": 0.99813,
            "f1": 0.99794,
            "support": 480000
          }
        },
        "confusion_matrix": [[...], [...], [...], [...]],
        "num_samples": 900000
      }
    }
  },
  "summary": {
    "test_6digit": {
      "accuracy": 0.99726,
      "macro_f1": 0.99693,
      "num_samples": 900000,
      "elapsed_sec": 4.64
    }
  }
}
```

`confusion_matrix` は以下の向きです。

```text
confusion_matrix[true_label][predicted_label]
```

行が正解ラベル、列が予測ラベルです。

## 8. Metrics

評価では以下を保存します。

```text
Accuracy
class-wise Precision
class-wise Recall
class-wise F1
Macro Precision
Macro Recall
Macro F1
Confusion matrix
```

FizzBuzz はクラス不均衡を含むため、Accuracy だけでなく Macro F1 を重視します。

## 9. Typical Workflow

基本的な実験手順は以下です。

```bash
# 1. small model で動作確認
python scripts/01_train.py --config configs/small.yaml
python scripts/02_eval_extrapolation.py --config configs/small.yaml

# 2. 問題なければ全モデルを学習
python scripts/01_train.py --all

# 3. 全モデルを外挿評価
python scripts/02_eval_extrapolation.py --all
```

再実験で重みを上書きする場合は以下です。

```bash
python scripts/01_train.py --all --overwrite
python scripts/02_eval_extrapolation.py --all
```

評価スクリプトは既存の `eval_results.json` と confusion matrix 画像を上書きします。`--overwrite` は不要です。