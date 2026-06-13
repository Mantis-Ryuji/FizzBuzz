# FizzBuzz Digit Extrapolation Experiment

ニューラルネットワークが学習した FizzBuzz の規則を、学習時に見ていない桁数へ外挿できるかを検証する実験リポジトリです。

入力には `n % 3` や `n % 5` は与えません。整数を10進数字列として入力し、Digit Embedding + GRU + Linear classifier により、以下の4クラス分類を行います。

| Label | Class    | Condition        |
| ----: | -------- | ---------------- |
|     0 | Number   | 3でも5でも割り切れない     |
|     1 | Fizz     | 3で割り切れ、5では割り切れない |
|     2 | Buzz     | 5で割り切れ、3では割り切れない |
|     3 | FizzBuzz | 15で割り切れる         |

詳細は以下の Zenn 記事を参照してください。

[https://zenn.dev/mantis_ryuji/articles/fizz-buzz-exp](https://zenn.dev/mantis_ryuji/articles/fizz-buzz-exp)

## Project Structure

```text
FizzBuzz/
├─ src/fizzbuzz/
├─ scripts/
│  ├─ 01_train.py
│  ├─ 02_eval_extrapolation.py
│  ├─ 03_epoch_sweep.py
│  └─ 04_eval_grokking.py
├─ configs/
│  ├─ small.yaml
│  ├─ medium.yaml
│  └─ large.yaml
├─ runs/
│  ├─ weights/
│  ├─ grokking/
│  └─ images/
├─ LICENSE
├─ README.md
├─ requirements.txt
├─ .gitattributes
└─ .gitignore
```

## Setup

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

PyTorch は CUDA 環境に応じて適切な版をインストールしてください。

## Experiment Setting

学習には1桁から5桁までの整数を使います。

```text
Train: 1 <= n <= 99,999
```

外挿評価では、学習時に見ていない6桁・7桁・8桁の整数を評価します。

| Split       |                 Range | Purpose |
| ----------- | --------------------: | ------- |
| test_6digit |       100,000〜999,999 | 1段階外挿   |
| test_7digit |   1,000,000〜9,999,999 | 2段階外挿   |
| test_8digit | 10,000,000〜99,999,999 | 3段階外挿   |

## Model Configs

モデルは Digit Embedding + GRU + Linear classifier です。

| Config | Embedding dim | Hidden dim | GRU layers |
| ------ | ------------: | ---------: | ---------: |
| small  |            16 |         32 |          1 |
| medium |            64 |        128 |          2 |
| large  |           256 |        512 |          4 |

## Train

単一モデルを学習する場合:

```bash
python scripts/01_train.py --config configs/small.yaml
```

small / medium / large をまとめて学習する場合:

```bash
python scripts/01_train.py --all
```

既存の重みを上書きする場合は `--overwrite` を付けます。

```bash
python scripts/01_train.py --all --overwrite
```

学習結果は以下に保存されます。

```text
runs/weights/<model_name>/
├─ model.pt
├─ history.json
├─ config.yaml
└─ train_summary.json
```

## Evaluate

単一モデルを評価する場合:

```bash
python scripts/02_eval_extrapolation.py --config configs/small.yaml
```

small / medium / large をまとめて評価する場合:

```bash
python scripts/02_eval_extrapolation.py --all
```

評価結果は以下に保存されます。

```text
runs/weights/<model_name>/eval_results.json
runs/eval_summary.json
runs/images/confusion_matrix/
```

## Grokking Checkpoints

epoch ごとの性能推移を見る場合:

```bash
python scripts/03_epoch_sweep.py --all
python scripts/04_eval_grokking.py --all
```

`03_epoch_sweep.py` は最大 epoch まで1回だけ学習し、指定 epoch の重みを保存します。既に `model.pt` があるモデルは skip し、`last.pt` がある場合はそこから resume します。

```text
runs/grokking/<model_name>/
├─ model.pt
├─ last.pt
├─ history.json
├─ milestones/
│  ├─ epoch_000010.pt
│  ├─ epoch_000050.pt
│  └─ ...
└─ grokking_train_summary.json
```

評価結果は以下に保存されます。

```text
runs/grokking/<model_name>/grokking_eval_results.json
runs/grokking/<model_name>/evals/
runs/images/grokking/
```

## Metrics

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

FizzBuzz はクラス不均衡を含むため、Accuracy だけでなく Macro F1 も確認します。

## Typical Workflow

```bash
# 1. small model で動作確認
python scripts/01_train.py --config configs/small.yaml
python scripts/02_eval_extrapolation.py --config configs/small.yaml

# 2. 全モデルを学習
python scripts/01_train.py --all --overwrite

# 3. 全モデルを外挿評価
python scripts/02_eval_extrapolation.py --all

# 4. grokking checkpoint 実験
python scripts/03_epoch_sweep.py --all
python scripts/04_eval_grokking.py --all
```