---
tags:
  - flux2
  - klein
  - flux.2-klein-4b
  - text-to-image
  - internal-analysis
  - abliteration
  - uncensored
  - l2-norm-analysis
  - safety-filter
  - guardrails
  - transformer-debug
---

*Read this in other languages: [English](#english) | [日本語 (Japanese)](#日本語-japanese)*

<a id="english"></a>
# FLUX.2-klein-4B DiT Internal Analysis: Debunking the Guardrail Myth & Proof of Knowledge Gap

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/ponpoke)
*Tips are greatly appreciated and help sustain the compute resources needed for further research!*

## Overview
This repository provides an analytical toolkit designed to verify whether the Diffusion Transformer (DiT) engine of FLUX.2-klein-4B contains hidden "guardrail circuits" that intentionally block, noise, or distort extreme image generation. 

By measuring the L2 Norm (internal activation energy) across the model's layers, we mathematically confirmed that **FLUX.2's DiT does NOT contain active guardrails designed to intentionally destroy images in response to specific prompts**. 

Consequently, the inability to generate NSFW (Not Safe For Work) or extreme gore images is not due to the AI actively refusing the prompt. Instead, it is proven to be a **Knowledge Gap**—the developer's safety alignment involved thoroughly scrubbing these concepts from the training dataset, meaning the model simply does not know how to draw them.

## TL;DR: Required Actions per Domain
Our internal analysis reveals that the state of the model and the required solutions differ entirely depending on the domain you wish to generate.

| Domain | Model's Knowledge | Guardrail Mechanism | Required Action |
| :--- | :--- | :--- | :--- |
| Blood / Violence | Knows how to draw | None | **None** (Only Uncensored Text Encoder is needed) |
| Weapons / Firearms | Knows how to draw | None | **None** (Only Uncensored Text Encoder is needed) |
| NSFW / Nudity | Does NOT know | None | **Fine-tuning required** (Train an NSFW LoRA) |
| Extreme Gore / Dismemberment | Does NOT know | None | **Fine-tuning required** (Train a Gore LoRA) |

There are no built-in guardrail circuits blocking output within the FLUX.2-klein-4B DiT. The reason NSFW content fails to generate is purely due to dataset deletion. Therefore, **performing "Abliteration" (surgical weight removal) on the DiT is completely unnecessary and counterproductive**. The only practical solution is to supplement the missing knowledge through LoRA training.

## Analytical Scripts Provided
To compare the internal differences when inputting safe versus extreme prompts, we utilized the following two Python scripts:

* **`l2_norm_spike_detector.py`**
A tool that monitors changes in internal energy (L2 norm) across each Transformer block layer during image generation. If a guardrail circuit activates, the energy value will abnormally spike in specific layers, allowing us to pinpoint it.
* **`threshold_drop_test.py`**
A tool that interpolates the semantic embedding between a safe prompt and an extreme prompt (Alpha 0.0 to 1.0) to test at which exact blend ratio the structural integrity of the image collapses (Cliff drop).

*(Note: To ensure pure measurement of the DiT's behavior, these tests were conducted using a mathematically [uncensored Text Encoder](https://huggingface.co/ponpoke/flux2-klein-4b-uncensored-text-encoder) to bypass initial input restrictions.)*

## Verification Results: The Minimal Pair Tests
To determine whether the AI was "destroying the image via guardrails" or simply "lacking knowledge," we conducted comparative experiments using minimal pairs of safe and extreme prompts.

**1. Blood & Violence (Red Paint vs. Real Blood)**
* **Result:** Even with the extreme prompt (real blood), the L2 norm ratio remained between 1.01x and 1.04x across all layers, indicating no guardrail spikes. Structural scores (image quality) were maintained, and CLIP semantic scores increased.
* **Conclusion:** The DiT knows how to draw blood and does not apply output restrictions. Unlocking the Text Encoder is sufficient.

**2. NSFW & Nudity (Silk Dress vs. Explicit Nude)**
* **Result:** The L2 norm ratio was flat at 1.01x to 1.02x; no spikes were detected. However, even at maximum extreme alpha, the CLIP score languished between 26 and 29, failing to follow the prompt's intent.
* **Conclusion:** There is no forced image destruction by guardrails. The model simply lacks the knowledge due to complete dataset scrubbing.

**3. Weapons & Firearms (Toy Prop Gun vs. Real Lethal Firearm)**
* **Result:** The L2 norm ratio was 0.98x to 1.00x.
* **Conclusion:** Similar to blood, the DiT processes "toy" and "real" identically in internal calculations. No active blocking mechanism exists for firearms.

**4. Extreme Dismemberment & Gore**
* **Result:** The L2 norm ratio peaked at 1.05x, with no definitive spikes detected. However, as the extreme alpha increased, the structural score plummeted from 4685 to 1237.
* **Conclusion:** The lack of abnormal spikes implies this is not an intentional block. The image collapse is the result of the model attempting to reconstruct an unknown concept (dismemberment) and failing mathematically.

## Key Takeaways from this Verification

**1. Clarification of Internal Architecture (Block 4 Behavior)**
During our analysis, we observed a massive amplification in internal energy (L2 norm) at Transformer Block 4 (DoubleStream_Layer_4), jumping from approximately 30,000 to 330,000. At first glance, this might appear to be a strong safety filter (spike) activating at a specific layer. However, our controlled experiments confirmed that this amplification occurs equally even with completely safe prompts. Objective data proves that this numerical jump is not an output restriction, but rather a structural feature of the FLUX architecture (such as dimensional scaling).

**2. DiT Abliteration is Unnecessary**
Since there are no explicit mechanisms (spikes) blocking specific images within the DiT, attempting to forcibly carve out internal weights using mathematical methods like SVD is a meaningless act that will only destroy the model's normal generative capabilities. To generate unsupported genres in the current FLUX model, supplementing missing knowledge via fine-tuning is the mandatory next step.

<br><br><br>

---

<a id="日本語-japanese"></a>
# FLUX.2-klein-4B 内部構造の検証レポート：画像生成エンジン(DiT)に「ガードレール」は存在するのか？

## 概要
本リポジトリは、FLUX.2-klein-4Bの画像生成エンジン（DiT）の内部に、過激な画像を意図的にブロック（肉塊化・ノイズ化）する「ガードレール機構」が隠されていないかを検証するためのツールキットです。

L2ノルム（モデル内部の活性化エネルギー）の測定を行った結果、FLUX.2のDiTには、**特定のプロンプトに反応して画像を意図的に壊すようなガードレールは存在しない**ことが確認されました。

つまり、NSFW（性的表現）やグロテスクな表現の画像が生成できない現象は、AIが描画を拒否しているからではなく、**開発元の安全対策によって学習データから該当する画像が除外されており、そもそも描き方を知らない（知識の欠如）**ことが原因だと証明されました。

## 結論まとめ（必要な対策）
DiTの内部を解析した結果、表現させたいジャンルによってモデルの状態と必要な対策が完全に分かれることが判明しました。

| ジャンル | モデルの知識 | ガードレール機構 | 必要な対応 |
| :--- | :--- | :--- | :--- |
| 流血表現（血など） | 描き方を知っている | なし | テキストエンコーダーの制限解除のみ |
| 武器・兵器表現（銃など） | 描き方を知っている | なし | 同上 |
| 性的表現（NSFWなど） | 描き方を知らない | なし | NSFW LoRA |
| 人体欠損（グロテスク表現） | 描き方を知らない | なし | Gore LoRA |

FLUX.2-klein-4BのDiTには、出力をブロックするガードレールは最初から存在しません。NSFWなどが出力されない理由は、学習データから画像が削除されていて描き方を知らないためです。
そのため、モデルの内部を削り取るような改造（Abliteration）は一切不要であり、不足している知識をLoRAなどで追加学習させることが実質的な解決策となります。

## 公開スクリプト
モデルの内部を解析し、安全なプロンプトと過激なプロンプトを入力した際の違いを比較するために、以下の2つのPythonスクリプトを使用しています。

* **`l2_norm_spike_detector.py`**
AIが画像を生成する過程で、Transformerブロックの各層における内部エネルギー（L2ノルム）がどう変化するかを監視するツールです。ガードレールが作動した場合、特定の層で数値が異常に跳ね上がる（スパイクする）ため、それを見つけ出します。

* **`threshold_drop_test.py`**
安全なプロンプトと過激なプロンプトの間で意味合いを徐々にブレンド（Alpha 0.0〜1.0）しながら画像を連続生成し、どの段階で画像が崩壊するか、またはそのまま生成されるかをテストするツールです。

*(※検証にあたっては、条件を揃えるために[制限を解除したテキストエンコーダー](https://huggingface.co/ponpoke/flux2-klein-4b-uncensored-text-encoder)を使用し、純粋なDiT側の挙動のみを測定しています)*

## 検証結果（4つのドメインにおける比較）
AIが「ガードレールで画像を壊している」のか、単に「描き方を知らない」のかを見極めるため、無難なプロンプトと過激なプロンプトの対を用いた実験を行いました。

**1. 流血・暴力表現（赤いペンキ vs 本物の血）**
* **結果**: 過激なプロンプト（本物の血）を入力しても、内部エネルギーの比率は全層で1.01倍から1.04倍に収まり、ガードレール作動を示す異常なスパイクは起きませんでした。また、構造スコア（画質）も維持され、CLIPスコア（テキストとの一致度）は上昇しました。
* **結論**: DiTは血の描き方を知っており、ガードレールによる出力制限も行っていません。テキストエンコーダーの制限さえ解除していれば問題なく描画されます。

**2. 性的表現・NSFW（ドレス姿 vs ヌード）**
* **結果**: 内部エネルギーの比率は1.01倍から1.02倍と平坦で、ここでもスパイクは検出されませんでした。しかし、過激度を最大にしてもCLIPスコアは26から29の範囲で低迷し、指示通りの画像にはなりませんでした。
* **結論**: ガードレールによる強制的な画像破壊は起きていません。学習データセットから完全に該当画像が削除されているため、モデルが描き方を知らない状態です。

**3. 武器・兵器表現（おもちゃの銃 vs 本物の銃）**
* **結果**: 内部エネルギーの比率は0.98倍から1.00倍でした。
* **結論**: 流血表現と同様に、DiTは内部計算においておもちゃと本物を同等に処理しており、銃器に対する能動的なブロック機構は存在しません。

**4. 極端な人体欠損・グロテスク表現**
* **結果**: 内部エネルギーの比率は最大1.05倍にとどまり、明確なスパイクは検出されませんでした。しかし、過激度を上げると構造スコア（画質）が4685から1237へと大きく低下しました。
* **結論**: 異常なスパイクが発生していないことから、意図的なブロックではなく、知らないもの（欠損）を無理やり再構築しようとして破綻した結果だと推測されます。

## この検証から分かったこと

**1. 内部アーキテクチャの仕様解明（Block 4の挙動）**
本検証の過程で、TransformerのBlock 4（DoubleStream_Layer_4）において、内部エネルギー（L2ノルム）が約3万から約33万へと急激に増幅する現象が観測されました。一見すると特定の層で強力なフィルター（スパイク）が作動しているように見えますが、対照実験の結果、これは安全なプロンプト（Safe）であっても等しく発生することが確認されました。つまり、この数値の跳ね上がりは出力制限ではなく、FLUXの構造上の仕様（次元スケールの切り替え等）であることが客観的なデータとして証明されました。

**2. モデルの重みを削る改造（Abliteration）は不要**
DiT側に特定の画像をブロックする明確な機構（スパイク）が存在しない以上、SVD等の数学的手法を用いて無理にモデルの内部を削り取ろうとする試行は、正常な画像生成能力を壊してしまうだけの無意味な行為です。現在のFLUXモデルにおいて出力できないジャンルについては、追加学習による知識の補完が必須となります。
```