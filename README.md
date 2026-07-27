# 大模型参数高效微调笔记

本笔记根据《大模型微调 · 袁茜笔记》整理，主要介绍 **LoRA、AdaLoRA、Prefix Tuning 与 P-Tuning** 的基本原理、训练过程及差异，并在文末补充基于 DeBERTa-v3-base 的 IMDB 情感分类 Kaggle 提交结果。

---

## 1. 冻结模型是什么意思

冻结不是不使用原模型，而是：

- 原模型继续参与前向传播；
- 原模型继续负责计算并提供已有知识；
- 反向传播后，优化器不更新原模型权重；
- 训练误差通过新增的 PEFT 分支传递，只更新少量新增参数。

假设预训练模型中有一个权重矩阵：

\[
W_0 \in \mathbb{R}^{d \times k}
\]

传统微调直接更新整个权重矩阵：

\[
W = W_0 + \Delta W
\]

其中，\(\Delta W\) 与原始权重矩阵一样大，因此包含大量需要训练的参数。

冻结后的核心状态是：

```text
W₀ 参与计算
W₀ 不参与更新
```

训练流程可以简化为：

```text
输入 x ─────────────→ 冻结的 W₀ ──────────┐
   │                                       │
   └────────────→ A ─────────→ B ──────────┤
                                           ↓
                                      相加并输出 y
```

其中：

- \(W_0\)：冻结；
- \(A\)、\(B\)：参与训练。

LoRA 初始化时通常：

```text
A：随机初始化
B：初始化为 0
```

训练初期主要先更新 \(B\)，之后 \(A\) 和 \(B\) 共同更新。

---

## 2. LoRA

### 2.1 核心思想

普通 LoRA 会给选定的线性层增加两个低秩矩阵：

\[
\Delta W = BA
\]

原始线性层：

\[
y = W_0x
\]

加入 LoRA 后：

\[
y = W_0x + BAx
\]

因此，LoRA 的本质是：

> 不改变输入，而是给模型内部权重增加一个低秩修正。

### 2.2 训练内容

LoRA 训练时主要更新：

- 低秩矩阵 \(A\)；
- 低秩矩阵 \(B\)；
- 序列分类任务中可能还包括分类头。

原模型大部分参数保持冻结。

### 2.3 普通 LoRA 的局限

普通 LoRA 通常会为每一层使用相同或预先固定的秩 \(r\)。

例如：

```text
第 1 层：r = 16
第 2 层：r = 16
第 3 层：r = 16
……
```

不论某一层对任务是否重要，通常都分配相似的参数预算。

---

## 3. AdaLoRA

### 3.1 AdaLoRA 是什么

AdaLoRA 中的 “Ada” 是 **Adaptive（自适应）**。

它的基本思想是：

> 在训练过程中判断不同层、不同低秩方向的重要程度，把更多参数预算分配给重要位置，把不重要的方向逐渐裁剪掉。

普通 LoRA：

```text
每一层平均分配固定数量的学习方向
```

AdaLoRA：

```text
重要层：保留更多方向
不重要层：减少方向
总参数预算：尽量保持受控
```

### 3.2 低秩更新表示

普通 LoRA 可以写成：

\[
\Delta W = BA
\]

为了分析不同方向的重要性，AdaLoRA 可以近似理解为：

\[
\Delta W = P\Lambda Q
\]

其中：

- \(P\)：左侧方向矩阵；
- \(Q\)：右侧方向矩阵；
- \(\Lambda\)：描述不同低秩方向的重要程度。

例如：

```text
第 1 个方向：重要程度高
第 2 个方向：重要程度一般
第 3 个方向：几乎没有作用
……
```

AdaLoRA 会在训练过程中不断估计这些方向的重要性。

### 3.3 AdaLoRA 的训练过程

```text
加载预训练模型
      ↓
冻结大部分原模型参数
      ↓
加入 AdaLoRA 低秩候选方向
      ↓
输入数据完成前向传播
      ↓
计算分类 Loss
      ↓
反向传播
      ↓
更新 AdaLoRA 参数
      ↓
估计各低秩方向的重要性
      ↓
裁剪不重要方向
      ↓
把预算分配给更重要的层
      ↓
继续训练直至结构稳定
```

代码中通常只需要创建 `AdaLoraConfig`，再交给 `get_peft_model()`，内部矩阵结构和参数管理由 PEFT 库完成。

---

## 4. Prefix Tuning

### 4.1 Prefix 的本质

Prefix 的训练过程，本质上是：

> 把 DeBERTa 主体当成一个固定不变的大函数，只把 Prefix 向量当成可学习参数，通过损失函数反向传播不断修改这些 Prefix 向量。

刚开始时，Prefix 并不懂“积极”或“消极”。

它只是一组随机初始化的可训练向量，例如：

```text
P1  = [ 0.12, -0.08, 0.31, ...]
P2  = [-0.15,  0.27, 0.04, ...]
...
P20 = [...]
```

这些不是词，也不是句子，而是模型内部的浮点数参数，可以理解为一组空白的“任务提示槽位”。

### 4.2 Prefix 如何作用于注意力

PEFT 会根据这些虚拟 Token，生成模型注意力层需要的 Prefix 信息。

Prefix 通常会影响 Transformer 各层注意力中的 Key 和 Value。

普通注意力使用：

\[
Q,\ K,\ V
\]

加入 Prefix 后，概念上变为：

\[
K' = [K_{\text{prefix}};K]
\]

\[
V' = [V_{\text{prefix}};V]
\]

然后计算：

\[
\operatorname{Attention}(Q,K',V')
\]

因此，文本中的每个词不仅可以关注真实句子，还可以关注 Prefix 提供的任务信息。

### 4.3 所有样本共享同一套 Prefix

并不是每一条评论都有一套 Prefix，而是所有 IMDB 评论共同使用同一组虚拟 Token。

训练过程可以理解为：

```text
第 1 条评论
   ↓
修改 Prefix 一点

第 2 条评论
   ↓
再修改 Prefix 一点

第 3 条评论
   ↓
继续修改 Prefix

……
整个训练集训练完成
   ↓
Prefix 逐渐适应 IMDB 情感分类任务
```

最终学到的是：

> 一套适用于整个 IMDB 情感分类任务的 Prefix。

### 4.4 Prefix Tuning 训练流程

```text
初始化 Prefix 参数
        ↓
为各层生成 Prefix Key 和 Prefix Value
        ↓
加入每层注意力计算
        ↓
冻结的 DeBERTa 完成前向传播
        ↓
输出积极 / 消极
        ↓
计算 Loss
        ↓
反向传播
        ↓
更新各层 Prefix 参数
```

Prefix 从随机向量开始，每次用电影评论进行预测，根据预测结果与真实标签之间的误差计算梯度。误差穿过冻结的 DeBERTa 传回 Prefix，优化器只修改 Prefix 参数，使其逐渐学会怎样引导 DeBERTa 完成情感分类。

---

## 5. P-Tuning

### 5.1 核心思想

P-Tuning 在输入表示前学习一组“软提示”。

原来的输入可以理解为：

```text
[CLS] This movie is wonderful [SEP]
```

加入 20 个虚拟 Token 后，概念上变成：

```text
[P1] [P2] ... [P20]
[CLS] This movie is wonderful [SEP]
```

其中，\(P_1\) 到 \(P_{20}\) 不是实际单词，而是可训练向量。

因此，P-Tuning 主要改变的是：

> 模型最开始看到的输入表示。

可以把它理解为：正式把电影评论交给 DeBERTa 之前，先给模型一段自动学习出来的“隐形提示”。

### 5.2 Prompt Encoder

P-Tuning 中的虚拟 Token 通常还会经过 Prompt Encoder 进一步加工：

```text
随机初始化虚拟 Token
        ↓
Prompt Encoder 加工
        ↓
生成任务软提示
        ↓
与真实文本 Embedding 拼接
        ↓
进入冻结的 DeBERTa
```

### 5.3 P-Tuning 训练内容

P-Tuning 主要训练：

- 虚拟 Token 向量；
- Prompt Encoder 参数；
- 序列分类任务中可能还包括分类头。

### 5.4 P-Tuning 训练流程

```text
随机初始化 20 个虚拟 Token
          ↓
经过 Prompt Encoder 加工
          ↓
与真实文本 Embedding 拼接
          ↓
进入冻结的 DeBERTa
          ↓
输出积极 / 消极
          ↓
计算 Loss
          ↓
反向传播
          ↓
更新虚拟 Token 和 Prompt Encoder
```

---

## 6. LoRA 与 Prefix Tuning 对比

| 对比项 | LoRA | Prefix Tuning |
|---|---|---|
| 核心思想 | 学习低秩权重修正 | 学习虚拟任务前缀 |
| 训练参数 | \(A\)、\(B\) 矩阵 | Prefix 向量 |
| 影响位置 | 模型线性层 | 各层注意力 Key、Value |
| 原模型参数 | 冻结 | 冻结 |
| 是否增加虚拟 Token | 否 | 是 |
| 是否可合并权重 | 可以 | 通常不可以 |
| 推理额外开销 | 合并后很低 | 通常存在 |
| 当前应用广度 | 很广 | 相对少一些 |
| 关键参数 | `r`、`alpha` | `num_virtual_tokens` |
| 直观理解 | 给权重打补丁 | 给模型增加任务提示 |

一句话记忆：

```text
LoRA：不改变输入，给模型内部权重增加一个低秩修正。
Prefix：不直接改变权重，给每层注意力增加一组可学习的虚拟提示。
```

---

## 7. P-Tuning 与 Prefix Tuning 对比

| 对比项 | P-Tuning | Prefix Tuning |
|---|---|---|
| 主要位置 | 输入 Embedding 附近 | Transformer 各层注意力 |
| 新增参数 | 虚拟 Token + Prompt Encoder | 各层 Prefix 参数 |
| 是否使用 Prompt Encoder | 通常使用 | 当前配置中不一定显式设置 |
| 影响范围 | 先改变输入表示 | 持续影响各层注意力 |
| 训练内容 | 虚拟 Token、Prompt Encoder | Prefix Key、Prefix Value |
| 直观理解 | 在模型入口处给软提示 | 在每层注意力中持续提供前缀 |

一句话记忆：

```text
P-Tuning：在输入表示前学习一组软提示。
Prefix Tuning：给 Transformer 各层注意力加入可学习的前缀信息。
```

---

## 8. 四种方法的统一理解

```text
LoRA
└── 学习低秩权重修正

AdaLoRA
└── 学习低秩权重修正，并动态分配秩预算

Prefix Tuning
└── 学习加入各层注意力 Key / Value 的任务前缀

P-Tuning
└── 学习输入端的虚拟 Prompt 和 Prompt Encoder
```

它们的共同点：

- 都属于参数高效微调；
- 都尽量冻结大模型主体；
- 都只训练少量新增参数；
- 都通过损失函数反向传播学习任务信息；
- 都可以明显减少需要更新的参数量。

---

## 9. Kaggle 实验结果

基于 **DeBERTa-v3-base** 完成 IMDB 情感分类实验后，分别提交了 LoRA、Prefix 与 P-Tuning 三组预测结果。Kaggle 页面显示的提交分数如下。

<p align="center">
  <img src="./assets/kaggle_submission_scores.png" alt="DeBERTa-v3-base 不同参数高效微调方法的 Kaggle 提交分数" width="100%">
</p>

<p align="center"><em>图 1　DeBERTa-v3-base 不同参数高效微调方法的 Kaggle 提交结果</em></p>

| 排名 | 方法 | Kaggle 分数 |
|:---:|---|---:|
| 1 | Prefix | **0.95004** |
| 2 | LoRA | **0.93820** |
| 3 | P-Tuning | **0.85028** |

从本次提交结果看：

- Prefix 的分数最高，为 **0.95004**；
- LoRA 的分数为 **0.93820**，比 Prefix 低 **0.01184**；
- P-Tuning 的分数为 **0.85028**，与 Prefix 相差 **0.09976**；
- 本表中的方法名称沿用 Kaggle 提交说明，具体实现与超参数应以对应训练脚本为准。

需要注意的是，这些分数反映的是当前训练配置下的单次提交结果。学习率、训练轮数、随机种子、最大文本长度、虚拟 Token 数量以及参数初始化等因素，都可能影响最终表现。

---

## 10. 最终总结

### LoRA

> 给原模型内部权重增加低秩补丁。

### AdaLoRA

> 在 LoRA 基础上动态判断不同层和方向的重要性，把参数预算集中到更重要的位置。

### Prefix Tuning

> 给 Transformer 每层注意力加入可训练的 Prefix Key 和 Prefix Value。

### P-Tuning

> 在输入 Embedding 前加入经过 Prompt Encoder 加工的软提示。
