# Decoder KD 与 OPD (On-Policy Distillation) 实验说明


## 目录结构

- `0_split_data.ipynb`：数据预处理，计算 Prompt Token 长度并切分。
- `1_teacher_finetune.ipynb`：使用 Alpaca 数据微调 `Qwen2.5-1.5B`（或 7B），保存为 `qwen_teacher_finetune`。
- `2_decoder_kd.ipynb`：传统 Decoder KD（Off-Policy）。学生模型直接在数据集的标准答案 Token 上，对齐 Teacher 的 Logits。
- `3_gen_opd.ipynb`：普通在线蒸馏（General OPD），直接在全量数据上进行纯 KL 散度约束训练。
- `4_cl_opd.ipynb`：课程在线蒸馏（Curriculum OPD）实验，按长度递进训练。

---

## OPD (On-Policy Distillation)

**OPD (On-Policy Distillation)** 的核心思想是：**让学生模型在自己真实的生成轨迹上接受教师的指导。**


### 1. 屏蔽标准答案
数据预处理阶段，不喂数据集中的 Response，模型在计算 Loss 前自己作答。

### 2. 学生模型自主生成
计算 `compute_loss` 阶段，先切换为推理态（`no_grad` + `eval`），输入 Prompt ,使用 `.generate()` 自行续写出 `max_new_tokens` 个 token 的回答。


### 3. 构造 Loss 掩码
拿到完整的 `[Prompt + 生成文本]` 序列，把 Prompt 长度范围内的 token Labels 设为 `-100`。只针对学生自主生成的 token。

### 4. 双路前向传播提取 Logits
将这串由学生自己生成的完整序列同时喂给两个模型：
*   **Student (带梯度计算):** 获得 `student_logits` 和 `SFT_Loss`。
*   **Teacher (无梯度):** 获得 `teacher_logits`（教师对这段“学生答案”的概率预测）。

### 5. KL 散度计算
提取两者词表的最小交集截断 Logits。调用 `losses.py`，计算目标 Token 上的 **Reverse KL**：
$$D_{KL}(P_{student} || P_{teacher})$$

然后 KL 散度除以有效 token 数量取平均，防止模型利用短序列规避惩罚。


### 6. 损失计算

在前一次的 OPD 实验中，总损失包含 SFT 损失（$Loss = \alpha \cdot Loss_{KL} + (1 - \alpha) \cdot Loss_{SFT}$），导致模式崩溃，学生模型为了最小化 SFT 交叉熵，倾向于生成短截断文本（如输出 2 个 token 后直接预测 EOS）。

使用 **GKD 范式**，仅保留 KL 散度：

$$Loss_{total} = Loss_{KL}$$

模型无法通过缩短生成长度降低损失，必须在完整的生成窗口内拟合教师分布。

---

## 对比实验：

### 1. 实验设置

*   **数据**：2000 条训练集，200 条固定测试集。
*   **生成**：训练与评估阶段最大生成 128 Tokens。


**Untrained Student**
*   未经任何蒸馏训练的初始学生模型，作为基准参照。

**Standard KD (Off-Policy)**

*   基线离线蒸馏。使用完整 Ground Truth 数据，学生模型计算交叉熵与 KL 散度，共训练 4 Epochs。

**General OPD (On-Policy)**

*   普通在线蒸馏。使用 Full 数据集，纯 KL 散度约束，全量训练 4 Epochs。

**Curriculum OPD (On-Policy)**

*   累加式课程学习。将训练数据按 Prompt 长度分位数切分为 Easy(25%)、Medium(50%)、Hard(75%)、Full(100%)。
*   模型依次在四个阶段中进行蒸馏，每阶段训练 1 Epoch，中途不重置优化器状态。

### 2. 评估与结果

**评估**：在测试集（200 条）上进行自回归生成，计算生成轨迹与教师模型预测分布之间的平均 KL 散度。

测试集 KL 散度结果如下：

| 模型 | 训练 | 遍历数据量 (条) | 测试集 KL 散度 |
| :--- | :--- | :--- | :--- |
| **Untrained Student** | N/A (初始权重) | 0 | 0.3608 |
| **Standard KD** | Off-Policy | 8,000 | 1.2210 |
| **General OPD** | On-Policy | 8,000 | 0.2022 |
| **Curriculum OPD** | On-Policy | 5,044 (522+1022+1500+2000) | 0.2077 |


![kl_divergence_comparison](\images\kl_divergence_comparison.png)

### 3. 结果分析

#### 1：Off-Policy
Standard KD 在测试集上的 KL 散度（1.2210）远高于其他组，明显比未经训练的初始学生模型还差。

**分析**：离线蒸馏依赖 Teacher Forcing，模型在训练时未见过自身的预测错误。在测试阶段进行自回归生成时，一旦偏离既定轨迹即进入 Out-of-Distribution 状态，严重偏离教师分布。

#### 2：On-Policy 与纯 KL 约束的有效性
General OPD 的 KL 散度从初始的 0.3608 显著降至 0.2022。

**分析**：模型通过探索自身轨迹并接受 KL 散度校准，消除了长度捷径（Length Hacking）。生成文本逻辑完整且紧密贴合目标教师分布，证明 On-Policy 蒸馏能够拉近师生分布。

#### 3：Curriculum OPD 的样本效率
Curriculum OPD 在测试集的 KL 散度为 0.2077，与 General OPD 接近，但训练数据遍历量减少了约 37%（5,044 条 vs 8,000 条）。

**分析**：采用累加式阶段训练保证了短序列的充分拟合，长序列的引入未造成灾难性遗忘。还可以进一步探索每个难度分级训练不同的epoch对结果的影响。

