# Decoder KD 与 OPD (On-Policy Distillation) 实验说明

## 目录结构

- `0_split_data.ipynb`：数据预处理，计算 Prompt Token 长度并切分。
- `1_teacher_finetune.ipynb`：使用 Alpaca 数据微调 Qwen2.5-1.5B（或 7B），保存为 `qwen_teacher_finetune`。
- `2_decoder_kd.ipynb`：传统 Decoder KD（Off-Policy）。学生模型直接在数据集的标准答案 Token 上，对齐 Teacher 的 Logits。
- `3_gen_opd.ipynb`：普通在线蒸馏（General OPD），直接在全量数据上进行纯 KL 散度约束训练。
- `4_cl_opd.ipynb`：课程在线蒸馏（Curriculum OPD）实验，按长度递进训练。
- `5.ad_cl_opd.ipynb`：自适应截断在线蒸馏（绝对阈值）。
- `5.ad_cl_opd_rel.ipynb`：自适应截断在线蒸馏（相对阈值），目前的最佳方案。

---

## OPD (On-Policy Distillation)

**OPD (On-Policy Distillation)** 的核心思想是：**让学生模型在自己真实的生成轨迹上接受教师的指导。**

### 1. 屏蔽标准答案
数据预处理阶段，不喂数据集中的 Response，模型在计算 Loss 前自己作答。

### 2. 学生模型自主生成
计算 `compute_loss` 阶段，先切换为推理态（`no_grad` + `eval`），输入 Prompt，使用 `.generate()` 自行续写出 `max_new_tokens` 个 token 的回答。

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

## 自适应截断在线蒸馏  (Adaptive Truncation OPD)

针对长文本生成中常见的误差累积（Compounding Errors）问题，引入了基于学生模型实时状态的**自适应截断机制**。

### 1. 核心思想

系统在训练中实时监控学生模型自回归生成每一步的预测概率分布，计算策略熵（Policy Entropy）：

$$H_t = -\sum_{v \in \mathcal{V}} p_\theta(v \mid x, y_{\lt t}) \log p_\theta(v \mid x, y_{\lt t})$$

如果连续 $N$ 个 Token 的熵值超出动态容忍阈值 $E_{max}$，表明模型开始产生幻觉或逻辑崩溃。通过一维卷积滑窗机制定位崩溃起点，利用后置掩码（将对应的 labels 改为 -100）截断后续无效轨迹的梯度回传。

### 2. 动态容忍阈值 ($E_{max}$) 的设计

#### 方案 A：绝对阈值 (Absolute Threshold)

对全局训练步数 $t$ 预设的绝对数值进行线性衰减，前期容错高，后期收紧：

$$E_{max}(t) = E_{start} - (E_{start} - E_{end}) \cdot \frac{t}{T_{total}}$$

#### 方案 B：相对阈值 (Relative Threshold - Z-score)

相对阈值方案会计算当前 Batch 内部所有生成 Token 策略熵集合 $\mathcal{H}$ 的均值与标准差，进行动态异常检测：

$$\mu_{\mathcal{H}} = \frac{1}{N} \sum_{i=1}^{N} H_i$$

$$\sigma_{\mathcal{H}} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (H_i - \mu_{\mathcal{H}})^2}$$

$$E_{max} = \mu_{\mathcal{H}} + k \cdot \sigma_{\mathcal{H}}$$

截断行为能自发适应不同数据的固有难度，仅剔除严重偏离当前平均认知水平的异常分布。

---

## 对比实验：

### 1. 实验设置

*   **数据**：2000 条训练集，200 条固定测试集。
*   **生成**：训练与评估阶段最大生成 128 Tokens。

**Untrained Student**
*   未经任何蒸馏训练的初始学生模型，作为基准参照。

**Standard KD (Off-Policy)**
*   基线离线蒸馏。使用完整 Ground Truth 数据，共训练 4 Epochs。

**General OPD (On-Policy)**
*   普通在线蒸馏。使用 Full 数据集，纯 KL 散度约束，全量训练 4 Epochs。

**Curriculum OPD (On-Policy)**
*   累加式课程学习。将训练数据按 Prompt 长度分位数切分为 Easy(25%)、Medium(50%)、Hard(75%)、Full(100%)。每阶段训练 1 Epoch。

**Adaptive OPD - Absolute (On-Policy)**
*   自适应绝对阈值截断。设置 $E_{start} = 3.0$，$E_{end} = 1.5$，耐心值 $N=3$。全量训练 4 Epochs。

**Adaptive OPD - Relative (On-Policy)**
*   自适应相对阈值截断。动态阈值乘数 $k=1.5$，放宽耐心值 $N=8$ 给予模型自我纠错缓冲。全量训练 4 Epochs。

### 2. 评估与结果

**评估**：在测试集（200 条）上进行自回归生成，计算生成轨迹与教师模型预测分布之间的平均 KL 散度。

| 模型 | 训练 | 遍历数据量 (条) | 测试集 KL 散度 |
| :--- | :--- | :--- | :--- |
| **Untrained Student** | N/A (初始权重) | 0 | 0.3608 |
| **Standard KD** | Off-Policy | 8,000 | 1.2210 |
| **Adaptive OPD (Absolute)** | On-Policy | 8,000 | 0.2660 |
| **Curriculum OPD** | On-Policy | 5,044 | 0.2077 |
| **General OPD** | On-Policy | 8,000 | 0.2022 |
| **Adaptive OPD (Relative)** | On-Policy | 8,000 | **0.2014** |

![kl_divergence_comparison_new](/images/kl_divergence_comparison_new.png)

### 3. 结果分析

#### 1：Off-Policy

Standard KD 在测试集上的 KL 散度（1.2210）远高于未经训练的模型。

**分析**：离线蒸馏的 Teacher Forcing 导致模型未见过自身的预测错误，测试生成时极易进入 Out-of-Distribution 状态并严重偏离教师分布。

#### 2：On-Policy

General OPD 的 KL 散度显著降至 0.2022。

**分析**：模型通过探索自身轨迹接受校准，有效消除了长度捷径问题。

#### 3：绝对阈值

Adaptive OPD (Absolute) 的 KL 散度出现明显退化（0.2660）。

**分析**：由于不同任务固有的信息熵差异极大，硬编码的绝对阈值显得过于严苛，导致模型频繁被截断,无法充分吸收教师的分布知识。

#### 4：相对阈值

Adaptive OPD (Relative) 目前取得了最低的 KL 散度（0.2014），优于静态课程学习与普通在线蒸馏基线。

**分析**：采用基于批次均值和标准差的相对阈值（Z-score检测），系统能够根据当前样本的实际难度动态伸缩容忍度边界，避免对困难知识的“错杀”与梯度饥饿。