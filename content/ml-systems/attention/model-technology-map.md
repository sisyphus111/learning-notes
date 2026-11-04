---
title: 模型技术总表
category: attention
tags: [DeepSeek, Kimi, Qwen, MLA, MoE]
description: 按模型查 attention · FFN / MoE · residual · 条件记忆 · MTP · 训练技术
updated: 2026-11-04
---
# 模型技术地图：DeepSeek、Kimi K3 与 Qwen3.8

本页于 **2026-11-02** 扩充。先看前半页的技术归属；[第 7 节起](#7-deepseek-主系列谱系与尺寸)列出各代尺寸、头数与专家配置，再分别解释 V4.1、Qwen3.8 和 K3/K3.1。配置来自当天重新核验的官方文件。

## 1. 主干总表：先看 attention、FFN 和 residual

这三列分别回答：**怎样读取历史、怎样变换当前 token 的特征、怎样把表示交给后面的层。** 同一行是一个具体模型；同一列才是作用相近、适合放在一起对照的技术。

| 模型 | Attention：历史怎样读取 | FFN：当前 token 怎样变换 | Residual：不同深度怎样传递 |
| --- | --- | --- | --- |
| **DeepSeek-V2** | **MLA**：每个 token 保存压缩的 K/V 表示，读取全部因果历史 | **DeepSeekMoE**：细粒度 routed experts + shared experts；首层 dense | 普通单流残差加法 |
| **DeepSeek-V3** | **MLA**：延续 V2 的核心路线 | **DeepSeekMoE**；前三层 dense，后续使用专家 | 普通单流残差加法 |
| **DeepSeek-V3.2** | **MLA + DSA**：先选历史 token，再读取对应 latent | 延续 V3 系列的 DeepSeekMoE | 普通单流残差加法 |
| **DeepSeek-V4-Flash** | **CSA / HCA + SWA**：压缩历史条目，另保留近期窗口；底层采用 shared K=V 的 MQA | **MoE**；前三层采用 hash routing | **mHC**：4 条残差流，约束分支之间的混合 |
| **DeepSeek-V4-Pro** | 同属 **CSA / HCA + SWA** 路线；具体层布局和规模不同于 Flash | **MoE**；前三层采用 hash routing，专家池更大 | **mHC**：4 条残差流 |
| **DeepSeek-V4.1-Flash** | **CED + CSA2 + SWA**：因果编码器/解码器，跨层复用主 KV 与索引 | MoE：384 routed 选 6，另有 shared expert | **Single-Pass mHC**，4 条残差流 |
| **Kimi K3.1 Pro** | 官方确认是 K3 后训练版；独立配置待核验 | 以 K3 为结构参照，避免把参照尺寸当作已发布配置 | K3 的 Block AttnRes 为结构参照 |
| **Kimi K3** | **KDA + gated MLA**：递推状态层与历史读取层混合 | 首层 dense；后续 **Stable LatentMoE**：routed experts 用较窄空间，shared experts 保持主干宽度 | **Block AttnRes**：按权重读取同一 token 的不同深度块 |
| **Qwen3.8-27B** | **GDN + gated full attention**，full-attention 层使用 GQA | **Dense FFN**：每个 token 经过同一套 FFN | 普通单流残差加法 |
| **Qwen3.8-2.4T-A95B** | **GDN + gated full attention**，full-attention 层使用 GQA | **MoE**：routed experts + shared expert | 普通单流残差加法 |
| **Qwen3.8-Flash-Next** | **GDN + QSA**：递推状态层与稀疏历史读取层混合 | **MoE**：routed experts + shared expert | **GR**：4 个残差分支，控制读取和写入 |

这里把“Qwen3.8”拆成 27B dense、2.4T-A95B MoE 两个代表配置，再单列 Flash-Next。**Qwen3.8 不是一种唯一的结构。** V4 也拆成 Flash 和 Pro，避免把二者的层布局与规模混为一谈。

机制来源见[第 6 节](#6-来源与详细阅读入口)，本轮固定配置见第 12 节。已有长篇解释保持独立：本页用于定位技术，再按链接阅读机制，不需要在这里记住参数量或计算公式。

## 2. Attention 对照：到底压缩了什么，选择了什么

不同名称可以组合，因为它们改变的对象不同。比如 **MLA** 决定历史怎样表示，**DSA** 决定当前读哪些历史；因此 V3.2 可以同时使用二者。

| 技术 | 改变的对象 | 当前 query 读取什么 | 本页中的模型 |
| --- | --- | --- | --- |
| **MLA** | 压缩每个 token 的 K/V 特征宽度 | 普通 MLA 读取各历史 token 的 latent；可与稀疏选择组合 | V2、V3、V3.2；K3 的 MLA 层 |
| **DSA** | 选择原始历史 token 的位置 | indexer 选中的 token 对应的 MLA 条目 | V3.2 |
| **CSA** | 将相邻 token 合成压缩条目，再选择条目 | 选中的压缩条目，另有近期窗口 | V4-Flash / Pro |
| **HCA** | 将更长的 token 片段合成粗粒度条目 | 全部因果可见的粗条目，另有近期窗口 | V4-Flash / Pro |
| **SWA** | 限制局部读取窗口 | 最近一段历史的条目 | V4 的局部分支 |
| **GQA / MQA** | 让多个 query heads 共享 K/V heads | 读取范围另由 full attention、窗口或稀疏规则决定 | Qwen 的 full-attention / QSA 层；V4 的 MQA |
| **GDN** | 将历史持续写进固定形状的递推状态 | 当前状态；其他混合层仍可以读取历史 KV | Qwen3.8 的三个代表分支 |
| **KDA** | 通过带门控的 delta 更新维护历史状态 | 当前状态；K3 的 MLA 层另存历史缓存 | Kimi K3 |
| **QSA** | 用 micro-block 索引筛选一组历史 token | **选中块内原始 token 的 K/V** | Qwen3.8-Flash-Next |

**最容易混的三组关系：**

- **MLA 与 CSA/HCA**：MLA 缩窄每条历史记录；CSA/HCA 把多个 token 合成更少的记录。
- **DSA 与 QSA**：DSA 按 token 选择，QSA 按 micro-block 选择；二者主 attention 仍读取选中 token 的表示。
- **QSA 与 CSA**：QSA 的主 attention 读取原始 token K/V；CSA 的主 attention 读取压缩条目。两者名字相近，读取对象不同。

GDN/KDA 的递推状态在形状上不随历史长度增长，但混合模型还有 full attention 或 MLA 层，所以不能说整个模型的缓存都是常量。HCA 虽然压缩历史，压缩条目仍随序列增长，也不能归入固定状态的 linear attention。

详细解释：[MLA 与各类稀疏/混合 attention](model-architectures.md)、[DeepSeek V2–V4 演进](../DeepSeek/evolution.md)。

## 3. 条件记忆、MTP 与视觉：放在主干旁边单独看

**条件记忆**是额外的参数查表路径；**MTP**是额外预测目标与辅助模块；**视觉编码器**把图像转换成模型能处理的表示。它们与第一张表的三个主干部件作用不同。

下表中的“未列”表示所核对的报告/发布配置未列出该专用模块，不代表模型没有训练得到的知识，也不是对所有后续衍生实现的判断。

| 模型 | 专用条件记忆 | MTP / 草稿生成 | 公开模型的视觉路径 |
| --- | --- | --- | --- |
| **DeepSeek-V2** | 未列 | 原始报告未列 MTP | 文本主干 |
| **DeepSeek-V3** | 未列 | **MTP**，可用于推测生成 | 文本主干 |
| **DeepSeek-V3.2** | 未列 | 发布配置保留 **MTP** | 文本主干 |
| **DeepSeek-V4-Flash** | 未列；hash routing 属于专家路由 | **MTP** | 文本主干 |
| **DeepSeek-V4-Pro** | 未列；hash routing 属于专家路由 | **MTP** | 文本主干 |
| **DeepSeek-V4.1-Flash** | **Engram**，额外196B参数 | **DSpark**，独立训练draft | DeepSeek-ViT + projector |
| **Kimi K3.1 Pro** | 独立配置待核验 | 官方确认后训练升级；draft配置待核验 | 官方确认原生视觉 |
| **Kimi K3** | 未列 n-gram / Engram | 预训练 **MTP**；后续微调为 **EAGLE-3-style draft** | **MoonViT-V2 + projector**，图像/视频 |
| **Qwen3.8-27B** | 未列 n-gram / Engram | **MTP**，多步训练 | 公开 checkpoint 带视觉编码器 |
| **Qwen3.8-2.4T-A95B** | 未列 n-gram / Engram | **MTP**，多步训练 | **公开 checkpoint 为纯文本**；勿混同带视觉的托管 Qwen3.8-Max |
| **Qwen3.8-Flash-Next** | **N-gram embedding**：bigram/trigram + 上下文门控 | **MTP**，多步训练 | 公开 checkpoint 带视觉编码器 |

K3 报告讨论了 MTP 到 draft 的训练路径，但发布主干配置中的 `num_nextn_predict_layers=0`，不能据报告就认为主干包自带可直接运行的 MTP 层。

MTP 的“训练采用”和“这个发布包能直接用于草稿生成”要分开。服务框架还需实际支持并启用草稿与验证路径，才能讨论推理加速。

### Engram 应该放到哪里

**DeepSeek-V4.1-Flash 采用 Engram，原版 V4 的那一行不要填 Engram。** V4.1 的 CED、CSA2、Single-Pass mHC、DSpark 也应放在 V4.1 自己的版本条目里。V4.1 的具体尺寸和复用路径见第 8 节。

**Qwen3.8-Flash-Next 使用自己的 n-gram embedding 模块。** 它与 Engram 都包含“局部 token 组合查表、按上下文融合”的思路，但不能把 DeepSeek 原论文的每个细节都直接套到 Qwen。

**DeepSeek V4 的 hash routing 是选择专家。** 它根据 token ID 确定 expert IDs，随后仍需执行专家网络。n-gram embedding 则根据局部 token 组合读取参数向量。虽然都有查地址这一步，最终拿到的对象不同。

详细解释：[n-gram embedding、Engram 与 MTP](residual-experts-memory.md)。

## 4. 训练与数值技术：不要放进 attention 那一列

这些名称描述训练权重、平衡专家负载或组织设备执行的方法。本表只列帮助识别各代设计的代表项，不是完整训练配方。

| 模型 | 代表技术 | 它负责什么 |
| --- | --- | --- |
| **DeepSeek-V2** | AdamW；专家/设备/通信均衡辅助损失 | 更新参数，控制 MoE 路由的负载分布 |
| **DeepSeek-V3** | 辅助损失自由的负载均衡；**FP8 混合精度训练、DualPipe** | 平衡专家、降低矩阵计算成本、重叠流水线计算与通信 |
| **DeepSeek-V3.2** | **DSA indexer 的 warm-up 与稀疏训练**；低精度 indexer | 让选择器学会找重要 token，而非随机截断历史 |
| **DeepSeek-V4-Flash / Pro** | **Muon** 与部分参数 AdamW；EP 通信计算重叠；FP4 QAT | 改进权重更新、专家并行执行和低精度部署 |
| **Kimi K3** | **Per-Head Muon、Quantile Balancing、SiTU-GLU、MXFP4/MXFP8 QAT** | 分别涉及权重更新、路由均衡、激活稳定、量化适配 |
| **Qwen3.8-27B / 2.4T-A95B** | 本页未核定独有 optimizer 配方 | 不从 Flash-Next 的报告推断整个 Qwen3.8 系列使用同一配方 |
| **Qwen3.8-Flash-Next** | **Muon + AdamW** 按参数类型使用；与 GR 配合的稳定性设计 | 调整参数更新方式，并控制训练中的数值尺度 |

V3 的“辅助损失自由”主要指专家负载用选择偏置调节，它仍保留小权重的序列均衡辅助项。SiTU-GLU 本身是网络里的激活结构，训练后仍参与前向；这里按其稳定训练的作用列出。

**Muon 是 optimizer，DualPipe 是流水线调度，FP8/FP4 是数值格式或训练部署方案。** 它们不是 MLA、KDA、QSA 那样的序列计算机制。模型使用 FP8 或 FP4，也不表示所有权重、缓存和算子都使用同一种精度。

## 5. 不熟悉的名词，按它负责的事情找

| 想弄清楚的事情 | 对应名词 | 详细页面 |
| --- | --- | --- |
| 多个 query heads 怎样共享历史表示 | GQA、MQA、MLA | [模型结构与注意力变体](model-architectures.md) |
| 当前 token 选哪些历史读取 | DSA、QSA | [模型结构与注意力变体](model-architectures.md) |
| 历史怎样合并成更少的条目 | CSA、HCA | [DeepSeek 演进](../DeepSeek/evolution.md) |
| 历史怎样累积成递推状态 | GDN、KDA | [线性注意力](flash-linear-attention.md) |
| 哪些 FFN 专家执行，专家在多宽空间工作 | MoE、shared/routed experts、LatentMoE | [专家与条件记忆详解](residual-experts-memory.md) |
| 前面层的结果怎样交给后面层 | 普通 residual、mHC、Block AttnRes、GR | [mHC](../DeepSeek/evolution.md#8-mhc-同一个-token-的多条残差流)、[GR 与 Block AttnRes](residual-experts-memory.md) |
| 局部 token 组合怎样查训练得到的记忆 | n-gram embedding、Engram | [条件记忆详解](residual-experts-memory.md) |
| 怎样训练额外预测、提出多个候选 token | MTP、speculative decoding | [多模态与生成路径](model-architectures.md#12-多模态-mtp-与训练方法属于哪些层次) |
| 权重怎样更新、训练怎样跑得动 | AdamW、Muon、FP8、DualPipe | [Muon](../DeepSeek/evolution.md#9-muon-训练时变换更新矩阵)、[精度与量化](../GPU/precision-and-quantization.md) |

这里的各列可以组合：**一个模型可以同时用 MLA、MoE 和普通 residual，也可以同时用 KDA/MLA、LatentMoE 和 Block AttnRes。** 学新名词时先确定它改变哪部分，再回到对应模型行查看组合。

## 6. 来源与详细阅读入口

前六节初次核对：**2026-10-29**；本轮扩充核对：**2026-11-02**。表格描述指定模型的公开结构与训练设计，不等同于某个 serving 框架已经启用的 kernel 或生成配置。未运行模型或性能测试。

| 模型/主题 | 官方来源 | 本站进一步阅读 |
| --- | --- | --- |
| DeepSeek-V2 | [报告](https://arxiv.org/html/2405.04434v5)、[固定配置](https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/4461458f186c35188585855f28f77af5661ad489/config.json) | [V2–V4 演进](../DeepSeek/evolution.md) |
| DeepSeek-V3 | [报告](https://arxiv.org/html/2412.19437v2) | [V3 的 MLA、MoE 与训练](../DeepSeek/evolution.md#4-v3-mla-不变-把训练与-moe-扩展到更大规模) |
| DeepSeek-V3.2 | [报告](https://arxiv.org/html/2512.02556v1)、[固定配置](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/a7e62ac04ecb2c0a54d736dc46601c5606cf10a6/config.json) | [DSA](../DeepSeek/evolution.md#5-v3-2-dsa-indexer-选择位置-主-mla-计算结果) |
| DeepSeek-V4 | [报告](https://arxiv.org/html/2606.19348v1)、[Flash 配置](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/config.json)、[Pro 配置](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/config.json) | [CSA/HCA 与 mHC](../DeepSeek/evolution.md) |
| Kimi K3 | [报告](https://arxiv.org/html/2607.24653v1)、[模型卡](https://huggingface.co/moonshotai/Kimi-K3)、[固定配置](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/config.json) | [Block AttnRes 与 LatentMoE](residual-experts-memory.md) |
| Qwen3.8-27B | [模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)、[固定配置](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json) | [Qwen 各分支](model-architectures.md#7-qwen-dense-moe-与-attention-路线是独立选择) |
| Qwen3.8-2.4T-A95B | [模型卡](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B)、[固定配置](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/blob/207bd685a7e3696cfaff12ded7c6a7ea0f88c996/config.json) | [Qwen 各分支](model-architectures.md#7-qwen-dense-moe-与-attention-路线是独立选择) |
| 常规 Qwen3.8 的 residual | [dense 实现](https://github.com/huggingface/transformers/blob/v5.15.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L674)、[MoE 实现](https://github.com/huggingface/transformers/blob/v5.15.0/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L747)，与上述 checkpoint 模型类对应 | [与 GR 区分](residual-experts-memory.md) |
| Qwen3.8-Flash-Next | [报告](https://arxiv.org/html/2608.30320v1)、[模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | [GR 与 n-gram embedding](residual-experts-memory.md) |
| Engram / V4.1 边界 | [Engram 原论文](https://arxiv.org/html/2601.07372v1)、[V4.1 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | [Engram 机制](residual-experts-memory.md) |


## 7. DeepSeek 主系列：谱系与尺寸

### 先看哪些版本共享主干

```text
早期 DeepSeek LLM：dense Transformer（7B MHA、67B GQA）
DeepSeekMoE 16B：细粒度专家 + shared experts，尚未采用 MLA
    ↓
V2 / V2-Lite：MLA + DeepSeekMoE
    └→ V2.5：延续 V2 尺寸，融合通用与代码能力
    ↓
V3：更大 MLA/MoE 主干 + MTP + FP8 训练 + DualPipe
    ├→ V3-0324
    ├→ R1 / R1-0528：相同主要尺寸，改变后训练
    └→ V3.1 / Terminus：主要尺寸延续，混合推理与后训练更新
         ↓
V3.2-Exp / V3.2 / Speciale：MLA + DSA indexer
    ↓ 架构路线转变，不是仅换 checkpoint
V4-Flash / Pro：shared K=V MQA + CSA/HCA + SWA + mHC
    ↓
V4.1-Flash / 1017：CED + CSA2 + Engram + Single-Pass mHC + DSpark
```

这是结构路线图，不表示图中每条线都是一次直接权重初始化关系。“尺寸相同”也不表示权重、训练数据或能力相同。R1-Distill 使用学生模型自身架构，不能填进原版 R1 的 MLA 行。

### 主要尺寸与 attention heads

L 是语言主干层数，D 是 hidden 宽度。`内容K+位置分支 / V` 是 MLA 的逻辑 head 宽度；共同 latent 则是历史存储的另一个维度。MLA 配置里的 `num_key_value_heads=128` 不能读成普通 GQA 的 128 份独立 KV cache。

| 模型 / 主干组 | L / D | Q heads | 主 attention 表示 | Routed 池 / 每 token 选 / Shared |
| --- | --- | --- | --- | --- |
| DeepSeek LLM 7B | 30 / 4096 | 32 | MHA；每头 128 | dense FFN |
| DeepSeek LLM 67B | 95 / 8192 | 64 | GQA；8 KV heads，每头 128 | dense FFN |
| DeepSeekMoE 16B | 28 / 2048 | 16 | MHA；16 KV heads，每头 128 | 64 / 6 / 2 |
| V2-Lite | 27 / 2048 | 16 | MLA；内容K 128+位置64 / V128；latent512 | 64 / 6 / 2 |
| V2 / V2.5 | 60 / 5120 | 128 | MLA；内容K 128+位置64 / V128；latent512 | 160 / 6 / 2 |
| V3 / 0324 / R1 / 0528 / V3.1 / Terminus | 61 / 7168 | 128 | MLA；内容K 128+位置64 / V128；latent512 | 256 / 8 / 1 |
| V3.2-Exp / V3.2 / Speciale | 61 / 7168 | 128 | MLA + DSA；主 head/latent 尺寸延续 V3 | 256 / 8 / 1 |
| V4-Flash | 43 / 4096 | 64 | shared K=V MQA；1 KV head，宽512 | 256 / 6 / 1 |
| V4-Pro | 61 / 7168 | 128 | shared K=V MQA；1 KV head，宽512 | 384 / 6 / 1 |
| V4.1-Flash / 1017 | 40 / 5120 | 64 | shared K=V MQA；1 KV head，宽512 | 384 / 6 / 1 |

V2-Lite 的 Q 投影不采用大 V2 的同一套低秩 Q 路径。V2/V2.5、V3 家族、V3.2 家族的尺寸分组来自分别读取各 checkpoint 配置，而非根据版本号猜测。

### 参数量口径

| 主干组 | 官方常用总量 / 激活量口径 |
| --- | --- |
| DeepSeek LLM | 7B / 67B为dense模型规模 |
| V2-Lite | 约16B / 2.4B |
| V2 / V2.5 | 约236B / 21B |
| V3、R1、V3.1主干组 | 约671B / 37B |
| V3.2家族 | 延续约671B规模并增加indexer，不能据此声称与V3参数逐项相同 |
| V4-Flash / Pro | 284B / 13B；约1.6T / 49B |
| V4.1-Flash | 552B主干 + 196B Engram；prefill8B / decode16B；视觉和draft还应单独看发布口径 |

总量决定需要保存的权重规模，激活量描述每token所用路径；两者都不是可直接代入的运行显存或延迟。上表保留报告近似口径，不把量化包显示的tensor数量直接等同主干参数。

### FFN、索引器与残差路径

| 主干组 | FFN intermediate / dense 例外 | Indexer 与局部历史 | Residual / 代表性训练技术 |
| --- | --- | --- | --- |
| DeepSeek LLM 7B / 67B | dense：11008 / 22016 | 无本页所述稀疏 indexer | 单流 residual |
| DeepSeekMoE 16B | 专家1408；首层 dense10944 | 无 MLA/DSA | 单流；细粒度专家、共享专家 |
| V2-Lite | 专家1408；首层 dense10944 | 无 DSA | 单流 residual |
| V2 / V2.5 | 专家1536；首层 dense12288 | MLA 读取全历史 | 单流；MoE 负载均衡 |
| V3 / R1 / V3.1 主干组 | 专家2048；前三层 dense18432 | 非 DSA 的 MLA | 单流；V3 引入 MTP、辅助损失自由均衡、FP8 训练和 DualPipe；后训练版不等于重新预训练 |
| V3.2 家族 | 专家2048；前三层 dense18432 | **64 indexer Q heads，宽128，top-2048 token** | 单流；indexer warm-up 和稀疏训练 |
| V4-Flash | 专家2048；各层为 MoE，前三层 hash routing | **64×128 indexer，top-512 压缩条目**；CSA步长4、HCA128；SWA128 | **mHC 4流**；Muon、低精度专家路径 |
| V4-Pro | 专家3072；各层为 MoE，前三层 hash routing | **64×128 indexer，top-1024 压缩条目**；CSA步长4、HCA128；SWA128 | **mHC 4流**；专家池与层布局不同于 Flash |
| V4.1-Flash / 1017 | 专家2304；384选6+1 shared | **32×128 indexer，top-512**；CSA2跨层复用、分级候选池；SWA128 | **Single-Pass mHC 4流**；Engram、DSpark、原生视觉 |

V4-Flash 开头两层为纯 SWA；V4-Pro 开头两层为 HCA，后续交错 CSA/HCA。CSA 的压缩条目有来源重叠，步长4不等于每条只含4个 token。HCA 读全部可见粗摘要，不执行同样的 top-k。

### “全系列”还包括哪些支线

这里详列通用语言主干的主要公开版本，专用模型另外定位：

| 支线 | 与本表的关系 | 官方入口 |
| --- | --- | --- |
| DeepSeek-Coder / Coder-V2 | 代码训练分支；Coder-V2 有16B/236B、MLA+MoE路线，不能按所有 Coder 都是同一种架构 | [Coder-V2](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct) |
| DeepSeekMath / Math-V2 | 数学推理与验证训练分支；Math-V2 采用 V3.2 模型类 | [Math-V2](https://huggingface.co/deepseek-ai/DeepSeek-Math-V2) |
| DeepSeek-Prover | Lean 形式化证明；Prover-V2 有671B分支，但“V2”不能解读成通用 DeepSeek-V2 的236B主干 | [Prover-V2-671B](https://huggingface.co/deepseek-ai/DeepSeek-Prover-V2-671B) |
| R1-Distill-Qwen / Llama | 蒸馏后的学生模型；沿用各自 Qwen/Llama 主干 | [R1 官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-R1) |
| VL / VL2、Janus、OCR | 视觉理解、生成或文档视觉等独立路线；需看各自视觉编码器和语言主干 | [DeepSeek 官方模型目录](https://huggingface.co/deepseek-ai/models) |

专用支线在这里提供家族入口，不声称已逐一收录所有大小、量化、Base/Instruct 和日期版 checkpoint。

## 8. DeepSeek-V4.1-Flash：这次改变的是执行路径与跨层复用

### 40 层为何分成 20 + 20

V4.1 主干 **40层，D=5120，64个Q heads，1个共享KV head，head_dim=512**；Q低秩宽度1280，输出按8组投影，每组中间宽度1024。模型卡口径为 **552B主干，另有196B Engram**，prefill激活8B、decode激活16B。这些口径不能合成“每 token 固定16B”。

前20层是 causal encoder，后20层是 decoder。decoder 的 global KV 可以从 encoder 最后输出投影得到，减少 prompt 在 decoder 内的完整计算。生成的新 token 仍经过 encoder/decoder 路径。encoder 保持因果性；decoder 的局部 SWA 还需从其自身计算中恢复，不能推断 prompt 完全不用经过任何 decoder 工作。

### CSA2 的三种层角色

| 模式 | 是否生成新的主 KV / indexer K | 本层是否重新选择位置 |
| --- | --- | --- |
| **Full** | 按指定来源生成 | 是 |
| **Reindex** | 复用来源层的缓存 | 是，当前 query 仍可选不同内容 |
| **Reuse** | 复用缓存 | 复用指定来源的 top-k 选择 |

“共享缓存”和“共享选择结果”是两件事。后层可对同一缓存重新选择，也可连选择一起复用。decoder 的 hierarchical indexer 先构建候选池，后续层只在池内筛选，减少逐层扫描全部历史的重复工作。

公开配置给出的 KV 来源层为 `[2,8,14,20]`，索引来源层为 `[2,8,14,20,24,28,32,36]`，这些是**零基编号**。前两个主干槽位纯局部；encoder 后续的压缩步长为2，decoder 的 global来源组织与它不同，不要把 `compress_ratios=1` 当成独立逐层重建普通全历史 KV。数组尾部还有 draft 项，应按40层主干截开阅读。

### 另外四个模块各在做什么

| 模块 | 具体作用与配置 |
| --- | --- |
| **Single-Pass mHC** | 4条 residual streams，改变残差混合与融合执行组织；不是额外增加 attention heads |
| **Engram** | 指定层的局部组合查表；`engram_layer_ids=[1,14]`，最大n-gram阶数4，8个hash heads，每head宽256；这些head不属于attention |
| **DSpark** | 独立训练的半自回归 draft 路径；block_size=5，辅助层字段为3，target层为 `[37,38,39]`；不能照字段名把它直接等同 V3 的预训练 MTP |
| **FP4 main KV + Bounded Replay** | 前者减少主KV每元素的存储字节；后者在prefix复用时恢复局部SWA状态，减少持久化存储。Bounded Replay不是ReplaySSM的同名实现 |

视觉路径是 **DeepSeek-ViT → 两层MLP projector → 文本主干**。vision config为32层、宽1024、16heads、patch14，使用3×3 pixel-unshuffle下采样。原生视觉属于V4.1公开模型，不应反推原版V4同样带此路径。

训练上使用Muon更新主干线性矩阵，对Q/K使用head-wise更新；Engram、token embedding与prediction head采用动量加Sinkhorn balancing，norm与非矩阵参数使用AdamW。这里的Sinkhorn是更新矩阵的行列尺度平衡，与mHC中约束残差混合矩阵的Sinkhorn用途不同。

原版 Flash 与 `Flash-1017` 在本轮核对的主要结构字段一致，仍各自保留 revision。它们是两个发布条目，不能由结构相同推断权重相同。[V4.1报告](https://arxiv.org/html/2609.19969v1)、[官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)

## 9. Qwen3.8：同一品牌下的三种组合

### 主干与两类 attention 的尺寸分别读

| 配置 | 主干 L / D | 序列层组成 | 主 softmax Q / KV heads；head_dim |
| --- | --- | --- | --- |
| **3.8-27B** | 64 / 5120 | 48 GDN + 16 gated full attention | 24 / 4；256 |
| **3.8-2.4T-A95B** | 92 / 8192 | 69 GDN + 23 gated full attention | 64 / 4；256 |
| **3.8-Flash-Next** | 48 / 2560 | 36 GDN + 12 QSA | 24 / 2；256 |

| 配置 | GDN 的 QK / V heads | GDN key / value 宽度 | FFN / 专家 |
| --- | --- | --- | --- |
| 3.8-27B | 16 / 48 | 128 / 128 | dense SwiGLU，intermediate17408 |
| 3.8-2.4T-A95B | 16 / 128 | 128 / 128 | 512 routed选10+1 shared，单专家intermediate2048 |
| 3.8-Flash-Next | 16 / 48 | 128 / 128 | 512 routed选10+1 shared，单专家intermediate640 |

第一张表的 heads 用于 softmax 层，第二张用于 GDN，不能互相代入。GDN 每个 value/state head 对应固定形状的矩阵状态；QK heads 更少时按模型定义分组共享。D也不必等于Q heads×head_dim，例如2.4T版本的主Q投影宽度为16384，主干hidden仍为8192。

### 2.4T-A95B 与 Flash-Next 的差别

| 部件 | 2.4T-A95B | Flash-Next |
| --- | --- | --- |
| 长程读取 | gated full GQA | **QSA**，indexer挑micro-block，主GQA读取选中块的原始KV |
| Indexer | 未采用QSA这套indexer | 4个Q heads、1个共享K head、宽128；块大小4，预算2048 token，即最多512完整块，另处理未满块尾部 |
| Residual | 普通pre-norm residual add；attention output gate不等于GR | **GR 4分支**：逐特征读门、每分支标量写门 |
| N-gram参数表 | 未列该模块 | **51B额外参数**；bigram/trigram，第2层注入；hash heads=8，不是attention heads |
| 主干参数口径 | 约2.4T总量、95B激活 | 125B主干、6B激活，另有51B表和4B MTP |
| MTP | 1层，多步训练 | 1层，多步训练 |
| 视觉 | 公开权重为纯文本；带视觉托管Max是另一发布对象 | 公开权重带视觉编码器 |

两种配置都把多数序列层换为GDN，但Flash-Next又改变少数全局层、残差以及参数记忆。它不是2.4T简单缩小后的同构模型。51B表主要扩大可访问的参数容量，640则是专家中间宽度；二者不是同一轴。

27B与Flash-Next的公开vision config均为27层、宽1152、16heads、patch16，输出分别映射到5120/2560。三者text config的默认最大位置为262144；模型卡另给YaRN扩展到约1M的方法。这是默认配置与扩展方案的区别，不应不加条件地统一写成默认1M。

普通27B/2.4T的主要结构来自其各自配置；Flash-Next报告中的Muon、GR与稳定性实验不能推广成常规3.8所有分支的完整训练配方。[Qwen官方模型目录](https://huggingface.co/Qwen/models)、[Flash-Next报告](https://arxiv.org/html/2608.30320v1)

## 10. Kimi K3 与 K3.1：哪些已经有配置，哪些仍以公告为准

### K3 的可核验尺寸

| 对象 | 公开配置 |
| --- | --- |
| 主干 | 93层，D=7168；约2.8T总参数 / 104B激活 |
| 序列层 | 69 KDA + 24 gated MLA；MLA位于1基编号4、8…92、93层 |
| KDA | 96heads，每head宽128；短卷积kernel4，full-rank gate，衰减下界设计 |
| MLA | 96个Q heads；Q低秩1536、共同KV latent512；内容K128、额外key分支64、V128；`mla_use_nope=true`、output gate开启 |
| FFN | 首层dense，intermediate33792；后续Stable LatentMoE |
| Routed experts | 896选16；工作宽度3584，单专家intermediate3072 |
| Shared experts | 2个，直接在主干全宽路径工作 |
| Residual | Block AttnRes，深度块大小12；层分组而非序列窗口 |
| Activation / training | SiTU-GLU、聚合后RMSNorm、Quantile Balancing、Per-Head Muon、QAT |
| Vision / context | MoonViT-V2，原生视觉；配置max_position_embeddings=1048576 |
| MTP / draft | 报告描述预训练MTP后微调draft；此发布主干 `num_nextn_predict_layers=0` |

MLA的NoPE表示此分支不执行相应位置旋转，不表示它与KDA都失去顺序信息，也不能仅因字段名 `qk_rope_head_dim=64` 就断言一定运行RoPE。KDA的heads也不是96个独立的历史KV缓存。

### Latent expert 的两层宽度变化

```text
主干 token [7168]
   ├────────────────────────→ 2个全宽shared experts ───────┐
   ↓ 共享下投影                                             │
latent [3584]                                              │
   ↓ 16个选中routed experts，各自3584 → 3072 → 3584           │
按路由权重合并 → RMSNorm → 上投影 → [7168] ──────────────────┴→ FFN输出
```

3072是专家内部的intermediate，不是主干hidden；它也不一定比3584更宽。因此FFN不能一概描述为“先扩维再降维”，这里应读实际投影尺寸。SiTU-GLU还有gate/up两路，图只概括宽度路径。

### K3.1 Pro 的确认边界

截至 **2026-11-02**，[官方公告](https://www.kimi.com/en/blog/kimi-k3-1-pro)确认：

- K3.1系列首个模型为 **K3.1 Pro**，另有 **Pro Highspeed**。
- 它是 **K3的后训练版本**，增加后训练计算，支持3T级、原生视觉和1M上下文。
- 官方承诺完整权重在 **2026-11-04之前/当天**发布；本轮未核验到可独立固定revision的K3.1配置。

因此，K3的93层、96heads、896专家可以作为理解其基础主干的参照，但本页不把它们标成已经从K3.1配置逐项验证的数字。Highspeed的服务加速声明，也不能据此推断它删层、减heads、改用MQA或更换模型权重。

## 11. 特殊技术放在什么位置：避免同名误认

| 技术 | 作用位置 | 当前这些模型里的归属 |
| --- | --- | --- |
| MLA | 历史KV的表示宽度 | DeepSeek V2–V3.2、K3的gated MLA层 |
| DSA / QSA | 主attention之前的选择器 | V3.2按token；Flash-Next按micro-block |
| CSA / HCA | 序列压缩与历史读取 | 原版V4，搭配局部SWA；HCA没有同样的top-k |
| CSA2 | 跨层缓存、索引与候选组织 | V4.1，不能仅理解为CSA换压缩率 |
| GDN / KDA | 固定形状状态递推 | Qwen3.8各分支 / K3；混合模型仍有增长的KV缓存 |
| mHC / Single-Pass mHC | 扩展的残差流与混合 | V4 / V4.1 |
| GR | 扩展残差的门控读写 | Qwen Flash-Next |
| Block AttnRes | 同一token沿深度的候选加权 | K3；K3.1为后训练继承关系，独立配置待核验 |
| LatentMoE | routed专家输入输出的工作宽度 | K3 |
| Engram / n-gram embedding | 根据局部token组合查参数向量 | V4.1 / Qwen Flash-Next |
| Hash routing | 决定执行哪些专家 | 原版V4前3层；与n-gram表的结果对象不同 |
| MTP / DSpark | 预测、draft和验证路径 | 多个家族有MTP；V4.1专门使用DSpark |
| ReplaySSM / SWA Bounded Replay | 递推状态或局部缓存的执行/恢复方案 | 前者针对SSM状态管理；后者是V4.1局部KV恢复；都不是新的attention head布局 |

## 12. 本轮固定配置来源

下面保存 **2026-11-02** 重新获取的checkpoint revision。原有的2026-10-28快照保留，新增[完整配置快照](../assets/model-catalog-configs.json)供查字段。未下载模型权重、未跑CUDA或测量性能。

| 官方 checkpoint | 固定revision | 配置 / 模型卡 |
| --- | --- | --- |
| deepseek-ai/deepseek-llm-7b-base | `7683fea62d` | [config](https://huggingface.co/deepseek-ai/deepseek-llm-7b-base/blob/7683fea62db869066ddaff6a41d032262c490d4f/config.json) / [card](https://huggingface.co/deepseek-ai/deepseek-llm-7b-base/blob/7683fea62db869066ddaff6a41d032262c490d4f/README.md) |
| deepseek-ai/deepseek-llm-67b-base | `c3f813a112` | [config](https://huggingface.co/deepseek-ai/deepseek-llm-67b-base/blob/c3f813a1121c95488a20132d3a4da89f4a46452f/config.json) / [card](https://huggingface.co/deepseek-ai/deepseek-llm-67b-base/blob/c3f813a1121c95488a20132d3a4da89f4a46452f/README.md) |
| deepseek-ai/deepseek-moe-16b-base | `521d2bc4fb` | [config](https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/blob/521d2bc4fb69a3f3ae565310fcc3b65f97af2580/config.json) / [card](https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/blob/521d2bc4fb69a3f3ae565310fcc3b65f97af2580/README.md) |
| deepseek-ai/DeepSeek-V2-Lite | `604d5664dd` | [config](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/604d5664dddd88a0433dbae533b7fe9472482de0/config.json) / [card](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/604d5664dddd88a0433dbae533b7fe9472482de0/README.md) |
| deepseek-ai/DeepSeek-V2 | `4461458f18` | [config](https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/4461458f186c35188585855f28f77af5661ad489/config.json) / [card](https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/4461458f186c35188585855f28f77af5661ad489/README.md) |
| deepseek-ai/DeepSeek-V2.5 | `c85b5ede86` | [config](https://huggingface.co/deepseek-ai/DeepSeek-V2.5/blob/c85b5ede86f2a598af339624cac5723861e557ed/config.json) / [card](https://huggingface.co/deepseek-ai/DeepSeek-V2.5/blob/c85b5ede86f2a598af339624cac5723861e557ed/README.md) |
| deepseek-ai/DeepSeek-V3 | `e815299b0b` | [config](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/config.json) / [card](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/README.md) |
| deepseek-ai/DeepSeek-V3-0324 | `e9b33add76` | [config](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324/blob/e9b33add76883f293d6bf61f6bd89b497e80e335/config.json) / [card](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324/blob/e9b33add76883f293d6bf61f6bd89b497e80e335/README.md) |
| deepseek-ai/DeepSeek-R1 | `56d4cbbb4d` | [config](https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad/config.json) / [card](https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad/README.md) |
| deepseek-ai/DeepSeek-R1-0528 | `4236a6af53` | [config](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528/blob/4236a6af538feda4548eca9ab308586007567f52/config.json) / [card](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528/blob/4236a6af538feda4548eca9ab308586007567f52/README.md) |
| deepseek-ai/DeepSeek-V3.1 | `c0781d039f` | [config](https://huggingface.co/deepseek-ai/DeepSeek-V3.1/blob/c0781d039fb7a1ba2abc4add0bdc293e92d2b8db/config.json) / [card](https://huggingface.co/deepseek-ai/DeepSeek-V3.1/blob/c0781d039fb7a1ba2abc4add0bdc293e92d2b8db/README.md) |
| deepseek-ai/DeepSeek-V3.1-Terminus | `19510d6dc6` | [config](https://huggingface.co/deepseek-ai/DeepSeek-V3.1-Terminus/blob/19510d6dc61f79dbd925bd51ee8a9081c509a4b6/config.json) / [card](https://huggingface.co/deepseek-ai/DeepSeek-V3.1-Terminus/blob/19510d6dc61f79dbd925bd51ee8a9081c509a4b6/README.md) |
| deepseek-ai/DeepSeek-V3.2-Exp | `194c67e12b` | [config](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/blob/194c67e12b1b0d6df0ef373ddcf215bc84027409/config.json) / [card](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/blob/194c67e12b1b0d6df0ef373ddcf215bc84027409/README.md) |
| deepseek-ai/DeepSeek-V3.2 | `a7e62ac04e` | [config](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/a7e62ac04ecb2c0a54d736dc46601c5606cf10a6/config.json) / [card](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/a7e62ac04ecb2c0a54d736dc46601c5606cf10a6/README.md) |
| deepseek-ai/DeepSeek-V3.2-Speciale | `c562883eda` | [config](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Speciale/blob/c562883eda916fd699a44bbf1f4987a53b96008b/config.json) / [card](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Speciale/blob/c562883eda916fd699a44bbf1f4987a53b96008b/README.md) |
| deepseek-ai/DeepSeek-V4-Flash | `60d8d70770` | [config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/config.json) / [card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/README.md) |
| deepseek-ai/DeepSeek-V4-Pro | `b5968e9190` | [config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/config.json) / [card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/README.md) |
| deepseek-ai/DeepSeek-V4.1-Flash | `314113f8a7` | [config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/config.json) / [card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/README.md) |
| deepseek-ai/DeepSeek-V4.1-Flash-1017 | `1bae0e1088` | [config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/config.json) / [card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/README.md) |
| moonshotai/Kimi-K3 | `f831ab6681` | [config](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/config.json) / [card](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/README.md) |
| Qwen/Qwen3.8-27B | `1d4bf0f2ff` | [config](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json) / [card](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/README.md) |
| Qwen/Qwen3.8-2.4T-A95B | `207bd685a7` | [config](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/blob/207bd685a7e3696cfaff12ded7c6a7ea0f88c996/config.json) / [card](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/blob/207bd685a7e3696cfaff12ded7c6a7ea0f88c996/README.md) |
| Qwen/Qwen3.8-Flash-Next | `de4b8e4d43` | [config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json) / [card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/README.md) |

K3.1以[官方发布公告](https://www.kimi.com/en/blog/kimi-k3-1-pro)为来源，不虚构checkpoint revision。机制推导继续看[模型结构详解](model-architectures.md)、[GR/AttnRes/Engram详解](residual-experts-memory.md)与[DeepSeek逐代演进](../DeepSeek/evolution.md)。
