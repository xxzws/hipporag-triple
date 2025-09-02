# HippRAG Model

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# ALL IN HF https://huggingface.co/xxzws/hipporag-triple

## Introduction / 介绍


**English:**  
This model is designed for extracting Chinese HippRAG triples (subject-predicate-object). It is trained exclusively on Chinese corpora but can be extended to simulate other languages via the provided training code. Based on Qwen3-1.7B, it supports extension to other models in the Qwen series; compatibility with other model families remains untested. The model also handles recognition of formats such as Markdown (MD) and LaTeX.

**中文：**  
该模型专为提HippRAG取中文三元组（主体-谓词-客体），仅使用中文语料进行训练，但可通过提供的训练代码扩展至模拟其他语言。基于Qwen3-1.7B，可扩展至Qwen系列的其他模型；与其他模型族的兼容性尚未验证。可支持Markdown（MD）、LaTeX等数据格式的识别。

## Usage / 使用

**English:**  
The invocation method aligns with Qwen3 (refer to [Qwen3 Documentation](https://huggingface.co/Qwen)). Due to partial incompatibility of Transformers with certain inference environments, generation may continue indefinitely; it is advisable to incorporate a stop token like `"}]}` as a safeguard.

**中文：**  
调用方式与Qwen3相同（参见[Qwen3文档](https://huggingface.co/Qwen)）。由于Transformers与部分推理环境的不完全兼容，可能导致生成无休止，建议添加停止符`"}]}`作为双重保障。

## Training / 训练

**English:**  
Given the lack of provided data, the model's performance is moderate. You can enhance it through further training using the code below (involving two datasets: one simple and one challenging).

**中文：**  
由于缺乏外部数据支持，模型效果中等。可使用以下代码进行增量训练（涉及两个数据集：一个简单，一个较复杂）。

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Windows 版：自定义复合损失 + 5轮课程学习（简单→复杂）
- 兼容 Windows 路径（Pathlib）
- 避免 Windows DataLoader 多进程问题（num_workers=0）
- 自动补 pad_token_id
- device_map="auto"（有 CUDA 则走 GPU）
"""

import os
import json
import math
import torch
import torch.nn.functional as F
from datetime import datetime
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from accelerate import Accelerator
from pathlib import Path

# ========== 环境建议 ==========
# Windows 上建议显式关闭多核分词的线程提示
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ========== 全局配置（按需修改） ==========
# 模型路径（Windows 示例）
MODEL_PATH  = r"H:\model\qwen3"

# 训练数据（把这两个改成你的本机路径）
TRAIN_FILE  = r"H:\data\train_fixed.json"
PARA_FILE   = r"H:\data\paragraph_train.json"

# 输出目录（时间戳）
OUTPUT_ROOT = Path(r"H:\model") / f"qwen3_custom_ft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

print("🚀 当前可见 GPU 数量:", torch.cuda.device_count())

# ========== 主流程 ==========
def step4_with_curriculum():
    print("=== Step4: 自定义复合损失 + 5轮课程学习（Windows 版） ===")
    out_dir = OUTPUT_ROOT / "step4_custom_curriculum"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载分词器与模型
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        # 没有 pad_token 就用 eos 兜底
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else "</s>"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        device_map="auto"  # 有 CUDA 会自动放 GPU
    )
    model.train()
    for p in model.parameters():
        if torch.is_floating_point(p):
            p.requires_grad = True

    # ========== 数据准备：simple_raw / complex_raw（再切两半）==========
    # Windows 路径用字符串即可，datasets 内部兼容
    orig_ds = load_dataset("json", data_files={"orig": str(TRAIN_FILE)})["orig"]
    para_ds = load_dataset("json", data_files={"para": str(PARA_FILE)})["para"].shuffle(seed=42)

    half = (len(para_ds) - len(orig_ds)) // 2 if (len(para_ds) > len(orig_ds)) else len(para_ds) // 2
    simple_raw  = concatenate_datasets([orig_ds, para_ds.select(range(half))])
    complex_raw = para_ds.select(range(half, len(para_ds)))

    # 将 complex 再切两半：c1 / c2
    c_half = len(complex_raw) // 2
    complex1_raw = complex_raw.select(range(c_half))
    complex2_raw = complex_raw.select(range(c_half, len(complex_raw)))
    complex_all_raw = concatenate_datasets([complex1_raw, complex2_raw])
    all_raw = concatenate_datasets([simple_raw, complex_all_raw])

    # ========== Prompt 构造 & 预处理 ==========
    INSTR = (
        "请从以下文本中抽取三元组，输出格式为标准JSON数组：\n"
        "请务必严格输出JSON，不要附加说明文字。\n"
        "字段: subject=主体, predicate=关系, object=客体；请尽可能提取所有相关关系且不要混淆主体与客体。\n\n"
    )
    def build_prompt(text):
        return f"<|user|>\n{INSTR}{text}\n<|assistant|>\n"

    MAX_LEN = 1024

    def preprocess(ex):
        # 兼容 input/output 可能是非字符串的情况
        src_inp = ex.get("input", "")
        tgt_out = ex.get("output", "")
        if not isinstance(src_inp, str):
            src_inp = str(src_inp)
        if not isinstance(tgt_out, str):
            tgt_out = json.dumps(tgt_out, ensure_ascii=False)

        prompt = build_prompt(src_inp)
        full   = prompt + tgt_out

        tok    = tokenizer(
            full,
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        ids  = tok.input_ids[0]
        mask = tok.attention_mask[0]
        labels = ids.clone()

        # 计算 prompt 长度，屏蔽其 loss
        plen = tokenizer(prompt, return_tensors="pt").input_ids.size(1)
        labels[:plen] = -100

        # predicate 掩码（朴素 token 匹配）
        pmask = torch.zeros_like(ids, dtype=torch.bool)
        try:
            # 这里 ex["output"] 若不是 JSON 字符串，会在上面改成字符串
            preds = [t["predicate"] for t in json.loads(tgt_out)]
            tokens = tokenizer.convert_ids_to_tokens(ids)
            for pred in preds:
                toks = tokenizer.tokenize(pred)
                L = len(toks)
                if L == 0:
                    continue
                for i in range(len(tokens) - L + 1):
                    if tokens[i:i+L] == toks:
                        pmask[i:i+L] = True
        except Exception:
            pass

        return {
            "input_ids": ids,
            "attention_mask": mask,
            "labels": labels,
            "predicate_mask": pmask
        }

    accel = Accelerator()

    # Windows 上 datasets 的 map 默认单进程即可（避免多进程 spawn 麻烦）
    with accel.main_process_first():
        simple      = simple_raw.map(preprocess, remove_columns=simple_raw.column_names)
        complex1    = complex1_raw.map(preprocess, remove_columns=complex1_raw.column_names)
        complex2    = complex2_raw.map(preprocess, remove_columns=complex2_raw.column_names)
        complex_all = complex_all_raw.map(preprocess, remove_columns=complex_all_raw.column_names)
        all_ds      = all_raw.map(preprocess, remove_columns=all_raw.column_names)

    for ds in (simple, complex1, complex2, complex_all, all_ds):
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "predicate_mask"])

    # DataLoader：Windows 下稳妥用单进程
    bs = 4
    num_workers = 0  # ★ Windows：0 最稳，避免多进程卡死
    dl_args = dict(batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    simple_loader      = DataLoader(simple,      **dl_args)
    complex1_loader    = DataLoader(complex1,    **dl_args)
    complex2_loader    = DataLoader(complex2,    **dl_args)
    complex_all_loader = DataLoader(complex_all, **dl_args)
    all_loader         = DataLoader(all_ds,      **dl_args)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    (model, optimizer,
     simple_loader, complex1_loader, complex2_loader, complex_all_loader, all_loader
    ) = accel.prepare(model, optimizer,
                      simple_loader, complex1_loader, complex2_loader, complex_all_loader, all_loader)

    # ========== 训练参数 ==========
    alpha, beta, delta = 1.0, 1.0, 0.2
    grad_accum = 4
    rounds = 8  # 固定 8 轮（你原注释写 5 轮，代码里是 8，我保持 8）

    # ========== 训练子流程 ==========
    def train_progressive_mix(loader_s, loader_c, round_idx):
        """第1轮：简单→复杂概率线性上升"""
        total_steps = max(len(loader_s), len(loader_c))
        it_s, it_c = iter(loader_s), iter(loader_c)
        total_loss, step_count = 0.0, 0
        for step in tqdm(range(total_steps), desc=f"Round {round_idx+1} (progressive mix)"):
            p = (step + 1) / total_steps
            pick_complex = torch.rand(1).item() < p
            if pick_complex:
                try:
                    batch = next(it_c)
                except StopIteration:
                    it_c = iter(loader_c)
                    batch = next(it_c)
            else:
                try:
                    batch = next(it_s)
                except StopIteration:
                    it_s = iter(loader_s)
                    batch = next(it_s)

            loss = compute_loss(model, batch, tokenizer, alpha, beta-0.5, delta)
            accel.backward(loss)
            if (step + 1) % grad_accum == 0:
                accel.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad()
            total_loss += loss.item()
            step_count += 1
        return total_loss, step_count

    def train_uniform_among_loaders(loaders, round_idx):
        """第2/3轮：按数据源均等采样（轮转）"""
        k = len(loaders)
        max_len = max(len(l) for l in loaders)
        steps = max_len * k
        iters = [iter(l) for l in loaders]
        total_loss, step_count = 0.0, 0

        for step in tqdm(range(steps), desc=f"Round {round_idx+1} (uniform across {k} sources)"):
            idx = step % k
            try:
                batch = next(iters[idx])
            except StopIteration:
                iters[idx] = iter(loaders[idx])
                batch = next(iters[idx])

            loss = compute_loss(model, batch, tokenizer, alpha, beta-0.3, delta)
            accel.backward(loss)
            if (step + 1) % grad_accum == 0:
                accel.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad()
            total_loss += loss.item()
            step_count += 1
        return total_loss, step_count

    def train_single_loader(loader, round_idx):
        """第4/5+轮：全量顺序训练"""
        total_loss, step_count = 0.0, 0
        for step, batch in enumerate(tqdm(loader, desc=f"Round {round_idx+1} (full data)")):
            loss = compute_loss(model, batch, tokenizer, alpha, beta, delta)
            accel.backward(loss)
            if (step + 1) % grad_accum == 0:
                accel.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad()
            total_loss += loss.item()
            step_count += 1
        return total_loss, step_count

    # ========== 五（八）轮课程学习 ==========
    for r in range(rounds):
        if r == 0:
            tot, cnt = train_progressive_mix(simple_loader, complex_all_loader, r)
        elif r == 1:
            tot, cnt = train_uniform_among_loaders([simple_loader, complex1_loader], r)
        elif r == 2:
            tot, cnt = train_uniform_among_loaders([simple_loader, complex1_loader, complex2_loader], r)
        else:
            tot, cnt = train_single_loader(all_loader, r)

        avg_loss = tot / max(1, cnt)
        print(f"✅ Round {r+1} avg loss: {avg_loss:.4f}")

    # 保存
    if accel.is_main_process:
        unwrapped = accel.unwrap_model(model)
        unwrapped.save_pretrained(str(out_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(out_dir))
    print("💾 保存至", out_dir)

# ========== 损失函数 ==========
def compute_loss(model, batch, tokenizer, alpha, beta, delta):
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"]
    )
    ce_loss = outputs.loss

    # F1 on predicate tokens（embedding 近似 P/R）
    pred_ids = outputs.logits.argmax(dim=-1)
    mask_flat = batch["predicate_mask"].view(-1)
    labels_flat = batch["labels"].view(-1)
    pred_flat = pred_ids.view(-1)
    valid_idx = mask_flat.nonzero(as_tuple=True)[0]

    if valid_idx.numel() > 0:
        true_ids = labels_flat[valid_idx]
        pred_sel = pred_flat[valid_idx]
        emb = model.get_input_embeddings()
        vocab_sz = emb.num_embeddings
        legal = (
            (true_ids >= 0) & (true_ids < vocab_sz) &
            (pred_sel >= 0) & (pred_sel < vocab_sz)
        )
        if legal.sum() > 0:
            true_ids = true_ids[legal]
            pred_sel = pred_sel[legal]
            t_emb = emb(true_ids)
            p_emb = emb(pred_sel)
            S = F.cosine_similarity(t_emb.unsqueeze(1), p_emb.unsqueeze(0), dim=-1)
            P_val = S.max(dim=1).values.mean()
            R_val = S.max(dim=0).values.mean()
            F1 = 2 * P_val * R_val / (P_val + R_val + 1e-8)
        else:
            F1 = torch.tensor(1.0, device=ce_loss.device)
    else:
        F1 = torch.tensor(1.0, device=ce_loss.device)

    # 非结构输出惩罚
    illegal = (batch["labels"] == -100) & (pred_ids != tokenizer.pad_token_id)
    x = illegal.sum().float().clamp(min=0.0)
    penalty = 1.0 - 1.0 / torch.log(x + 10.0)

    return alpha * ce_loss + beta * (1 - F1) + delta * penalty

# ========== 入口 ==========
if __name__ == "__main__":
    # Windows 下推荐：
    # 1) Python 3.10/3.11 + torch/cu 版本匹配
    # 2) 先把 TRAIN_FILE / PARA_FILE 改成你的真实路径
    step4_with_curriculum()
print("🎉 完成")
```

## Training Logic / 训练逻辑

**English:**  
The training adopts a curriculum learning strategy across 8 rounds, incorporating a composite loss function. Denote the simple dataset as $D_s$, the halves of the complex dataset as $D_{c1}$ and $D_{c2}$, with $D_c = D_{c1} \cup D_{c2}$, and $D_a = D_s \cup D_c$.

- **Round 1:** Progressive mixing: For each step $t = 1$ to $T = \max(|D_s|, |D_c|)$, sample from $D_c$ with probability $p_t = t / T$, otherwise from $D_s$. Loss: $L = \alpha \cdot L_{CE} + (\beta - 0.5) \cdot (1 - F1_p) + \delta \cdot P$, where $L_{CE}$ is cross-entropy loss, $F1_p$ approximates the F1-score on predicate tokens using cosine similarity of embeddings, and $P = 1 - 1 / \log(x + 10)$ penalizes non-structured outputs with $x$ being the count of illegal tokens.
- **Round 2:** Uniform sampling across $\{D_s, D_{c1}\}$: Cycle through loaders for $T = 2 \cdot \max(|D_s|, |D_{c1}|)$ steps, using $\beta - 0.3$.
- **Round 3:** Uniform across $\{D_s, D_{c1}, D_{c2}\}$: Similarly, $T = 3 \cdot \max$ over the three, using $\beta - 0.3$.
- **Rounds 4-8:** Full sequential training on $D_a$, employing the full $\beta$.

Optimization: AdamW with learning rate $5 \times 10^{-5}$, gradient accumulation every 4 steps, and clipping at 1.0. Parameters: $\alpha=1.0$, $\beta=1.0$, $\delta=0.2$.

**中文：**  
训练采用8轮课程学习策略，结合复合损失函数。设简单数据集为$D_s$，复杂数据集的两半为$D_{c1}$和$D_{c2}$，$D_c = D_{c1} \cup D_{c2}$，$D_a = D_s \cup D_c$。

- **第1轮：** 渐进混合：对于每个步$t = 1$到$T = \max(|D_s|, |D_c|)$，以概率$p_t = t / T$从$D_c$采样，否则从$D_s$。损失：$L = \alpha \cdot L_{CE} + (\beta - 0.5) \cdot (1 - F1_p) + \delta \cdot P$，其中$L_{CE}$为交叉熵损失，$F1_p$通过嵌入余弦相似度近似谓词token的F1分数，$P = 1 - 1 / \log(x + 10)$惩罚非结构输出（$x$为非法token数）。
- **第2轮：** 在$\{D_s, D_{c1}\}$上均匀采样：循环加载器$T = 2 \cdot \max(|D_s|, |D_{c1}|)$步，使用$\beta - 0.3$。
- **第3轮：** 在$\{D_s, D_{c1}, D_{c2}\}$上均匀采样：类似，$T = 3 \cdot \max$三者，使用$\beta - 0.3$。
- **第4-8轮：** 在$D_a$上全量顺序训练，使用完整$\beta$。

优化：AdamW，学习率$5 \times 10^{-5}$，每4步梯度累积，裁剪1.0。参数：$\alpha=1.0$，$\beta=1.0$，$\delta=0.2$。
