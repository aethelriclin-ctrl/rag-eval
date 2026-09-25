"""文本切分模块（RAG 的第 ① 步）。

RAG 的第一步是把长文档切成小块（chunk），因为：
    1. 检索的单位是"块"，不是"整篇文档" —— 整篇塞进去模型会被无关内容干扰
    2. 模型有上下文长度限制
    3. 块越小检索越精准，但太小会丢上下文 —— 所以要折中

两个关键参数：
    CHUNK_SIZE：每块最多多少字
    OVERLAP   ：相邻块共享多少字

⚠️ 为什么必须有重叠（这是 RAG 最常见的坑）：
    一个答案可能刚好横跨两段。如果切得干干净净，
    检索时"两边都只拿到半句"，谁也答不上来。
"""

import os
import glob

CHUNK_SIZE = 500     # 每块最多 500 字
OVERLAP = 50         # 相邻块重叠 50 字

# ⚠️ 块大小对照实验（2026-09-22）—— **500 是对的，250 更差**
#
#   假设：诊断发现答案要点全在索引里、但 top-5 只覆盖 80%，缺的都是"枚举列表"
#        （5 类评分器 / 4 类场景 / **四类归因**），怀疑 500 字的块把表格切散了。
#        ⚠️ 原文这里写的是"六类归因"——**写错了**：知识库口径是【四类】归因，
#           "六条"指的是失败案例的记录条数。
#   实验：只改 CHUNK_SIZE 500 → 250，重建索引后重测（重叠保持 50 不动）。
#   结果：**假设被证伪**
#        · 块数 103 → 230
#        · ⭐ top-5 覆盖　80% → **70%**（17 题 / 44 个要点的全量统计）
#        · 答案正确率　　58.3% → **33.3%**（⚠️ **n=3 的试点跑测**，不是 17 题全量）
#        · 含答案块中位排名　1 → 2，最大 74 → 141
#        · q03 的四个归因要点从"第 1 名"一起退到"第 8 名"
#
#   结论：**块越小，语义越碎** —— BM25 靠字面匹配，块被切碎后，
#        同一段内容里的词被分散到更多块，每块的"信号密度"下降，反而更难被排上来。
#        所以在"字面匹配 + 2-gram"这个组合下，**大块反而更稳**。
#
#   真正的教训：**我怀疑的原因（表格被切散）不是主因。**
#        主因更可能是——BM25 的 2-gram 打分对"哪几类 / 哪些 / 怎么"这类
#        **问句用词给的 IDF 权重过高**，把真正的关键词挤下去了。
#        这是"字面匹配"的固有边界，不是调参数能解决的 → 见 README「下一步」的向量检索对照。


def load_doc(path):
    """读一个文件，返回纯文本。

    ⚠️ 必须写 encoding="utf-8" —— Windows 上默认按 GBK 解码，
       中文会乱码。项目 2 踩过这个坑（用 PowerShell 扫中文标记时全变成 0）。
    """
    with open(path, encoding="utf-8") as f:
        return f.read()


def split_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """把长文本切成若干块。

    思路：
        1. 先按空行切段（\\n\\n）—— 段落是语义的天然边界
        2. 单段超过 chunk_size 的，硬切
        3. 把小段依次塞进"当前块"，塞满了就存起来、开新块
        4. 开新块时，把上一块的**最后 overlap 个字**放进来当开头（这就是重叠）
    """
    # ---- 第 1 步：按空行切段，顺便去掉每段两端空白 ----
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # ---- 第 2 步：把超长的段落硬切 ----
    pieces = []
    for p in paragraphs:
        if len(p) <= chunk_size:
            pieces.append(p)
        else:
            for i in range(0, len(p), chunk_size):
                pieces.append(p[i:i + chunk_size])

    # ---- 第 3、4 步：装箱 + 加重叠 ----
    # ⚠️ 关键点：重叠要**从 chunk_size 的预算里扣**，不是额外加上去。
    #    否则"480 字的新内容 + 50 字重叠 = 530 字"，箱子就爆了。
    #    （这个 bug 是自测发现的：第 3、24、26、28 块超长。）
    chunks = []
    current = ""
    for piece in pieces:
        # 计算"这一块的头"能留多少重叠：新内容越接近上限，能留的重叠越少
        room = max(0, chunk_size - len(piece) - 1)      # -1 是那半个换行符的位置
        head_budget = min(overlap, room)

        if not current:
            current = piece
        elif len(current) + len(piece) + 1 <= chunk_size:
            current += "\n" + piece                      # 装得下，直接拼
        else:
            chunks.append(current)                       # 装不下了，存起来
            tail = current[-head_budget:] if head_budget > 0 else ""
            current = (tail + "\n" + piece) if tail else piece

    if current:
        chunks.append(current)

    return chunks


def load_all_docs(docs_dir):
    """加载 docs 目录下所有 .md 和 .json，切分后返回 [(来源文件名, 块内容), ...]。

    为什么每块要记来源文件名：
        第 2 层要测"引用准确性" —— 答案有没有真的来自检索到的片段？
        那就必须知道每块是从哪个文件来的。
    """
    results = []
    for pattern in ("*.md", "*.json"):
        for path in sorted(glob.glob(os.path.join(docs_dir, pattern))):
            name = os.path.basename(path)
            text = load_doc(path)
            for chunk in split_text(text):
                results.append((name, chunk))
    return results


if __name__ == "__main__":
    """自测：python splitter.py —— 打印切分结果，人工看一眼合不合理。"""
    docs_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")

    sample = os.path.join(docs_dir, "项目1-README.md")
    if not os.path.exists(sample):
        print(f"找不到样本文件：{sample}\n请先运行：python setup.py")
    else:
        text = load_doc(sample)
        chunks = split_text(text)
        print(f"文件总字数：{len(text)}")
        print(f"切成 {len(chunks)} 块\n")

        for i, c in enumerate(chunks[:3], 1):     # 只看前 3 块，别刷屏
            print(f"--- 第 {i} 块（{len(c)} 字）---")
            print(c[:200] + ("..." if len(c) > 200 else ""))
            print()

        # 验收 1：每块长度不超过 chunk_size
        over = [i for i, c in enumerate(chunks) if len(c) > CHUNK_SIZE]
        print("✅ 每块都没超长" if not over else f"⚠️ 第 {over} 块超长了")

        # 验收 2：重叠是否生效
        if len(chunks) >= 2:
            tail, head = chunks[0][-OVERLAP:], chunks[1][:OVERLAP]
            print(f"第 1 块结尾：{tail!r}")
            print(f"第 2 块开头：{head!r}")
            print("✅ 重叠生效" if tail == head else "⚠️ 重叠没对上")
