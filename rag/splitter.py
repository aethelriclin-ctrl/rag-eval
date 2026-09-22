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
