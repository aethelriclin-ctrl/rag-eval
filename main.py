"""项目 4 第 1 层：最小 RAG 问答系统（命令行版）。

当前检索方式：**BM25（字面匹配）**
    —— 换向量检索只改 rag/factory.py 里的 RETRIEVER 一行。

用法：
    python setup.py          # ① 复制项目文档进 docs/（只需一次）
    python main.py --build   # ② 建索引（切分 + 建检索结构，只需一次）
    python main.py --test    # ③ 冒烟测试（5 个问题）
    python main.py           # ④ 交互式提问

⚠️ 索引要重建的时机：
    改了 docs/ 里的文档 → 必须重跑 --build。
    这是项目 3 的教训：**产物和源不一致时，结论就不可信。**
"""

import os
import sys

from rag import factory, generator, retriever


def get_index():
    r = factory.load()
    return r


def ask(question, index, top_k=3, show_context=False):
    """完整走一遍：问题 → 检索 → 组装上下文 → 生成。"""
    hits = index.retrieve(question, top_k=top_k)
    context = retriever.build_context(hits)

    if show_context:
        print("\n--- 检索命中的资料 ---")
        for i, h in enumerate(hits, 1):
            print(f"[{i}] {h['source']}   分数 {h['score']:.4f}")
            print(f"    {h['text'][:120]}...")
        print("--- 资料结束 ---\n")

    answer, usage = generator.generate(question, context)

    print(f"回答：{answer}")
    print(f"（用量：输入 {usage['tokens_in']} / 输出 {usage['tokens_out']} tokens）")
    print(f"检索命中：{', '.join(h['source'] for h in hits)}")
    return answer, hits


def smoke_test(index):
    """冒烟测试：5 个问题 —— 前三个能答，后两个库里没有。

    ⭐ 这套设计直接给第 2 层用：
       第 4、5 题是"幻觉探测器" —— 库里没有答案时，它会不会编？
       而第 1、2 题还能顺便看出 BM25 的强项（含精确词，字面能匹配）。
    """
    questions = [
        ("能答·含精确词", "我的评测项目一共几道题？"),
        ("能答·含精确词", "什么是六类失败归因机制？"),
        ("跨段/改写问法", "我做过哪几个项目，分别是什么方向？"),
        ("切知识库没有", "我的项目的作者是谁？"),
        ("肯定没有", "我的项目用了哪家公司的 GPU？"),
    ]
    print("=" * 64)
    print("冒烟测试（当前检索：BM25 字面匹配）")
    print("=" * 64)
    for kind, q in questions:
        print(f"\n【{kind}】{q}")
        ask(q, index, show_context=True)


def main():
    args = sys.argv[1:]

    if "--build" in args:
        factory.build()
        return

    index = get_index()
    if index is None:
        return

    if "--test" in args:
        smoke_test(index)
        return

    n = len(index.items)
    print("=" * 64)
    print(f"RAG 问答已就绪（BM25 检索，索引 {n} 块）")
    print("输入问题回车；输入 q 退出；问题前加 :ctx 可看检索到的资料")
    print("=" * 64)
    while True:
        try:
            q = input("\n问：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() == "q":
            break
        show_ctx = q.startswith(":ctx")
        if show_ctx:
            q = q[4:].strip()
        ask(q, index, show_context=show_ctx)


if __name__ == "__main__":
    main()
