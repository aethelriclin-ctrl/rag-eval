"""检索器工厂 —— 让"用哪种检索"变成一个可切换的配置。

为什么要这一层：
    我们先用 BM25（字面匹配）跑通全链路，以后再换/加向量检索（语义匹配）。
    如果 main.py 直接 import 某个具体检索器，换的时候就要改一堆地方。
    加一层工厂之后，**换检索只要改下面 RETRIEVER 这一个字符串。**

🚧 还没实现的部分（等 embedding key 通了再补）：
    VectorRetriever —— 语义检索。它需要 embedding 接口，
    而 DeepSeek 官方不提供 embedding（已查证），要用别家的 key。
"""

import os

from rag import bm25, splitter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
INDEX_PATH = os.path.join(BASE_DIR, "data", "index_bm25.json")

# ⭐ 想换检索方式，改这里
RETRIEVER = "bm25"      # 可选："bm25"（已实现） / "vector"（待实现）


def build(kind=RETRIEVER):
    """建索引：切分文档 → 建检索结构 → 缓存。"""
    chunks = splitter.load_all_docs(DOCS_DIR)
    if not chunks:
        print("⚠️ docs/ 是空的，先跑：python setup.py")
        return None
    print(f"① 切分完成：{len(chunks)} 块")

    print(f"② 建 {kind} 索引")
    if kind == "bm25":
        r = bm25.BM25Retriever()
        r.build(chunks)
        r.save(INDEX_PATH)
        return r

    raise NotImplementedError(
        f"检索器 '{kind}' 还没实现。\n"
        "它需要可用的 embedding 接口 —— 先跑 tools/probe_embedding.py 找到能用的地址。"
    )


def load(kind=RETRIEVER):
    """读缓存；没有就现建。"""
    if kind == "bm25":
        r = bm25.BM25Retriever.load(INDEX_PATH)
        if r is not None:
            return r
    print("没有可用索引，开始建……")
    return build(kind)
