"""检索器工厂 —— 让"用哪种检索"变成一个可切换的配置。

为什么要这一层：
    第 1、2 层用 BM25（字面匹配）跑通了全链路，现在加向量检索（语义匹配）做对照。
    如果 main.py / 评测脚本直接 import 某个具体检索器，互换时就要改一堆地方。
    加一层工厂之后，**换检索只要改下面 RETRIEVER 这一个字符串**，
    或者用环境变量 RETRIEVER=vector 覆盖（评测脚本里就是这么切来切去的）。

两个检索器的差别（第 4 组对照实验实测，**不是预期**）：
    · `bm25`   —— 字面匹配，本项目默认
    · `vector` —— 语义匹配（阿里云百炼 text-embedding-v3）

    ⚠️ **实测结果：在这个规模（103 块语料、28 个变体）上，两者整体打平**
    （同义问法 17/17 = 17/17，口语改写 26/28 = 26/28，只有 2 道题有分歧）。
    **所以"bm25 强在精确词、vector 强在语义相近"这个分工，是被数据否定的预期，不是结论。**
"""

import os

from rag import bm25, splitter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
INDEX_PATH = os.path.join(BASE_DIR, "data", "index_bm25.json")

# ⭐ 想换检索方式，改这里；或设环境变量 RETRIEVER=vector 临时覆盖。
RETRIEVER = os.environ.get("RETRIEVER", "bm25")   # "bm25" / "vector"


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

    if kind == "vector":
        # 延迟 import：只在真的用向量检索时才需要 numpy
        from rag import vector_retriever
        r = vector_retriever.VectorRetriever()
        r.build(chunks)
        return r

    raise NotImplementedError(
        f"检索器 '{kind}' 不存在。可选：'bm25' / 'vector'"
    )


def load(kind=RETRIEVER):
    """读缓存；没有就现建。"""
    if kind == "bm25":
        r = bm25.BM25Retriever.load(INDEX_PATH)
        if r is not None:
            return r

    if kind == "vector":
        from rag import vector_retriever
        r = vector_retriever.VectorRetriever.load()
        if r is not None:
            return r

    print(f"没有可用的 {kind} 索引，开始建……")
    return build(kind)
