"""向量检索（语义匹配）—— 本项目第 ② 个检索实现，用于和 BM25 做对照。

为什么要做这个（这是第 2 层末尾留下的那个悬念）：
    第 1、2 层用的是 **BM25（字面匹配）**，结论是：
       · 同义问法 Recall@3 = 100%
       · 口语改写 92.9%
       · **但失效集中在"关键词被整体替换"时**（"判分" → "打分"）
     而第 2 层的诊断进一步发现：**答案要点全在索引里（44/44），
     但 top-5 只覆盖 80%**，缺的全是"枚举列表"类要点。

    → 那 20% 的缺口，**能不能用"语义匹配"补回来？**
      这就是本模块存在的唯一理由：**给 BM25 一个对照物。**

它和 BM25Retriever 的接口完全一致（build / retrieve / save / load），
所以 `factory.py` 里只改一个字符串就能切换——这是当初把它做成"可替换部件"的回报。

原理（三句话）：
    ① 把每块文本送进 embedding 模型 → 得到一串数字（向量）
    ② 把问题也送进去 → 得到一个向量
    ③ 比"谁的方向最接近" → 用余弦相似度（而不是比谁的字面重合多）

⭐ 对照实验的看点：
    · BM25 强在**字面精确**（"exact"、"tool_call_id" 这种专有词）
    · 向量强在**语义相近**（"打分方法" ≈ "判分方式"）
    · 所以结果很可能是"各有胜负"——**而那正是最值得写进报告的东西**
"""

import json
import os

import numpy as np

from rag import embedder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(BASE_DIR, "data", "index_vector.npz")


def cosine(a, b):
    """余弦相似度：两个向量夹角的余弦值，范围 -1~1，越接近 1 越相似。

    为什么用余弦而不是欧氏距离：
        我们要比的是"方向像不像"（语义接近程度），不是"长度差多少"。
    """
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class VectorRetriever:
    """向量检索器。接口与 BM25Retriever 保持一致。"""

    def __init__(self):
        self.items = []      # [{"source":..., "text":...}]
        self.matrix = None   # shape (n_chunks, dim)，每行是一个块的向量

    # ---------- 建索引 ----------

    def build(self, items):
        """items: [(来源文件名, 块内容), ...]

        会把所有块送去 embedding 并缓存——**这一步要调 API，但会缓存**，
        所以只在第一次跑（或知识库变了）时才花钱。
        """
        self.items = [{"source": s, "text": t} for s, t in items]
        texts = [it["text"] for it in self.items]

        print(f"   正在向量化 {len(texts)} 块（模型 {embedder.EMBED_MODEL}）……")
        vectors = embedder.embed_batch(texts)
        self.matrix = np.array(vectors, dtype=np.float32)

        # 缓存：存 items 与向量矩阵。用 npz 是为了省空间、读得快。
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        np.savez_compressed(
            CACHE_PATH,
            matrix=self.matrix,
            sources=np.array([it["source"] for it in self.items]),
            texts=np.array([it["text"] for it in self.items]),
            model=np.array([embedder.EMBED_MODEL]),
        )
        print(f"   向量索引完成：{self.matrix.shape[0]} 块 × {self.matrix.shape[1]} 维")
        print(f"   已缓存到：{CACHE_PATH}")

    # ---------- 检索 ----------

    def retrieve(self, question, top_k=3):
        """把问题向量化，然后和所有块算余弦相似度，取前 top_k。"""
        qvec = np.array(embedder.embed_one(question), dtype=np.float32)
        # 向量化算相似度：matrix 每行与 qvec 做点积 / 模长乘积
        norms = np.linalg.norm(self.matrix, axis=1) * np.linalg.norm(qvec)
        sims = (self.matrix @ qvec) / np.maximum(norms, 1e-9)

        idx = np.argsort(sims)[::-1][:top_k]
        return [{"source": self.items[i]["source"],
                 "text": self.items[i]["text"],
                 "score": float(sims[i])} for i in idx]

    # ---------- 缓存 ----------

    def save(self, path=None):
        """本实现自带 npz 缓存（在 build 里做了），这里保留接口一致性。"""
        if self.matrix is None:
            raise RuntimeError("还没建索引，先调用 build()")

    @classmethod
    def load(cls, path=None):
        """读缓存。不存在或模型变了就返回 None（让调用方重建）。"""
        if not os.path.exists(CACHE_PATH):
            return None
        try:
            data = np.load(CACHE_PATH, allow_pickle=True)
            cached_model = str(data["model"][0]) if "model" in data else ""
            if cached_model != embedder.EMBED_MODEL:
                print(f"   ⚠️ 向量缓存用的模型是 {cached_model}，"
                      f"当前是 {embedder.EMBED_MODEL}，需要重建")
                return None
            r = cls()
            r.matrix = data["matrix"]
            r.items = [{"source": str(s), "text": str(t)}
                       for s, t in zip(data["sources"], data["texts"])]
            return r
        except Exception as ex:
            print(f"   ⚠️ 读向量缓存失败（{str(ex)[:60]}），将重建")
            return None


if __name__ == "__main__":
    """自测：python rag\\vector_retriever.py —— 用玩具语料验证语义检索。

    这个自测最能说明"向量 vs 字面"的差别：
        问"项目哪里坏了" —— 和"失败案例"一个字都不重合。
        BM25 会得 0 分（见 rag/bm25.py 的自测），而向量应当能找到它。
    """
    demo = [
        ("a.md", "评测失败归因机制把每次失败拆成六类"),
        ("b.md", "模型成本在多次跑测之间波动 1.8 到 2.8 倍"),
        ("c.md", "Agent 工具调用循环包含工具注册与结果回填"),
    ]
    r = VectorRetriever()
    r.build(demo)

    for q in ["失败归因", "成本波动", "工具调用循环", "项目哪里坏了", "打分方式会不会影响结果"]:
        hits = r.retrieve(q, top_k=2)
        print(f"\n问题：{q}")
        for h in hits:
            print(f"   {h['score']:.3f}  [{h['source']}] {h['text'][:30]}")

    print("\n⚠️ 注意最后两个问题——")
    print("   「项目哪里坏了」：BM25 会得 0 分（字面完全不重合），向量应当能找到 a.md。")
    print("   「打分方式会不会影响结果」：文档里写的是「判分」，字面对不上，向量应当能抓到相关性。")
