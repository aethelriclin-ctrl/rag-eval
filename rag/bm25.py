"""BM25 检索（字面匹配）—— 本项目第 ① 个检索实现。

BM25 是信息检索里的经典算法，搜索引擎用了几十年。它只需要"词频统计"，
不需要任何模型，所以：**纯本地、零成本、可离线跑。**

⭐ 它和后面的"向量检索"要解决同一个问题：
    给一个问题，从所有块里找出最相关的 top_k。
    所以两者可以**互换** —— 这正是我们做对照实验的基础。

---

BM25 要解决的四个问题（这四个问题就是它的全部）：

    ① 中文怎么分词？
       → 用 2-gram（"失败归因" → "失败"、"败归"、"归因"）
         为什么要重叠：中文没有空格，而"败归"这种跨界组合
         能捕捉到词的边界信息，比单纯单字切分准。
         ⚠️ 它的问题也在这：**不懂词的意思**，只认字面。

    ② 一个词出现越多就越相关吗？
       → 是，但**要有上限**。出现 10 次和 20 次不该差一倍。
         BM25 用参数 k1 控制这个"饱和"速度。

    ③ 常见词（"的"、"是"、"我"）没信息量怎么办？
       → 用 IDF（逆文档频率）：**越罕见的词权重越高**。
         "失败"只在 3 个块里出现 → 权重高；
         "的"在 100 个块里出现 → 权重接近 0。

    ④ 长块词天然多，会占便宜怎么办？
       → 长度归一化：**除以块长度**（相对于平均长度）。
         BM25 用参数 b 控制归一化的强度。
"""

import json
import math
import os
import re
from collections import Counter

# BM25 的两个经验参数（默认值来自经典文献，一般不用改）
K1 = 1.5    # 词频饱和速度：越大，词频的影响越持久
B = 0.75    # 长度归一化强度：0 = 不归一化，1 = 完全归一化


# ⚠️ 汉语里最高频的"功能字"：它们几乎不携带语义，但在任何文本里都大量出现。
#    2-gram 遇到它们会产生一堆垃圾碎片（"的是"、"了这"、"在这"…），
#    这些碎片在全文档里到处都是 —— **IDF 也拦不住它们**（虽然罕见度低，
#    但它们出现的次数太多），结果把真正关键词的权重稀释掉了。
#
#    这是冒烟测试（检索召回 0/5）之后加的第一次修复。
STOP_CHARS = set(
    "的了是在和与我你他她它这那有就不人都一也很到说要去会着看把被给对从为以所而"
    "吗呢吧啊呀哦嗯么什么怎样如何以及或者但因所以如果虽然但是不过还是可以这个那个"
)


def tokenize(text):
    """中文 2-gram 分词 + 停用字过滤（不依赖任何分词库）。

    做法：
        1. 只保留中文、字母、数字，其他字符当分隔符
        2. 对每一段连续文本，切出所有长度为 2 的相邻组合
        3. **丢掉"两个字都是功能字"的碎片**（这是本次修复新增的一步）

    例：
        "失败归因机制" → ["失败", "败归", "归因", "因机", "机制"]   （都不含功能字，全留）
        "这是一个测试" → ["这是"(丢), "是一"(丢), "一个"(留), "个测"(留), "测试"(留)]

    ⚠️ 这个方法的固有缺陷（后面做对照实验时会看到）：
        它只认字面。"项目哪里坏了" 和 "失败案例" 一个字都不重合，
        所以 BM25 找不到后者 —— 而人一眼就知道这两句是一个意思。
        **这就是"为什么需要 embedding"的第一个理由。**
    """
    tokens = []
    # 把文本切成"连续的中文/字母/数字"片段
    for segment in re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9_]+", text):
        if len(segment) == 1:
            if segment not in STOP_CHARS:          # 单字也过滤功能字
                tokens.append(segment)
        else:
            for i in range(len(segment) - 1):
                gram = segment[i:i + 2]
                # 两个字都是功能字 → 丢掉（如"的是""了这"）
                if gram[0] in STOP_CHARS and gram[1] in STOP_CHARS:
                    continue
                tokens.append(gram)
    return tokens


class BM25Retriever:
    """BM25 检索器。接口和未来的 VectorRetriever 保持一致：
           build(items)  建索引
           retrieve(question, top_k)  检索
           save(path) / load(path)    缓存
    """

    def __init__(self, k1=K1, b=B):
        self.k1 = k1
        self.b = b
        self.items = []        # [{"source": ..., "text": ...}]
        self.doc_tokens = []   # 每块的 token 列表
        self.doc_len = []      # 每块的长度
        self.avg_len = 0.0
        self.idf = {}          # 每个词的反文档频率
        self.tf = []           # 每块的词频字典

    # ---------- 建索引 ----------

    def build(self, items):
        """items: [(来源文件名, 块内容), ...]"""
        self.items = [{"source": s, "text": t} for s, t in items]
        self.doc_tokens = [tokenize(it["text"]) for it in self.items]
        self.doc_len = [len(toks) for toks in self.doc_tokens]
        self.avg_len = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0
        self.tf = [Counter(toks) for toks in self.doc_tokens]

        # IDF：出现在越多块里的词，权重越低。
        # 用带平滑的公式，避免除零（这是 BM25 的标准写法之一）
        n = len(self.items)
        df = Counter()
        for toks in self.doc_tokens:
            for w in set(toks):
                df[w] += 1
        self.idf = {
            w: math.log(1 + (n - c + 0.5) / (c + 0.5))
            for w, c in df.items()
        }
        print(f"   BM25 索引完成：{n} 块，{len(self.idf)} 个不同词，"
              f"平均块长 {self.avg_len:.0f} 个 token")

    # ---------- 检索 ----------

    def _score(self, query_tokens, i):
        """算第 i 块对这条 query 的 BM25 分数。

        公式（就是上面四个问题的数学化）：
            score = Σ  IDF(词) × [ 词频 × (k1+1) ]
                               [ 词频 + k1 × (1 - b + b × 块长/平均长) ]

            分子    ：词出现越多分越高
            分母    ：① 加了 k1 之后，词频增长会"饱和"（解决 ②）
                     ② 除的那一坨是长度归一化（解决 ④）
            IDF     ：常见词权重低（解决 ③）
        """
        score = 0.0
        for w in query_tokens:
            if w not in self.idf:
                continue                        # 索引里没有这个词，跳过
            f = self.tf[i].get(w, 0)
            if f == 0:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avg_len)
            score += self.idf[w] * (f * (self.k1 + 1)) / denom
        return score

    def retrieve(self, question, top_k=3):
        """返回 top_k 个最相关的块，格式与其他检索器统一。"""
        qtokens = tokenize(question)
        scored = [
            {"source": self.items[i]["source"],
             "text": self.items[i]["text"],
             "score": self._score(qtokens, i)}
            for i in range(len(self.items))
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    # ---------- 缓存 ----------

    def save(self, path):
        """把索引存成 json。BM25 的索引就是几个数字表，存起来很快。"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "kind": "bm25",
            "k1": self.k1, "b": self.b,
            "items": self.items,
            "doc_tokens": self.doc_tokens,
            "doc_len": self.doc_len,
            "avg_len": self.avg_len,
            "idf": self.idf,
            "tf": [dict(c) for c in self.tf],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        print(f"   索引已缓存：{path}")

    @classmethod
    def load(cls, path):
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            p = json.load(f)
        if p.get("kind") != "bm25":
            return None
        r = cls(k1=p["k1"], b=p["b"])
        r.items = p["items"]
        r.doc_tokens = p["doc_tokens"]
        r.doc_len = p["doc_len"]
        r.avg_len = p["avg_len"]
        r.idf = p["idf"]
        r.tf = [Counter(d) for d in p["tf"]]
        return r


if __name__ == "__main__":
    """自测：python rag\\bm25.py —— 用一个玩具语料验证打分逻辑符合直觉。"""
    demo = [
        ("a.md", "评测失败归因机制把每次失败拆成四类"),
        ("b.md", "模型成本在多次跑测之间波动约 2 倍"),
        ("c.md", "Agent 工具调用循环包含工具注册与结果回填"),
    ]
    r = BM25Retriever()
    r.build(demo)

    for q in ["失败归因", "成本波动", "工具调用循环", "项目哪里坏了"]:
        hits = r.retrieve(q, top_k=2)
        print(f"\n问题：{q}")
        for h in hits:
            print(f"   {h['score']:.3f}  [{h['source']}] {h['text'][:30]}")

    print("\n⚠️ 注意最后那个问题「项目哪里坏了」——")
    print("   它和 a.md 的「失败」一个字都不重合，所以分数是 0。")
    print("   但人一眼就知道该找 a.md。**这就是 BM25 的边界，也是我们要评测的东西。**")
