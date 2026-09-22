"""向量化模块（RAG 的第 ② 步）—— **已写好，但当前尚未接进检索链路**。

📌 当前状态（重要）：
    系统的检索用的是 `bm25.py`（字面匹配），本模块**没有被调用**。
    原因：DeepSeek 官方不提供 embedding 接口（已查证官方 API 文档），
    试过的第三方中转 key 地址未通，所以先用 BM25 把链路跑通。

    等拿到可用的 embedding 后，把它接进 `factory.py` 的 VectorRetriever，
    就能和 BM25 做对照实验——**这正是本项目"下一步"里的第一项。**

要解决的事：把每一块文本变成一个"向量"（一串数字），
            这样计算机才能比较两段文本"意思像不像"。

为什么向量能表达语义：
    embedding 模型把文本映射到高维空间，意思相近的文本，
    它们的向量方向也相近（夹角小）。所以"比较相似度"就变成了"算夹角"。

⚠️ 本模块最重要的设计：**缓存**
    向量算一次就够了。如果不缓存，每次提问都要把整个知识库重新算一遍 ——
    又慢又费钱。所以第一次算完存进 data/vectors.json，之后直接读。
"""

import os
import json

from openai import OpenAI

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_CACHE = os.path.join(BASE_DIR, "data", "vectors.json")

# ⚠️ 模型名可能要改：不同平台的 embedding 模型名不一样。
#    跑的时候如果报"model not found"，把这个字符串换成平台文档里的名字。
EMBED_MODEL = "deepseek-embedding"

_client = None


def get_client():
    """惰性创建客户端。

    为什么不写在模块顶层：
        项目 1 踩过这个坑 —— 模块导入时就创建客户端，
        如果环境变量没设，会直接抛 "Missing credentials"，
        把"你没设 API Key"这个友好提示给盖掉了。
        惰性创建能让错误信息更清楚。
    """
    global _client
    if _client is None:
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError(
                "没读到 DEEPSEEK_API_KEY。在 PowerShell 里先执行：\n"
                '  $env:DEEPSEEK_API_KEY="你的key"'
            )
        _client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
    return _client


def embed_one(text):
    """把一段文本转成向量（list of float）。"""
    resp = get_client().embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding


def embed_batch(texts, batch_size=16):
    """批量向量化。

    为什么要批量：一次请求带多条文本，比循环单条快得多（也省往返开销）。
    batch_size 不要太大，避免单次请求体过大被拒。
    """
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = get_client().embeddings.create(model=EMBED_MODEL, input=batch)
        # ⚠️ 注意：返回的 data 顺序**不一定**和输入一致（有些平台会乱序），
        #    所以按 index 排序取回来，别直接 append。
        items = sorted(resp.data, key=lambda d: d.index)
        vectors.extend([it.embedding for it in items])
        print(f"  已向量化 {min(i + batch_size, len(texts))}/{len(texts)}")
    return vectors


def build_index(chunks):
    """把 [(来源, 块内容)] 全部向量化，返回索引结构并写入缓存。

    返回结构：
        {
          "model": "用的是哪个 embedding 模型",   ← 记下来！换模型后旧向量就失效了
          "items": [{"source": 来源文件名, "text": 块内容, "vector": [...]}, ...]
        }
    """
    texts = [c[1] for c in chunks]
    print(f"开始向量化 {len(texts)} 块……")
    vectors = embed_batch(texts)

    index = {
        "model": EMBED_MODEL,
        "items": [
            {"source": src, "text": txt, "vector": vec}
            for (src, txt), vec in zip(chunks, vectors)
        ],
    }

    os.makedirs(os.path.dirname(VECTOR_CACHE), exist_ok=True)
    with open(VECTOR_CACHE, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    print(f"索引已缓存到：{VECTOR_CACHE}")
    return index


def load_index():
    """读缓存。如果缓存不存在、或模型换了，返回 None（让调用方重建）。"""
    if not os.path.exists(VECTOR_CACHE):
        return None
    with open(VECTOR_CACHE, encoding="utf-8") as f:
        index = json.load(f)
    if index.get("model") != EMBED_MODEL:
        print(f"⚠️ 缓存用的模型是 {index.get('model')}，当前是 {EMBED_MODEL}，需要重建")
        return None
    return index
