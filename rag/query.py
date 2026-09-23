"""查询改写（Query Rewriting）—— 检索之前的"清噪"步骤。

为什么需要它（这是被诊断数据逼出来的）：
    diagnose_retrieval.py 显示：答案要点 44/44 全在索引里，但 top-5 只覆盖 80%。
    缺掉的要点集中在"枚举列表"类问题（哪几类 / 哪些场景）。

    根因是 BM25 + 2-gram 的一个固有毛病：
        **"哪几类""哪些""怎么"这类疑问词，在文档里几乎不出现 → IDF 极高**
        → 它们把真正的关键词（exact / code / json…）挤了下去
        → 排上来的反而是"碰巧含这些疑问字"或"含高频词（评测/流水线）"的块

    解法：**在检索之前，把问题改写成"文档里真会出现的词"。**

例子：
    "评测流水线实现了哪几类评分器？"
      → "评分器 类型 exact code json 精确匹配 执行判分"

    "六类失败归因机制把失败分成了哪几类？"
      → "失败归因 分类 能力不足 格式不合规 评分器误判 题目歧义"

    ⚠️ 注意改写后的查询里**没有疑问词**，而且补上了可能的关键词（同义词扩展）。

两种实现：
    ① LLM 改写（默认）—— 调一次模型，质量好
    ② 机械剥疑问词（兜底）—— 不花钱；LLM 失败时自动使用
"""

import json
import os
import re

from openai import OpenAI

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(BASE_DIR, "data", "rewrites.json")

# ⚠️ 为什么要缓存改写结果（2026-09-23 的实测驱动）：
#    check_determinism.py 测出——**同一个问题改写三次，三次结果都不同**：
#      [1] 评测项目 题目 考题 用例 case 数量 总数 题量
#      [2] 评测项目 题目 数量 总数 题数 题目数 用例数 样本数 数据集 条目
#      [3] 评测项目 题目 数量 总数 题量 题目数 用例数
#    三次都合理，但三次不一样 → 检索到的块不同 → 答案不同 → **跑测结果不可复现**。
#    （实测：eval_answer 连跑两次，关键词判分 83.5% → 80.6%，LLM 判分 85.0% → 77.2%）
#
#    这是我引入的问题：查询改写修好了"疑问词污染"，却带来了新的随机源。
#    解法——**把改写结果固化**：首次调用时问模型并写进 data/rewrites.json，
#    之后同一个问题永远读缓存。这样"改写 → 检索 → 答案"这条链就确定了。
#
#    注意：缓存会让"已跑过的问题"固定，但**新增问题仍会引入新改写**——
#    所以报告里的数字要注明"基于首次改写结果"。想要全部刷新用 --refresh。

# 常见疑问词/问句结构 —— 它们在文档里几乎不出现，却会拉高 BM25 的 IDF 权重
QUESTION_WORDS = [
    "哪几类", "哪几项", "哪几个", "哪些", "哪几", "什么", "怎么", "如何",
    "为什么", "为何", "是不是", "是否", "多少", "几道", "几次", "几轮",
    "一共", "分别", "请问", "介绍", "说明", "列举", "总结",
]

REWRITE_PROMPT = """你是检索查询改写器。把用户的问题改写成**适合字面检索的关键词串**。

规则：
1. **删掉所有疑问词和问句结构**（哪几类、哪些、什么、怎么、为什么、一共、分别、请问…）
2. **保留核心名词与技术术语**（原词照抄，不要换成同义词）
3. **补充可能的同义表述**，用空格分隔（例如"评分器"可以补"判分 打分"）
4. 只输出关键词串，**不要输出任何解释、不要标点、不要换行**

示例：
问题：评测流水线实现了哪几类评分器？
输出：评分器 判分 类型 exact 精确匹配 code 执行判分 json contains

问题：六类失败归因机制把失败分成了哪几类？
输出：失败归因 分类 能力不足 格式不合规 评分器误判 题目歧义

问题：我的项目用了哪家公司的 GPU？
输出：GPU 显卡 服务器 厂商
"""

_client = None


def get_client(api_key=None):
    import os
    global _client
    if _client is None:
        key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("没读到 DEEPSEEK_API_KEY")
        _client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
    return _client


def strip_question_words(question):
    """兜底方案：机械删掉疑问词。不花钱，但也能去掉一部分噪声。"""
    q = question
    for w in QUESTION_WORDS:
        q = q.replace(w, " ")
    q = re.sub(r"[？?。，,、；;：:！!]", " ", q)
    return re.sub(r"\s+", " ", q).strip() or question


def load_cache():
    """读改写缓存：{问题原文: 改写结果}"""
    if not os.path.exists(CACHE_PATH):
        return {}
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def rewrite(question, use_llm=True, model="deepseek-flash", refresh=False):
    """把问题改写成检索用的关键词串。

    ⭐ 优先读缓存——保证同一个问题**永远得到同一条检索词**（可复现）。
       缓存文件：data/rewrites.json
       refresh=True 时忽略缓存、重新问模型，并覆盖缓存里这一条。

    返回 (改写后的查询, 用的是什么方法)
    方法取值："cache"（读缓存）/ "llm"（新问的）/ "rule"（兜底）
    """
    if not use_llm:
        return strip_question_words(question), "rule"

    cache = load_cache()
    if not refresh and question in cache:
        return cache[question], "cache"

    try:
        resp = get_client().chat.completions.create(
            model=model,
            messages=[{"role": "user",
                       "content": f"{REWRITE_PROMPT}\n\n问题：{question}\n输出："}],
            temperature=0,
        )
        out = (resp.choices[0].message.content or "").strip()
        out = out.replace("\n", " ").strip()
        # 合理性检查：太短或太长都视为失败，走兜底
        if 2 <= len(out) <= 200 and not out.startswith("{"):
            cache[question] = out          # ⭐ 写回缓存，下次直接用
            save_cache(cache)
            return out, "llm"
        fallback = strip_question_words(question)
        cache[question] = fallback
        save_cache(cache)
        return fallback, "rule"
    except Exception as ex:
        print(f"    （查询改写失败，走兜底：{str(ex)[:60]}）")
        return strip_question_words(question), "rule"


if __name__ == "__main__":
    """自测：python rag\\query.py —— 看改写效果（会调 API，几分钱）"""
    tests = [
        "我的评测项目一共有几道题？",
        "评测流水线实现了哪几类评分器？",
        "六类失败归因机制把失败分成了哪几类？",
        "Agent 工具调用评测一共几道题，覆盖哪几类场景？",
    ]
    for q in tests:
        new, how = rewrite(q)
        print(f"\n原问题：{q}")
        print(f"改写后：{new}　（方法：{how}）")
