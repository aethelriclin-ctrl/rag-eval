"""生成模块（RAG 的第 ④ 步）。

要做的事：把"检索到的资料 + 用户的问题"拼成提示词，交给模型回答。

⭐ 这个文件里藏着一个决定项目成败的设计：
   SYSTEM_PROMPT 里那句"资料里没有就说没有"。
   —— 第 2 层要测"模型会不会编"，前提是我**已经明确给了它免责条件**。
      如果我允许它自由发挥，那它编了也不算幻觉，只能算我没约束。
      这是从项目 2 的幻觉实验里学到的：**先把规则说清楚，再测它守不守。**
"""

import os
from openai import OpenAI

# 回答用的模型：用便宜的 flash（项目 3 的结论：精度无差异，没必要用贵的）
CHAT_MODEL = "deepseek-flash"

# ⭐⭐ 这段提示词你可以自己改、对比效果 —— 改了什么、结果变了多少，都值得记下来
SYSTEM_PROMPT = """你是一个基于资料回答问题的小助手。

规则（必须严格遵守）：
1. 只根据下面提供的【资料】回答，不要使用资料之外的知识。
2. 如果资料里没有足够信息回答问题，直接回答"资料中没有相关信息"，不要猜测。
3. 回答时请注明你依据的是哪几段资料（例如"依据资料 1、3"）。
"""

_client = None


def get_client():
    """惰性创建客户端（原因同 embedder.py）。"""
    global _client
    if _client is None:
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError('没读到 DEEPSEEK_API_KEY。先执行：$env:DEEPSEEK_API_KEY="你的key"')
        _client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
    return _client


def generate(question, context):
    """根据资料回答问题。

    参数：
        question: 用户的问题
        context:  retriever.build_context() 拼好的资料文本

    返回：
        (回答文本, 用量信息dict)   —— 把用量返回出来，方便第 2 层算成本
    """
    user_content = f"""【资料】
{context}

【问题】
{question}
"""

    resp = get_client().chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0,      # 评测场景要可复现，温度设 0
    )

    answer = (resp.choices[0].message.content or "").strip()

    # 记录用量：第 2 层要算成本（项目 1、3 都这么做过）
    usage = {
        "tokens_in": getattr(resp.usage, "prompt_tokens", None),
        "tokens_out": getattr(resp.usage, "completion_tokens", None),
    }
    return answer, usage
