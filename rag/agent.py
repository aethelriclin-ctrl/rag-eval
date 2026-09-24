"""Agentic RAG：把检索做成 Agent 的一个工具，让它自己决定"该不该查"。

和第 2 层的区别（这是本层的全部意义所在）：

    第 2 层：问题 → 【强制检索】 → 资料 → 生成
    第 3 层：问题 → 【Agent 自己判断】 → 要查才查 → 生成

    多出来的那个"判断"，是本层要评测的对象。

设计上的三个决定（都有理由）：

    ① 只给一个工具（search_knowledge_base）
       —— 决策空间越小，"该不该查"这个判断才越纯粹。工具一多，
          就分不清是"判断错"还是"选了别的工具"。

    ② 提示词里明确写清规则（什么时候该查、什么时候不该查）
       —— 先把规则说清，再测它守不守。这是项目 2 学到的做法：
          如果规则本身含糊，测出来的失败到底是"模型不守规矩"还是
          "我没说清楚"，就分不清了。

    ③ 完整记录每一次工具调用
       —— 没有记录就没法评测决策。所以要记：调了没、调了几次、参数是什么。
"""

import json

from openai import OpenAI

CHAT_MODEL = "deepseek-flash"
MAX_ROUNDS = 5          # 最多几轮（防止无限循环，项目 2 见过循环到上限的失败模式）

SYSTEM_PROMPT = """你是一个助手，配有一个知识库检索工具。

工具用途：**检索关于「使用者本人做过的项目」的资料**（项目细节、数字、结论、失败案例等）。

规则（严格遵守）：
1. **涉及使用者项目的事实、数字、结论** → 必须先调用 search_knowledge_base 检索，再回答。
2. **通用常识、计算、翻译、写代码** → **直接回答，不要调用工具**（知识库里没有这些，查了也没用）。
3. **需要实时信息**（现在几点、今天的天气）→ 说明你无法获取实时信息，不要编造，也不要查库。
4. 检索到的资料里没有答案时，明确说「资料中没有相关信息」，**不要猜测或编造**。
5. 回答时注明依据来自哪几段资料。
"""

# 工具定义（OpenAI Function Calling 格式）
TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": "检索关于使用者本人做过的项目的知识库。"
                       "仅在问题涉及他的项目事实/数字/结论时使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索用的关键词或问句",
                }
            },
            "required": ["query"],
        },
    },
}]

_client = None


def get_client():
    global _client
    if _client is None:
        import os
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError('没读到 DEEPSEEK_API_KEY。先执行：$env:DEEPSEEK_API_KEY="你的key"')
        _client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
    return _client


def run(question, retriever, top_k=5, use_rewrite=True, verbose=False):
    """跑一轮 Agent：让它自己决定要不要检索。

    返回一个记录 dict：
        {
          "answer": 最终回答,
          "called_tool": 是否调用过检索,
          "tool_calls": [{"query":..., "rewritten":..., "n_hits":..., "sources":[...]}, ...],
          "rounds": 用了几轮,
          "citations": 回答里引用的资料编号（由评测脚本解析）,
          "hit_sources": 实际检索到的来源文件（供引用准确性核对）,
          "usage": {...}
        }
    """
    from rag import query as query_mod

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tool_calls_log = []
    hit_sources = []
    tokens = {"in": 0, "out": 0}
    answer = ""

    for rnd in range(MAX_ROUNDS):
        resp = get_client().chat.completions.create(
            model=CHAT_MODEL, messages=messages, tools=TOOLS, temperature=0,
        )
        msg = resp.choices[0].message
        tokens["in"] += getattr(resp.usage, "prompt_tokens", 0) or 0
        tokens["out"] += getattr(resp.usage, "completion_tokens", 0) or 0

        # 没有工具调用 → 这就是最终回答
        if not msg.tool_calls:
            answer = (msg.content or "").strip()
            break

        # 把助手的"工具请求"这一轮加进历史（必须，否则下一轮会报错）
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [{
                "id": tc.id, "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            } for tc in msg.tool_calls],
        })

        # 逐个执行工具，把结果回填
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}
            raw_query = args.get("query", "")

            # 复用第 2 层的查询改写（缓存固化，所以是确定的）
            if use_rewrite:
                search_q, how = query_mod.rewrite(raw_query)
            else:
                search_q, how = raw_query, "none"

            hits = retriever.retrieve(search_q, top_k=top_k)
            from rag import retriever as retriever_mod
            context = retriever_mod.build_context(hits)

            tool_calls_log.append({
                "query": raw_query, "rewritten": search_q, "rewrite_method": how,
                "n_hits": len(hits),
                "sources": [h["source"] for h in hits],
                "top_score": round(hits[0]["score"], 3) if hits else 0,
                # ⭐ 把拼接好的资料原文也记下来 —— 评测脚本要用它核对"引用准确性"
                #    （判断答案要点是否真的出现在模型引用的那些块里）
                "context": context,
            })
            hit_sources.extend(h["source"] for h in hits)

            if verbose:
                print(f"    🔧 调用检索：{raw_query!r}")
                print(f"       改写后：{search_q[:60]}")
                print(f"       命中 {len(hits)} 块：{', '.join(h['source'] for h in hits[:3])}…")

            messages.append({
                "role": "tool", "tool_call_id": tc.id, "content": context,
            })

    if not answer:
        answer = "（达到最大轮数仍未给出最终回答）"

    return {
        "answer": answer,
        "called_tool": len(tool_calls_log) > 0,
        "tool_calls": tool_calls_log,
        "rounds": rnd + 1,
        "hit_sources": hit_sources,
        "usage": tokens,
    }


if __name__ == "__main__":
    """自测：跑两道题，一道该查、一道不该查（会花钱，几分钱）"""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rag import factory

    r = factory.load()
    if r is None:
        print("没有可用索引，先跑：python main.py --build")
        sys.exit(1)

    for q in ["我的评测项目一共有几道题？", "1+1 等于几？"]:
        print(f"\n{'=' * 60}\n问题：{q}")
        out = run(q, r, verbose=True)
        print(f"  调用工具：{'是' if out['called_tool'] else '否'}　轮数 {out['rounds']}")
        print(f"  回答：{out['answer'][:120]}")
