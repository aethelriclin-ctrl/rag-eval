"""探测哪个 API 提供方可用的 embedding 接口（不猜，直接测）。

背景：
    DeepSeek 官方不提供 embedding（已查证：官方 API 文档无 Embeddings 接口）。
    所以要用别家的 key。但中转站是否支持 embeddings 接口，文档往往说不清 ——
    唯一可靠的办法是**发一个最小请求，看它回什么**。

用法：
    1. 先设好环境变量（有几个设几个）：
         $env:QWEN_KEY="你的七牛云key"
         $env:RELAY_KEY="你的中转站key"
         $env:RELAY_BASE="https://你的中转站域名/v1"
    2. 跑：python tools\probe_embedding.py

它会逐个试，最后打印结论：哪个能用、用哪个模型名、向量维度多少。

⚠️ 每个请求只花掉一次"你好"级别的 token，成本可忽略。
⚠️ 脚本不会打印你的 key，只打印它的前 4 位 + 长度 —— 避免 key 泄漏到聊天记录里。
"""

import os
import sys

from openai import OpenAI

# 待探测的候选：每家可以给多个 base_url（因为地址常常要靠试）
# 结构：(名字, [base_url 列表], [候选模型名...], 读哪个环境变量当 key)
CANDIDATES = [
    (
        "阿里云百炼 DashScope（2026-09-26 文档查证）",
        ["https://dashscope.aliyuncs.com/compatible-mode/v1"],
        ["text-embedding-v3", "text-embedding-v4", "text-embedding-v2"],
        "DASHSCOPE_API_KEY",
    ),
    (
        "七牛云 AI",
        [
            "https://api.qnaigc.com/v1",
            "https://ai.qiniuapi.com/v1",
            "https://openai.qiniu.com/v1",
        ],
        ["bge-m3", "text-embedding-v3", "bge-large-zh-v1.5",
         "embedding-3", "text-embedding-ada-002"],
        "QWEN_KEY",
    ),
    (
        "中转站",
        [os.environ.get("RELAY_BASE", "")],
        ["text-embedding-3-small", "text-embedding-3-large",
         "text-embedding-ada-002", "bge-m3", "embedding-3"],
        "RELAY_KEY",
    ),
]

TEST_TEXT = "这是一句用来测试向量化接口的话。"


def mask(key):
    """只显示前 4 位和长度，避免把 key 打进聊天记录。"""
    if not key:
        return "（未设置）"
    return f"{key[:4]}...（共 {len(key)} 位）"


def try_one(label, base_url, api_key, model):
    """试一个组合，返回 (是否成功, 说明)。"""
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.embeddings.create(model=model, input=TEST_TEXT)
        vec = resp.data[0].embedding
        return True, f"✅ 成功！维度 = {len(vec)}"
    except Exception as ex:
        msg = str(ex)
        # 把长错误压短，只留关键信息
        if len(msg) > 160:
            msg = msg[:160] + "…"
        return False, f"❌ {msg}"


def main():
    print("=" * 66)
    print("embedding 接口探测（不猜，直接测）")
    print("=" * 66)

    any_ok = False

    for label, base_urls, models, key_name in CANDIDATES:
        api_key = os.environ.get(key_name)

        print(f"\n【{label}】")
        print(f"  api_key（变量名 {key_name}）: {mask(api_key)}")

        if not api_key:
            print(f"  ⏭️  跳过（需要设置环境变量 {key_name}）")
            continue

        for base_url in base_urls:
            if not base_url:
                print("  ⏭️  跳过（地址未设置）")
                continue
            print(f"  base_url : {base_url}")
            for model in models:
                ok, detail = try_one(label, base_url, api_key, model)
                print(f"    {model:<26} {detail}")
                if ok:
                    any_ok = True
                    print(f"\n  ⭐⭐ 记住这个组合：")
                    print(f"      base_url = {base_url}")
                    print(f"      model    = {model}")
                    print(f"      key 变量 = {key_name}")
                    break        # 这个地址找到一个能用的就够了
            if any_ok:
                break            # 这家已经找到了，不用再试别的地址

    print("\n" + "=" * 66)
    if any_ok:
        print("结论：有可用的 embedding 接口 —— 把上面 ⭐ 那行的 base_url 和 model")
        print("      填进 rag/embedder.py 顶部的常量，就能继续第 ② 步。")
    else:
        print("结论：都没测通。那我们就先走 BM25 方案（纯本地、0 成本），")
        print("      把整条链路跑通并完成检索评测；embedding 之后再补。")
    print("=" * 66)


if __name__ == "__main__":
    try:
        import openai  # noqa: F401
    except ImportError:
        print("缺依赖，先跑：pip install openai")
        sys.exit(1)
    main()
