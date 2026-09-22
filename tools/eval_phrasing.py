"""措辞敏感性测试：同一件事换 4 种问法，看检索召回率会不会崩。

为什么做这个实验：
    eval_retrieval.py 里 17/17 全中，但那些问题的用词和文档高度重合
    —— 相当于"照着答案出题"，测不出真实场景。

    真实用户不会用文档里的原话提问。所以这里做一组对照：
    **同一个意图，四种问法，从"文档用词"逐步过渡到"口语化"，
    看 BM25 在第几种问法上开始失效。**

预期价值：
    如果"原话问法"命中、"口语问法"不命中 —— 那就用数据证明了
    **"字面匹配的字面，是指'提问者用的字'还是'文档里的字'，这两者经常不一样"**
    而这正是需要向量检索（语义匹配）的理由。

用法：
    python tools\\eval_phrasing.py
    python tools\\eval_phrasing.py --k 5      # 放宽到 5 条
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import factory  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SET_PATH = os.path.join(BASE_DIR, "eval_phrasing_set.json")
RESULT_PATH = os.path.join(BASE_DIR, "data", "eval_phrasing_result.json")


def main():
    args = sys.argv[1:]
    k = int(args[args.index("--k") + 1]) if "--k" in args else 3

    with open(SET_PATH, encoding="utf-8") as f:
        groups = json.load(f)["groups"]

    retriever = factory.load()
    if retriever is None:
        print("没有可用索引，先跑：python main.py --build")
        return

    print("=" * 78)
    print(f"措辞敏感性测试　（Recall@{k}，检索方式：{factory.RETRIEVER}）")
    print("=" * 78)

    total, hit_total = 0, 0
    all_rows = []
    # 按"第几种问法"统计：位置 0 = 文档原话，位置 3 = 最口语
    by_position = {}

    for g in groups:
        print(f"\n【{g['id']}】{g['topic']}　期望来源：{g['expected_sources']}")
        for idx, q in enumerate(g["variants"]):
            results = retriever.retrieve(q, top_k=k)
            got = [r["source"] for r in results]
            hit = any(s in g["expected_sources"] for s in got)
            score = results[0]["score"] if results else 0.0

            total += 1
            hit_total += 1 if hit else 0

            by_position.setdefault(idx, [0, 0])
            by_position[idx][1] += 1
            if hit:
                by_position[idx][0] += 1

            mark = "✅" if hit else "❌"
            print(f"  {mark} [{idx}] {q}")
            if not hit:
                print(f"       实际命中：{got[0] if got else '（无）'}"
                      f"　最高分 {score:.2f}")

            all_rows.append({
                "group": g["id"], "topic": g["topic"], "position": idx,
                "question": q, "expected": g["expected_sources"],
                "got": got, "hit": hit, "top_score": round(score, 4),
            })

    print("\n" + "=" * 78)
    print(f"【总结果】Recall@{k} = {hit_total}/{total} = {hit_total / total * 100:.1f}%")

    print("\n【按问法位置拆开看】—— 位置越大，越接近日常口语")
    labels = {0: "文档原话", 1: "轻微改动", 2: "口语化", 3: "很口语/很短"}
    for idx in sorted(by_position):
        ok, n = by_position[idx]
        bar = "█" * int(ok / n * 20) if n else ""
        print(f"  位置 {idx}（{labels.get(idx, '?'):<8}）"
              f"{ok}/{n} = {ok / n * 100:5.1f}%  {bar}")

    print("=" * 78)

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump({"retriever": factory.RETRIEVER, "k": k,
                   "recall": hit_total / total if total else 0,
                   "by_position": {str(k2): v for k2, v in by_position.items()},
                   "rows": all_rows}, f, ensure_ascii=False, indent=2)
    print(f"原始结果已写入：{RESULT_PATH}")


if __name__ == "__main__":
    main()
