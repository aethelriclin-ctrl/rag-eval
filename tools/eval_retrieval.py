"""检索质量评测：算出 Recall@k，而不是靠肉眼看输出。

为什么需要这个脚本（这是整个项目 4 的关键）：
    没有它，每次改检索都只能说"好像好一点"；
    有了它，每次改动都能报出"召回率从 0% 到 X%"。
    **有了刻度，调参才不是盲调。**

用法：
    python tools\\eval_retrieval.py          # 默认 k=3
    python tools\\eval_retrieval.py --k 5    # 看放宽到 5 条时能命中多少
    python tools\\eval_retrieval.py --detail # 打印每一题命中了什么

⚠️ 它不调用生成模型（不花钱）——只测【检索】这一步。
   因为要评的是"检索找没找到对的块"，跟模型答得好不好是两件事。
   （这两件事分开测，是项目 1 学到的：**先确认尺子，再量东西**。）
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import factory  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_SET = os.path.join(BASE_DIR, "eval_set.json")
RESULT_PATH = os.path.join(BASE_DIR, "data", "eval_retrieval_result.json")


def main():
    args = sys.argv[1:]
    k = int(args[args.index("--k") + 1]) if "--k" in args else 3
    detail = "--detail" in args

    with open(EVAL_SET, encoding="utf-8") as f:
        data = json.load(f)
    questions = data["questions"]

    retriever = factory.load()
    if retriever is None:
        print("没有可用索引，先跑：python main.py --build")
        return

    recallable = [q for q in questions if q["expected_sources"]]   # 有标准答案的题
    refuse_only = [q for q in questions if not q["expected_sources"]]  # 只能拒答的题

    print("=" * 70)
    print(f"检索质量评测　（Recall@{k}）")
    print(f"检索方式：{factory.RETRIEVER}　|　题库：{len(questions)} 题"
          f"（可召回 {len(recallable)} 题，应拒答 {len(refuse_only)} 题）")
    print("=" * 70)

    hits_ok = 0
    rows = []

    for q in recallable:
        results = retriever.retrieve(q["question"], top_k=k)
        got_sources = [r["source"] for r in results]
        # 命中判据：top_k 里至少有一个来源属于 expected_sources
        hit = any(s in q["expected_sources"] for s in got_sources)
        if hit:
            hits_ok += 1

        rows.append({
            "id": q["id"], "question": q["question"],
            "expected": q["expected_sources"], "got": got_sources,
            "hit": hit, "must_refuse": False,
            "top_score": round(results[0]["score"], 4) if results else 0,
        })
        mark = "✅" if hit else "❌"
        print(f"\n{mark} [{q['id']}] {q['question']}")
        if detail or not hit:
            print(f"    期望来源：{q['expected_sources']}")
            print(f"    实际命中：{got_sources}   （最高分 {rows[-1]['top_score']}）")
            for r in results[:2]:
                print(f"      · {r['source']}  {r['score']:.2f}  {r['text'][:60]}…")

    # 应拒答的题：这里只记录它检索到了什么（拒答与否由生成端负责，另有统计）
    for q in refuse_only:
        results = retriever.retrieve(q["question"], top_k=k)
        rows.append({
            "id": q["id"], "question": q["question"],
            "expected": [], "got": [r["source"] for r in results],
            "hit": None, "must_refuse": True,
            "top_score": round(results[0]["score"], 4) if results else 0,
        })

    recall = hits_ok / len(recallable) if recallable else 0.0

    print("\n" + "=" * 70)
    print(f"【结果】Recall@{k} = {hits_ok}/{len(recallable)} = {recall * 100:.1f}%")
    print("=" * 70)

    # 按题型给点诊断线索：失败的是哪几题
    failed = [r for r in rows if r["hit"] is False]
    if failed:
        print(f"\n没命中的 {len(failed)} 题：")
        for r in failed:
            print(f"  ❌ [{r['id']}] {r['question']}")
            print(f"       期望 {r['expected']}")
            print(f"       实际 {r['got'][0]}（最高分 {r['top_score']}）")

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "retriever": factory.RETRIEVER, "k": k,
            "recall": recall, "hits": hits_ok, "total": len(recallable),
            "rows": rows,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n原始结果已写入：{RESULT_PATH}")
    print("（这个文件就是你的'基线数据'——以后每次改检索都重跑一次，对比召回率）")


if __name__ == "__main__":
    main()
