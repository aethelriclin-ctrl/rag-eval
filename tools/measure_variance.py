"""生成端波动测量：检索固定，只重复生成，看答案质量波动多大。

为什么单独做这件事：
    第 2 层跑两次，数字不一样（关键词 83.5% → 80.6%，LLM 85.0% → 77.2%）。
    查下来有两个随机源：
        ① 查询改写 —— **已用缓存固化**（check_determinism 验证：0/3 不一致）
        ② 生成端 —— temperature=0 仍不确定（3/3 不一致）

    所以现在只剩②。与其把整条评测跑三遍（每题 3 次生成 + 3 次裁判，很慢），
    不如**把改写和检索固定，只重复生成**：
        · 改写：读缓存 → 确定
        · 检索：同样输入 → 确定（BM25 是纯计算）
        · 生成：变量 ← 只测它

    这样 20 题 × N 轮只有 20N 次生成调用，比跑 N 遍完整评测快得多，而且更干净。

用法：
    python tools\\measure_variance.py           # 每题生成 3 次
    python tools\\measure_variance.py --n 5     # 每題生成 5 次
"""

import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import factory, generator, query, retriever  # noqa: E402
from tools.eval_answer import coverage, is_refusal  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_SET = os.path.join(BASE_DIR, "eval_set.json")
DATA_DIR = os.path.join(BASE_DIR, "data")
TOP_K = 5


def main():
    args = sys.argv[1:]
    N = int(args[args.index("--n") + 1]) if "--n" in args else 3

    with open(EVAL_SET, encoding="utf-8") as f:
        questions = json.load(f)["questions"]

    r = factory.load()
    if r is None:
        print("没有可用索引，先跑：python main.py --build")
        return

    answerable = [q for q in questions if not q["must_refuse"]]
    refusable = [q for q in questions if q["must_refuse"]]

    print("=" * 74)
    print(f"生成端波动测量　|　每题生成 {N} 次　|　改写读缓存、检索固定")
    print(f"共 {len(questions)} 题 × {N} 轮 = {len(questions) * N} 次生成调用")
    print("=" * 74)

    per_question = []      # 每题的覆盖率列表
    refuse_counts = []     # 应拒答题：N 轮里拒了几次

    for q in questions:
        search_q, how = query.rewrite(q["question"])
        hits = r.retrieve(search_q, top_k=TOP_K)
        context = retriever.build_context(hits)

        covs = []
        for i in range(N):
            ans, _ = generator.generate(q["question"], context)
            if q["must_refuse"]:
                covs.append(1.0 if is_refusal(ans) else 0.0)
            else:
                cov, _, _ = coverage(q["expected_points"], answer=ans)
                covs.append(cov)

        # 每题一行进度（可答题显示覆盖率，应拒答题显示拒了几次）
        if q["must_refuse"]:
            print(f"  [{q['id']:>4}] 拒答 {int(sum(covs))}/{N}　{q['question'][:28]}")
        else:
            marks = " ".join(f"{c * 100:>3.0f}%" for c in covs)
            print(f"  [{q['id']:>4}] {marks}　{q['question'][:28]}")

        per_question.append((q["id"], q["must_refuse"], covs))

    # ---------------- 汇总 ----------------
    ans_rows = [(i, c) for i, must, c in per_question if not must]
    ref_rows = [(i, c) for i, must, c in per_question if must]

    print("\n" + "=" * 74)
    print("【结果：答案正确率（关键词判分）】")
    round_means = [statistics.mean(c[i] for _, c in ans_rows) for i in range(N)]
    for i, m in enumerate(round_means, 1):
        print(f"  第 {i} 轮：{m * 100:.1f}%")
    print(f"  → 均值　　：{statistics.mean(round_means) * 100:.1f}%")
    print(f"  → 区间　　：{min(round_means) * 100:.1f}% ~ {max(round_means) * 100:.1f}%"
          f"（跨度 {(max(round_means) - min(round_means)) * 100:.1f} 个点）")

    # 逐题波动：哪些题不稳定
    unstable = [(i, c) for i, c in ans_rows if len(set(c)) > 1]
    print(f"\n  逐题看：{len(unstable)}/{len(ans_rows)} 题的覆盖在 {N} 轮里变过")
    for i, c in unstable[:6]:
        print(f"     [{i}] {' / '.join(f'{x * 100:.0f}%' for x in c)}")

    if ref_rows:
        print("\n【结果：拒答率（1 - 幻觉率）】")
        for i, c in ref_rows:
            print(f"  [{i}] {N} 轮里拒答 {int(sum(c))}/{N}")

    print("=" * 74)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(DATA_DIR, f"variance_{stamp}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "n_rounds": N,
            "round_means": round_means,
            "mean": statistics.mean(round_means),
            "per_question": [{"id": i, "must_refuse": m, "coverages": c}
                             for i, m, c in per_question],
        }, f, ensure_ascii=False, indent=2)
    print(f"\n原始结果已写入：{out}")
    print("\n怎么用这个数字：")
    print("  报告里不要写单次结果，写成『均值 X%（N 轮区间 Y%~Z%）』——")
    print("  因为 temperature=0 不等于确定，单次跑测的数字没有代表性。")


if __name__ == "__main__":
    main()
