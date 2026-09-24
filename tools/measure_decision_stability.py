"""第 3 层的稳定性测量：Agent 的"该不该查"判断会翻转吗？

为什么单独测这个（2026-09-24 的实测驱动）：
    eval_agentic.py 跑了两次，代码、题库、模型都没变，但结果不同：
        决策正确率　94.1% → 100%
        漏查率　　　9.1% → 0%
    也就是说——**同一道题，一次查了、一次没查**。

    "生成的答案会变"我们在第 2 层已经见过（措辞不同）。
    但“**该不该查这个决定本身**会变”，性质更严重：
    它是**控制决策层面的不稳定**，不是文本层面的。

    所以本脚本：同一批题跑 N 轮，**只看"该不该查"的判断翻不翻转**，不重复算答案质量
    （省钱省时间——答案质量的波动已由 measure_variance.py 测过）。

用法：
    python tools\\measure_decision_stability.py           # 3 轮
    python tools\\measure_decision_stability.py --n 5     # 5 轮
"""

import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import agent, factory  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_SET = os.path.join(BASE_DIR, "eval_agentic_set.json")
DATA_DIR = os.path.join(BASE_DIR, "data")


def main():
    args = sys.argv[1:]
    N = int(args[args.index("--n") + 1]) if "--n" in args else 3

    with open(EVAL_SET, encoding="utf-8") as f:
        questions = json.load(f)["questions"]

    r = factory.load()
    if r is None:
        print("没有可用索引，先跑：python main.py --build")
        return

    print("=" * 76)
    print(f"决策稳定性测量：每題跑 {N} 轮，看「该不该查」的判断是否翻转")
    print(f"共 {len(questions)} 题 × {N} 轮 = {len(questions) * N} 次 Agent 调用")
    print("=" * 76)

    records = []
    for q in questions:
        calls, rounds_used = [], []
        for i in range(N):
            out = agent.run(q["question"], r)
            calls.append(out["called_tool"])
            rounds_used.append(out["rounds"])

        flips = len(set(calls)) > 1

        marks = " ".join("查" if c else "不查" for c in calls)
        flag = "⚠️ 翻转" if flips else "✅ 稳定"
        print(f"  [{q['id']:>4}] {marks:<{N * 4}}  {flag}   期望="
              f"{'该查' if q['expect_tool'] else '不该查'}　{q['question'][:24]}")

        records.append({
            "id": q["id"], "question": q["question"],
            "expect_tool": q["expect_tool"],
            "calls": calls, "flips": flips, "rounds": rounds_used,
        })

    # ---------------- 汇总 ----------------
    flips = [x for x in records if x["flips"]]
    n = len(records)

    # 每轮的决策正确率（看整体波动）
    round_acc = []
    for i in range(N):
        ok = sum(1 for x in records if x["calls"][i] == x["expect_tool"])
        round_acc.append(ok / n)

    print("\n" + "=" * 76)
    print("【决策稳定性结果】")
    print(f"  判断会翻转的题：{len(flips)}/{n} = {len(flips) / n * 100:.1f}%")
    print(f"  每轮决策正确率：{' / '.join(f'{a * 100:.1f}%' for a in round_acc)}")
    print(f"  → 均值 {statistics.mean(round_acc) * 100:.1f}%　"
          f"区间 {min(round_acc) * 100:.1f}%~{max(round_acc) * 100:.1f}%"
          f"（跨度 {(max(round_acc) - min(round_acc)) * 100:.1f} 个点）")

    if flips:
        print(f"\n⚠️ 翻转的题（{len(flips)} 道）：")
        for x in flips:
            marks = " / ".join("查了" if c else "没查" for c in x["calls"])
            print(f"  [{x['id']}] 期望{'该查' if x['expect_tool'] else '不该查'}　"
                  f"{marks}　{x['question'][:28]}")
    else:
        print("\n✅ 所有题的判断在 N 轮里完全一致——决策是稳定的")

    print("=" * 76)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(DATA_DIR, f"decision_stability_{stamp}.json")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"n_rounds": N, "flips": len(flips), "total": n,
                   "round_accuracy": round_acc, "records": records},
                  f, ensure_ascii=False, indent=2)
    print(f"\n原始结果已写入：{out_path}")


if __name__ == "__main__":
    main()
