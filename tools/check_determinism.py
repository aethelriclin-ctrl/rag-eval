"""确定性检查：同一输入跑三次，结果一样吗？

为什么要查这个：
    eval_answer.py 跑了两次，代码、题库、模型都没变，但结果不一样：
        关键词判分　83.5% → 80.6%
        LLM 判分　　85.0% → 77.2%
        而且 q09 那道题一次"两个判分一致"、一次"差 100 个点"。

    **同一个实验跑两次结果不同，那这个实验的结论就不可信。**
    所以必须先搞清楚随机性从哪来。链路上有两个嫌疑：

        嫌疑① 查询改写（rag/query.py）——每题临时调一次模型，两次可能改写出不同的词
        嫌疑② 生成（rag/generator.py）——temperature=0，但项目 2 已经证明它并非完全确定

    如果①是主因 → 解法很简单：**把改写结果缓存/固化**，它就变确定了。
    如果②是主因 → 那要在结论里写明"单次跑测不可靠"，并改用多次取均值。

用法：
    python tools\\check_determinism.py           # 测 3 个问题 × 3 次
    python tools\\check_determinism.py --n 5     # 每个问题跑 5 次
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import factory, generator, query, retriever  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 挑三道题：两道"改写曾经影响很大"的，一道普通的
QUESTIONS = [
    "评测流水线实现了哪几类评分器？",
    "六类失败归因机制把失败分成了哪几类？",
    "我的评测项目一共有几道题？",
]


def main():
    args = sys.argv[1:]
    N = int(args[args.index("--n") + 1]) if "--n" in args else 3

    r = factory.load()
    if r is None:
        print("没有可用索引，先跑：python main.py --build")
        return

    print("=" * 76)
    print(f"确定性检查：同一输入重复 {N} 次，看结果是否一致")
    print("=" * 76)

    rewrite_varies = gen_varies = 0

    for q in QUESTIONS:
        print(f"\n【问题】{q}")

        # ---- 嫌疑① 查询改写 ----
        rewrites = []
        for i in range(N):
            out, how = query.rewrite(q)
            rewrites.append(out)
            print(f"  改写[{i + 1}]（{how}）：{out[:64]}")
        same_rw = len(set(rewrites)) == 1
        if not same_rw:
            rewrite_varies += 1
        print(f"  → 改写是否一致：{'✅ 是' if same_rw else '❌ 否'}")

        # ---- 嫌疑② 生成（用同一条改写结果，隔离改写的影响）----
        fixed_q = rewrites[0]
        hits = r.retrieve(fixed_q, top_k=5)
        context = retriever.build_context(hits)
        answers = []
        for i in range(N):
            ans, _ = generator.generate(q, context)
            answers.append(ans)
        same_gen = len(set(answers)) == 1
        if not same_gen:
            gen_varies += 1
        print(f"  → 生成是否一致（检索输入已固定）：{'✅ 是' if same_gen else '❌ 否'}")
        if not same_gen:
            for i, a in enumerate(answers, 1):
                print(f"      [{i}] {a[:70]}")

    print("\n" + "=" * 76)
    print("【结论】")
    print(f"  查询改写不一致的题：{rewrite_varies}/{len(QUESTIONS)}")
    print(f"  生成不一致的题　　：{gen_varies}/{len(QUESTIONS)}")
    print("=" * 76)
    print("\n怎么读：")
    print("  · 改写不一致 → 随机性来自改写 → **解法：把改写结果固化（缓存）**")
    print("  · 生成不一致 → 随机性来自生成端 → **温度=0 也不确定，结论要写'单次跑测不可靠'**")
    print("  · 两者都一致 → 那两次跑测的差异另有原因，要回去查评测脚本")

    # 落盘：文件名带时间戳，避免覆盖
    import json as _json
    import time as _time
    data_dir = os.path.join(BASE_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    stamp = _time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(data_dir, f"determinism_n{N}_{stamp}.json")
    with open(out, "w", encoding="utf-8") as f:
        _json.dump({
            "n_rounds": N,
            "questions": QUESTIONS,
            "rewrite_varies": rewrite_varies,
            "gen_varies": gen_varies,
            "total_questions": len(QUESTIONS),
        }, f, ensure_ascii=False, indent=2)
    print(f"\n原始结果已写入：{out}")


if __name__ == "__main__":
    main()
