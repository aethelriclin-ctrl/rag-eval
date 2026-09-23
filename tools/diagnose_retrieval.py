"""检索诊断：定位"答案块到底排第几"。

为什么要写这个：
    eval_retrieval.py 报 Recall@3 = 100%，但 eval_answer.py 里模型反复说"资料中没有"。
    两个数字矛盾，所以必须先查清楚：是【检索没拿到答案块】，还是【模型没用上】。

    第 1 层的 Recall 判据是"top-k 里有没有来自正确【文件】的块"——
    这个判据太松：一个文件被切成 30 块，混进任意一块就算命中。
    本脚本改用更严的判据：**含答案要点的【块】排第几**。

用法：
    python tools\\diagnose_retrieval.py
    python tools\\diagnose_retrieval.py --k 20      # 看放宽到 20 块时能不能覆盖到
    python tools\\diagnose_retrieval.py --rewrite  # ⭐ 开启查询改写后再测（对照实验）
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import factory, query  # noqa: E402
from tools.eval_answer import point_hit  # noqa: E402  （复用同一套判据）

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_SET = os.path.join(BASE_DIR, "eval_set.json")


def main():
    args = sys.argv[1:]
    K = int(args[args.index("--k") + 1]) if "--k" in args else 10
    use_rewrite = "--rewrite" in args
    detail = "--detail" in args          # 默认只打印每题一行，加这个才看逐要点细节

    with open(EVAL_SET, encoding="utf-8") as f:
        questions = [q for q in json.load(f)["questions"] if not q["must_refuse"]]

    r = factory.load()
    if r is None:
        print("没有可用索引，先跑：python main.py --build")
        return

    items = r.items          # [{"source":..., "text":...}]，顺序 = 索引里的块号
    total_pts = 0
    found_in_index = 0        # 要点是否存在于索引的某个块里
    in_topk = 0               # 含该要点的块是否出现在 top-K

    print("=" * 78)
    print(f"检索诊断：含答案要点的【块】排第几？　（检索={factory.RETRIEVER}，考察 top-{K}）")
    print(f"查询改写：{'✅ 开启' if use_rewrite else '❌ 关闭'}")
    print("=" * 78)

    ranks_all = []
    # ⭐ 最关键的数字：逐题算"实际 top-K 里覆盖了多少要点"
    topk_pts_hit = topk_pts_total = 0

    for q in questions:
        # 检索用的查询：开改写则先改写
        search_q = q["question"]
        if use_rewrite:
            search_q, how = query.rewrite(q["question"])
        hits = r.retrieve(search_q, top_k=len(items))     # 全量排序，看真实排名
        rank_of = {h["text"]: i + 1 for i, h in enumerate(hits)}
        actual_context = "\n".join(h["text"] for h in hits[:K])

        q_hit = q_total = 0
        lines = []
        for grp in q["expected_points"]:
            total_pts += 1
            q_total += 1
            topk_pts_total += 1
            holders = [i for i, it in enumerate(items) if point_hit(grp, it["text"])]
            if not holders:
                lines.append(f"      ❌ {grp[0]:<10} 索引里没有任何块含此要点")
                continue
            found_in_index += 1
            best = min(rank_of.get(items[i]["text"], 10 ** 9) for i in holders)
            best = None if best >= 10 ** 9 else best
            ranks_all.append(best if best else 9999)
            in_actual = point_hit(grp, actual_context)
            if in_actual:
                topk_pts_hit += 1
                in_topk += 1
                q_hit += 1
            lines.append(f"      {'✅' if in_actual else '⚠️'} {grp[0]:<10} "
                         f"最好排名 {best}"
                         f"{'　|　在 top-' + str(K) if in_actual else '　|　【不在】top-' + str(K)}")

        # 一行汇总（默认输出）
        flag = "✅" if q_hit == q_total else ("⚠️" if q_hit else "❌")
        print(f"\n{flag} [{q['id']}] {q['question'][:34]:<34} "
              f"top-{K} 覆盖 {q_hit}/{q_total}")
        if use_rewrite:
            print(f"     检索词：{search_q[:70]}")
        if detail:
            for ln in lines:
                print(ln)

    print("\n" + "=" * 78)
    print("【诊断结果】")
    print(f"  要点总数　　　　　　　　：{total_pts}")
    print(f"  存在于索引中的要点　　　：{found_in_index} "
          f"（{found_in_index / total_pts * 100:.0f}%）")
    print(f"  ⭐ 实际 top-{K} 内容覆盖的要点：{topk_pts_hit}/{topk_pts_total} "
          f"（{topk_pts_hit / topk_pts_total * 100:.0f}%）")
    if ranks_all:
        valid = sorted(x for x in ranks_all if x < 9999)
        if valid:
            print(f"  含答案块的最好排名分布　：最小 {valid[0]}，"
                  f"中位 {valid[len(valid) // 2]}，最大 {valid[-1]}")
    print("=" * 78)
    print("\n怎么看这个结果：")
    print("  · 若「存在于索引」低 → 问题在【切分】：要点被切散了，或原文里就没有")
    print("  · 若「⭐ 实际 top-K 覆盖」低 → 问题在【排序】：好块被挤出去了")
    print("  · 若「⭐ 覆盖高但模型仍答不出」 → 问题在【生成端提示词】")


if __name__ == "__main__":
    main()
