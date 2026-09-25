"""检索方式对照实验：BM25（字面） vs 向量（语义）—— 一条命令跑完。

为什么做这个（这是项目 4 悬了很久的那块拼图）：
    第 1 层的结论是"BM25 扛得住句式改写、扛不住关键词替换"——
    两个失败案例都指向同一个病：
        · "那八道题都是什么类型的？"   → 字面对不上
        · "换一种打分方法，结果会变吗？" → 文档里写的是"判分"
    **而"换了说法就找不到"正是语义检索该解决的问题。**
    所以这组对照的看点很明确：**向量能不能把那 7.1% 的缺口补回来？**

它会自动做完四步（你不用管顺序，也不用设环境变量）：
    ① 建 BM25 索引（纯本地，不花钱）
    ② 建向量索引（调一次 embedding，但会缓存）
    ③ 两套评测集各跑两遍：Recall@3 + 措辞敏感性
    ④ 并排列出结果，并标出"谁赢在哪题"

用法：
    python tools\\compare_retrieval.py

前置条件：
    $env:DASHSCOPE_API_KEY="sk-..."   # 阿里云百炼的 key（向量检索要用）
    pip install numpy
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import bm25, query, splitter, vector_retriever  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
EVAL_SET = os.path.join(BASE_DIR, "eval_set.json")
PHRASING_SET = os.path.join(BASE_DIR, "eval_phrasing_set.json")

K = 3


def build_both():
    """建两个索引：BM25（本地）+ 向量（调 embedding，带缓存）。"""
    chunks = splitter.load_all_docs(DOCS_DIR)
    if not chunks:
        print("⚠️ docs/ 是空的，先跑：python setup.py")
        return None, None
    print(f"知识库：{len(chunks)} 块\n")

    print("① 建 BM25 索引（本地，不花钱）")
    r_bm25 = bm25.BM25Retriever()
    r_bm25.build(chunks)

    print("\n② 建向量索引（调 embedding，首次会花一点额度，之后读缓存）")
    r_vec = vector_retriever.VectorRetriever.load()
    if r_vec is None:
        r_vec = vector_retriever.VectorRetriever()
        r_vec.build(chunks)
    else:
        print(f"   已读缓存：{len(r_vec.items)} 块")

    return r_bm25, r_vec


def eval_recall(retriever, questions, k=K, use_rewrite=False):
    """算 Recall@k。返回 (命中数, 总数, 逐题明细)"""
    hit_n = 0
    details = []
    for q in questions:
        search_q = q["question"]
        if use_rewrite:
            search_q, _ = query.rewrite(q["question"])
        hits = retriever.retrieve(search_q, top_k=k)
        got = [h["source"] for h in hits]
        ok = any(s in q["expected_sources"] for s in got)
        hit_n += 1 if ok else 0
        details.append({"id": q["id"], "question": q["question"],
                        "hit": ok, "got": got[:1],
                        "top_score": round(hits[0]["score"], 3) if hits else 0})
    return hit_n, len(questions), details


def eval_phrasing(retriever, groups, k=K):
    """算措辞敏感性：按"第几种问法"统计命中率，并记录逐组明细。

    为什么要逐组明细：
        28 个变体里只差 1 题就是 3.6 个百分点——**光看总分判断不了"这个差异是不是噪声"**。
        但如果按组看：同一组里"赢一题又输一题"很可能是噪声；
        不同组各赢一处，才更像"强项互补"这个真结论。
    """
    by_pos = {}
    total = hit_total = 0
    per_group = []          # 每组的命中情况：[{gid, topic, hits:[bool,...]}]
    for g in groups:
        marks = []
        for idx, q in enumerate(g["variants"]):
            hits = retriever.retrieve(q, top_k=k)
            got = [h["source"] for h in hits]
            ok = any(s in g["expected_sources"] for s in got)
            total += 1
            hit_total += 1 if ok else 0
            marks.append(ok)
            by_pos.setdefault(idx, [0, 0])
            by_pos[idx][1] += 1
            if ok:
                by_pos[idx][0] += 1
        per_group.append({"gid": g["id"], "topic": g["topic"], "marks": marks})
    return hit_total, total, by_pos, per_group


def main():
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("⚠️ 没读到 DASHSCOPE_API_KEY —— 向量检索需要它（阿里云百炼的 key）。")
        print('   先执行：$env:DASHSCOPE_API_KEY="sk-..."')
        return

    with open(EVAL_SET, encoding="utf-8") as f:
        questions = [q for q in json.load(f)["questions"] if q["expected_sources"]]
    with open(PHRASING_SET, encoding="utf-8") as f:
        groups = json.load(f)["groups"]

    print("=" * 78)
    print("检索方式对照：BM25（字面匹配） vs 向量（语义匹配）")
    print("=" * 78 + "\n")

    r_bm25, r_vec = build_both()
    if r_bm25 is None:
        return

    results = {}
    for name, r in (("BM25", r_bm25), ("向量", r_vec)):
        print(f"\n{'—' * 78}\n【{name}】")
        n1, t1, d1 = eval_recall(r, questions)
        n2, t2, pos2, grp2 = eval_phrasing(r, groups)
        results[name] = {"recall": (n1, t1, d1),
                         "phrasing": (n2, t2, pos2), "groups": grp2}
        print(f"  Recall@{K}（同义问法）　= {n1}/{t1} = {n1 / t1 * 100:.1f}%")
        print(f"  Recall@{K}（口语改写）　= {n2}/{t2} = {n2 / t2 * 100:.1f}%")

    # ---------------- 并排汇总 ----------------
    print("\n" + "=" * 78)
    print("【并排对照】")
    print(f"  {'指标':<26}{'BM25':<16}{'向量':<16}谁赢")
    rows = []
    b1, v1 = results["BM25"]["recall"], results["向量"]["recall"]
    b2, v2 = results["BM25"]["phrasing"], results["向量"]["phrasing"]

    def line(label, a, b, ta, tb):
        who = "平" if a == b else ("向量 ✅" if b > a else "BM25 ✅")
        print(f"  {label:<26}{f'{a}/{ta} = {a / ta * 100:.1f}%':<16}"
              f"{f'{b}/{tb} = {b / tb * 100:.1f}%':<16}{who}")
        rows.append({"metric": label, "bm25": a / ta, "vector": b / tb})

    line(f"Recall@{K} 同义问法", b1[0], v1[0], b1[1], v1[1])
    line(f"Recall@{K} 口语改写", b2[0], v2[0], b2[1], v2[1])
    for idx in sorted(b2[2]):
        ba, bn = b2[2][idx]
        va, vn = v2[2][idx]
        line(f"  └ 第 {idx} 种问法", ba, va, bn, vn)

    # ---------------- 逐题差异 ----------------
    print("\n【逐题差异】（只看两套检索结果不同的题）")
    diff_count = 0
    for x, y in zip(b1[2], v1[2]):
        if x["hit"] != y["hit"]:
            diff_count += 1
            print(f"  [{x['id']}] {x['question'][:34]}")
            print(f"       BM25 {'✅' if x['hit'] else '❌'} (top1: {x['got']})"
                  f"　向量 {'✅' if y['hit'] else '❌'} (top1: {y['got']})")
    if diff_count == 0:
        print("  没有差异——两套检索在这批题上表现相同")

    # ---------------- ⭐ 措辞测试的逐组差异（判断"是不是噪声"的关键）----------------
    print("\n【措辞测试 · 逐组差异】")
    print("  （同一组里'赢一则输一则'很可能是噪声；不同组各赢一处，才像真差异）")
    bm_groups = {g["gid"]: g for g in results["BM25"]["groups"]}
    vc_groups = {g["gid"]: g for g in results["向量"]["groups"]}
    same_group_flip = 0
    cross_group = []
    for gid in bm_groups:
        bm_m = bm_groups[gid]["marks"]
        vc_m = vc_groups.get(gid, {}).get("marks", [])
        if bm_m == vc_m:
            continue
        bm_win = sum(1 for a, b in zip(bm_m, vc_m) if a and not b)
        vc_win = sum(1 for a, b in zip(bm_m, vc_m) if b and not a)
        topic = bm_groups[gid]["topic"]
        print(f"  [{gid}] {topic}")
        print(f"       BM25 {''.join('✅' if m else '❌' for m in bm_m)}"
              f"　向量 {''.join('✅' if m else '❌' for m in vc_m)}")
        print(f"       → BM25 赢 {bm_win} 处，向量赢 {vc_win} 处")
        if bm_win and vc_win:
            same_group_flip += 1
        else:
            cross_group.append(f"{gid}({'向量' if vc_win else 'BM25'}赢)")
    if not cross_group and not same_group_flip:
        print("  没有差异")
    else:
        print(f"\n  ⭐ 判读：")
        if same_group_flip:
            print(f"     · 有 {same_group_flip} 组出现'同一组内互有胜负' → 这部分很可能是噪声")
        if cross_group:
            print(f"     · 另有 {len(cross_group)} 组是单向胜负：{', '.join(cross_group)}"
                  f" → 这部分更像真实差异")

    print("\n" + "=" * 78)
    print("怎么读这个结果：")
    print("  · 向量在'口语改写'上更高 → 说明语义检索补上了字面匹配的缺口（预期结论）")
    print("  · BM25 在'同义问法'上不输 → 说明精确术语场景下字面匹配依然够用")
    print("  · 两者各有胜负 → 那就该考虑混合检索（Hybrid），而不是二选一")
    print("  · ⚠️ 样本只有 28 个变体，差 1 题 = 3.6 个点 —— 小幅差异不足以当结论")

    out = os.path.join(BASE_DIR, "data", "compare_retrieval_result.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"k": K, "rows": rows,
                   "bm25_detail": b1[2], "vector_detail": v1[2],
                   "bm25_groups": results["BM25"]["groups"],
                   "vector_groups": results["向量"]["groups"]},
                  f, ensure_ascii=False, indent=2)
    print(f"\n原始结果已写入：{out}")


if __name__ == "__main__":
    main()
