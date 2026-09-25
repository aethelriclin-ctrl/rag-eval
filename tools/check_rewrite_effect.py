"""验证：查询改写到底有没有真的改变检索词、有没有改变检索结果。

不花钱（改写走缓存、检索是纯计算），只打印对比。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import query, factory

r = factory.load()
TOP_K = 5

# 挑几道有代表性的题：改写可能救的、可能改坏的、本来就没问题的
CASES = [
    ("q02", "评测流水线实现了哪几类评分器？"),
    ("q03", "六类失败归因机制把失败分成了哪几类？"),
    ("q14", "幻觉的触发条件是什么？"),
    ("q09", "难度分层题库里包含哪几类题目？"),
]

print("=" * 78)
print("查询改写是否真的生效？")
print("=" * 78)

for qid, q in CASES:
    try:
        new_q, how = query.rewrite(q)
    except Exception as e:
        print(f"\n[{qid}] 改写报错：{type(e).__name__}: {e}")
        continue

    changed = "不同 ✅" if new_q.strip() != q.strip() else "完全相同 ❌"

    print(f"\n[{qid}]")
    print(f"  原问题    ：{q}")
    print(f"  改写后    ：{new_q}")
    print(f"  来源      ：{how}")
    print(f"  是否改变  ：{changed}")

    h1 = r.retrieve(q, top_k=TOP_K)
    h2 = r.retrieve(new_q, top_k=TOP_K)
    s1 = [x["source"] for x in h1]
    s2 = [x["source"] for x in h2]
    same_hits = "检索结果也相同" if s1 == s2 else "⭐ 检索结果不同"

    print(f"  原问题 top1：{s1[0]}  (score {h1[0]['score']:.2f})")
    print(f"  改写后 top1：{s2[0]}  (score {h2[0]['score']:.2f})")
    print(f"  → {same_hits}")

print("\n" + "=" * 78)
print("结论：如果上面每一题都是「不同 ✅」+「检索结果不同」，")
print("      说明 --rewrite 确实在起作用，83.5% vs 83.5% 是真结果。")
print("      如果出现「完全相同 ❌」，说明改写没生效，要查缓存。")
print("=" * 78)
