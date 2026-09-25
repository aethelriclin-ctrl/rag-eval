"""用逐题指纹判定每次跑测到底是"有改写"还是"无改写"。

背景：
    eval_answer.py 原先不落盘 --rewrite 字段，所以事后分不清哪次是哪种配置。
    但检索是确定性的（改写走缓存、检索是纯计算），所以**同一配置的跑测
    在那种"改写会改变检索结果"的题上会给出相同的逐题分数**——
    这就是配置指纹。本脚本用它来还原分组，而不是靠记忆。
"""
import json
import glob
import os
import statistics as st

d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# 挑出"改写会改变检索结果"的题作为指纹位：
#   q02「哪几类评分器」、q03「失败归因」、q14「幻觉触发条件」
FINGERPRINT = ["q02", "q03", "q14"]

runs = []
for f in sorted(glob.glob(os.path.join(d, "eval_answer_*.json"))):
    j = json.load(open(f, encoding="utf-8"))
    s = j.get("summary", {})
    if s.get("answerable") != 17:
        continue
    cov = {r["id"]: r.get("kw_coverage") for r in j["rows"] if not r.get("must_refuse")}
    runs.append({
        "name": os.path.basename(f),
        "fp": tuple(cov.get(q) for q in FINGERPRINT),
        "kw": s.get("kw_coverage_mean"),
        "rewrite_field": j.get("rewrite"),      # 新加的字段（旧文件没有）
    })

print("=" * 88)
print("每次跑测的指纹（q02 / q03 / q14 的 kw_coverage）")
print("=" * 88)
print(f"{'文件':<34} {'q02':>6} {'q03':>6} {'q14':>6}   {'kw均值':>7}  rewrite字段")
for r in runs:
    q2, q3, q14 = r["fp"]
    f2 = f"{q2:.2f}" if q2 is not None else "  - "
    f3 = f"{q3:.2f}" if q3 is not None else "  - "
    f4 = f"{q14:.2f}" if q14 is not None else "  - "
    rf = r["rewrite_field"] if r["rewrite_field"] is not None else "(旧文件无)"
    print(f"{r['name']:<34} {f2:>6} {f3:>6} {f4:>6}   {r['kw']*100:>6.1f}%  {rf}")

# 按指纹分组
groups = {}
for r in runs:
    groups.setdefault(r["fp"], []).append(r)

print()
print("=" * 88)
print("按指纹自动分组")
print("=" * 88)
for fp, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    kws = [x["kw"] for x in rs]
    label = " / ".join(f"{v:.2f}" if v is not None else "-" for v in fp)
    print(f"\n指纹 [{label}]   共 {len(rs)} 次")
    print(f"  文件：{', '.join(x['name'][13:19] for x in rs)}")
    print(f"  均值：{st.mean(kws)*100:.2f}%   区间：{min(kws)*100:.1f}% ~ {max(kws)*100:.1f}%")
    if len(kws) > 1:
        print(f"  标准差：{st.stdev(kws)*100:.2f} 个点")

# 用 variance 实验做交叉验证（它无条件调用 query.rewrite，所以必然是有改写配置）
vf = os.path.join(d, "variance_20260923-235840.json")
if os.path.exists(vf):
    vj = json.load(open(vf, encoding="utf-8"))
    print()
    print("=" * 88)
    print("交叉验证：measure_variance 那次（代码里无条件调用 query.rewrite → 必然有改写）")
    print("=" * 88)
    rounds = vj.get("rounds") or vj.get("runs") or []
    if isinstance(rounds, list) and rounds:
        r0 = rounds[0]
        rows = r0.get("rows") if isinstance(r0, dict) else None
        if rows:
            cov = {x["id"]: x.get("kw_coverage") for x in rows if not x.get("must_refuse")}
            print(f"  指纹：q02={cov.get('q02')}  q03={cov.get('q03')}  q14={cov.get('q14')}")
            print("  → 与上面哪一组指纹相同，那一组就是【有改写】")
    else:
        print("  没找到轮次结构，键：", list(vj.keys())[:8])
