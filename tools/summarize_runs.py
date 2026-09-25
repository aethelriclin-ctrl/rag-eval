"""汇总所有 17 题跑测，按"有无改写"分组（用已知的跑测记录标注）。"""
import json, glob, os, statistics as st

d = r"D:\vscode\rag-eval\data"

# 已知的跑测记录：文件名 -> 是否开启 --rewrite
# （脚本目前不落盘这个字段，所以只能人工标注——这正是要修的洞）
KNOWN = {
    "eval_answer_20260922-234011.json": False,
    "eval_answer_20260922-234412.json": False,
    "eval_answer_20260922-235339.json": False,
    "eval_answer_20260923-235010.json": False,
    "eval_answer_20260925-231822.json": False,   # 他刚跑的无改写基线
}

runs = []
for f in sorted(glob.glob(os.path.join(d, "eval_answer_*.json"))):
    name = os.path.basename(f)
    try:
        j = json.load(open(f, encoding="utf-8"))
    except Exception:
        continue
    s = j.get("summary", {})
    if s.get("answerable") != 17:
        continue
    runs.append({
        "name": name,
        "kw": s.get("kw_coverage_mean"),
        "cite": f"{s.get('citation_ok')}/{s.get('citation_total')}",
        "refuse": f"{s.get('refused_ok')}/{s.get('refusable')}",
        "rewrite": KNOWN.get(name, None),
    })

base = [r for r in runs if r["rewrite"] is False]
unk = [r for r in runs if r["rewrite"] is None]

print("=" * 76)
print(f"共 {len(runs)} 次 17 题跑测")
print("=" * 76)
for r in runs:
    tag = "无改写" if r["rewrite"] is False else "有改写" if r["rewrite"] is True else "未标注"
    print(f"  {r['name']}   kw={r['kw']*100:5.1f}%   cite={r['cite']:5s}  refuse={r['refuse']:3s}  [{tag}]")

if unk:
    print(f"\n⚠️ 有 {len(unk)} 次跑测无法从文件判断是哪种配置（脚本没记 --rewrite 字段）")
    print("   按已知规律推断：两种配置的结果都落在 80.6~86.5，无法靠数值反推。")
    print("   → 所以要改成先补上落盘字段，再重跑；或人工回忆。")
    print("   本次先剔除这些，只用能确定的。")

b = [r["kw"] for r in base]
print("\n" + "=" * 76)
print(f"无改写组（{len(b)} 次）：", ", ".join(f"{x*100:.1f}%" for x in b))
if b:
    print(f"  均值 = {st.mean(b)*100:.2f}%   范围 = {min(b)*100:.1f}% ~ {max(b)*100:.1f}%")
    if len(b) > 1:
        print(f"  标准差 = {st.stdev(b)*100:.2f} 个点")
print("=" * 76)
