"""对比"有改写"与"无改写"两次 17 题跑测，逐题比对。"""
import json, hashlib, glob, os

d = r"D:\vscode\rag-eval\data"
files = sorted(glob.glob(os.path.join(d, "eval_answer_*.json")))

runs = []
for f in files:
    try:
        j = json.load(open(f, encoding="utf-8"))
    except Exception as e:
        print("  [读不了]", os.path.basename(f), e)
        continue
    s = j.get("summary", {})
    if s.get("answerable") != 17:
        continue
    runs.append((os.path.basename(f), j, s))

print(f"共 {len(runs)} 次 17 题跑测（按文件名/时间排序）\n")
for name, j, s in runs:
    print(f"{name}")
    print(f"   retriever={j.get('retriever')!r}  top_k={j.get('top_k')!r}  llm_judge={j.get('llm_judge')!r}")
    print(f"   kw_mean={s.get('kw_coverage_mean'):.4f}  cite={s.get('citation_ok')}/{s.get('citation_total')}"
          f"  refuse={s.get('refused_ok')}/{s.get('refusable')}")

if len(runs) >= 2:
    print("\n" + "=" * 70)
    print("逐题对比：按时间顺序，列出每次的 kw_coverage")
    print("=" * 70)
    ids = [r["id"] for r in runs[0][1]["rows"] if not r.get("must_refuse")]
    header = "题号  " + "  ".join(f"{n[13:19]}" for n, _, _ in runs)
    print(header)
    for qid in ids:
        vals = []
        for _, j, _ in runs:
            row = next((r for r in j["rows"] if r["id"] == qid), None)
            vals.append(f"{row['kw_coverage']:.2f}" if row else " -- ")
        print(f"{qid}   " + "    ".join(vals))

    print("\n" + "=" * 70)
    print("回答文本是否完全相同（md5 前 8 位）")
    print("=" * 70)
    for qid in ids:
        hs = []
        for _, j, _ in runs:
            row = next((r for r in j["rows"] if r["id"] == qid), None)
            a = (row.get("answer") or "") if row else ""
            hs.append(hashlib.md5(a.encode("utf-8")).hexdigest()[:8])
        same = "完全相同!!" if len(set(hs)) == 1 else ""
        print(f"{qid}   " + "  ".join(hs) + f"   {same}")
