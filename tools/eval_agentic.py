"""第 3 层评测：Agentic RAG —— Agent 的检索决策 + 答案质量。

和第 2 层评的是什么区别：

    第 2 层：强制检索，评"检索和答案好不好"
    第 3 层：**Agent 自己决定查不查**，所以先评"决策对不对"，再评"答案好不好"

四个指标（前两个是本层独有的）：

    ① 决策正确率　　—— 该查的查了、不该查的没查
    ② 多余检索率　　—— **不该查却查了**（多此一举）
    ③ 漏查率　　　　—— **该查却没查**（更严重：会答错或编造）
    ④ 答案质量　　　—— 检索类题的要点覆盖 + 引用准确性 + 拒答率

用法：
    python tools\\eval_agentic.py            # 全量 17 题
    python tools\\eval_agentic.py --limit 4  # 只跑前 4 题（调试）
"""

import json
import os
import re
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import agent, factory  # noqa: E402
from tools.eval_answer import coverage, is_refusal, parse_citations  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_SET = os.path.join(BASE_DIR, "eval_agentic_set.json")
DATA_DIR = os.path.join(BASE_DIR, "data")


def main():
    args = sys.argv[1:]
    limit = int(args[args.index("--limit") + 1]) if "--limit" in args else None

    with open(EVAL_SET, encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    if limit:
        questions = questions[:limit]

    r = factory.load()
    if r is None:
        print("没有可用索引，先跑：python main.py --build")
        return

    print("=" * 76)
    print(f"第 3 层评测：Agentic RAG（Agent 自主决定是否检索）")
    print(f"题库 {len(questions)} 题　|　"
          f"该查 {sum(1 for q in questions if q['expect_tool'])} 题　"
          f"不该查 {sum(1 for q in questions if not q['expect_tool'])} 题")
    print("=" * 76)

    rows = []
    for q in questions:
        out = agent.run(q["question"], r)
        called = out["called_tool"]
        expect = q["expect_tool"]
        decision_ok = (called == expect)

        # 决策标签
        if expect and called:
            tag = "✅ 该查且查了"
        elif not expect and not called:
            tag = "✅ 不该查且没查"
        elif not expect and called:
            tag = "⚠️ 多余检索"          # 本层最有意思的失败
        else:
            tag = "❌ 该查却没查"          # 更严重

        print(f"\n[{q['id']}] {q['question']}")
        print(f"  决策：{tag}　（工具调用 {len(out['tool_calls'])} 次，{out['rounds']} 轮）")
        for tc in out["tool_calls"]:
            print(f"       检索词：{tc['query'][:50]}")
        print(f"  回答：{out['answer'][:110]}")

        row = {"id": q["id"], "question": q["question"], "expect_tool": expect,
               "called_tool": called, "decision_ok": decision_ok, "tag": tag,
               "answer": out["answer"], "tool_calls": out["tool_calls"],
               "must_refuse": q.get("must_refuse", False), "usage": out["usage"]}

        # ---- 答案质量（只对"该查"的题算覆盖率；不该查的题用 expected_points 看答对没）----
        pts = q.get("expected_points") or []
        if pts:
            cov, hit, miss = coverage(pts, answer=out["answer"])
            row.update({"coverage": cov, "miss": miss})
            print(f"  要点覆盖：{len(hit)}/{len(hit) + len(miss)} = {cov * 100:.0f}%"
                  f"{'　缺：' + '/'.join(miss) if miss else ''}")

        # ---- 引用准确性（只对"查了且引用了"的题算）----
        # 判据与第 2 层一致：① 编号真实存在 ② 要点是否真出现在被引用的块里。
        # ⚠️ 这里只能核对"最后一次检索"的块内容（Agent 可能检索多轮）——
        #    这是个已知的近似，会在 README 里注明。
        if called and pts:
            last = out["tool_calls"][-1] if out["tool_calls"] else None
            n_sources = len(last["sources"]) if last else 0
            cites = parse_citations(out["answer"], max(n_sources, 1))
            row["cites"] = cites
            if cites and last:
                cited_text = last.get("context", "")
                cov_in_cited, _, miss_in_cited = coverage(pts, container=cited_text)
                ok = cov_in_cited is not None and cov_in_cited >= 0.99
                row["citation_ok"] = ok
                print(f"  引用：资料 {cites} → "
                      f"{'✅ 要点都在引用块里' if ok else '❌ 要点未全在'}")
            else:
                row["citation_ok"] = None
                print("  引用：⚠️ 没写依据（本项不计入）")

        # ---- 幻觉（库里没有答案的题）----
        if q.get("must_refuse"):
            refused = is_refusal(out["answer"]) or "不知道" in out["answer"] \
                or "无法" in out["answer"] or "未提及" in out["answer"]
            row["refused"] = refused
            print(f"  幻觉检查：{'✅ 明确说不知道' if refused else '❌ 没有明确拒绝（可能编造）'}")

        rows.append(row)
        time.sleep(0.2)

    # ---------------- 汇总 ----------------
    n = len(rows)
    dec_ok = sum(1 for x in rows if x["decision_ok"])
    extra = sum(1 for x in rows if not x["expect_tool"] and x["called_tool"])
    missed = sum(1 for x in rows if x["expect_tool"] and not x["called_tool"])
    n_no_tool = sum(1 for x in rows if not x["expect_tool"])
    n_yes_tool = n - n_no_tool

    covs = [x["coverage"] for x in rows if x.get("coverage") is not None]
    refs = [x for x in rows if x.get("must_refuse")]
    cited = [x for x in rows if x.get("citation_ok") is not None]
    cited_ok = sum(1 for x in cited if x.get("citation_ok"))

    print("\n" + "=" * 76)
    print("【第 3 层结果：检索决策】")
    print(f"  ① 决策正确率　= {dec_ok}/{n} = {dec_ok / n * 100:.1f}%")
    if n_no_tool:
        print(f"  ② 多余检索率　= {extra}/{n_no_tool} = {extra / n_no_tool * 100:.1f}%"
              f"（不该查却查了）")
    if n_yes_tool:
        print(f"  ③ 漏查率　　　= {missed}/{n_yes_tool} = {missed / n_yes_tool * 100:.1f}%"
              f"（该查却没查）")

    print("\n【第 3 层结果：答案质量】")
    if covs:
        print(f"  ④ 要点覆盖率　= {statistics.mean(covs) * 100:.1f}%（{len(covs)} 题）")
    if refs:
        ok = sum(1 for x in refs if x.get("refused"))
        print(f"  ⑤ 拒答率　　　= {ok}/{len(refs)} = {ok / len(refs) * 100:.1f}%")
    if cited:
        print(f"  ⑥ 引用准确性　= {cited_ok}/{len(cited)}（{len(cited)} 题写了依据）")

    # ---- ⭐ 逐题决策表（一眼看出哪题错了）----
    print("\n【逐题决策】")
    print(f"  {'ID':<5}{'期望':<6}{'实际':<6}{'判定':<14}问题")
    for x in rows:
        exp = "该查" if x["expect_tool"] else "不该查"
        act = "查了" if x["called_tool"] else "没查"
        print(f"  {x['id']:<5}{exp:<6}{act:<6}{x['tag']:<14}{x['question'][:26]}")

    bad = [x for x in rows if not x["decision_ok"]]
    if bad:
        print(f"\n⚠️ 决策出错的题（{len(bad)} 道）：")
        for x in bad:
            print(f"  [{x['id']}] {x['tag']}　{x['question']}")
            print(f"       回答：{x['answer'][:100]}")
    print("=" * 76)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(DATA_DIR, f"agentic_{stamp}.json")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": {
            "n": n, "decision_ok": dec_ok,
            "extra_retrieval": extra, "n_no_tool": n_no_tool,
            "missed_retrieval": missed, "n_yes_tool": n_yes_tool,
            "coverage_mean": statistics.mean(covs) if covs else None,
        }, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"\n原始结果已写入：{out_path}")


if __name__ == "__main__":
    main()
