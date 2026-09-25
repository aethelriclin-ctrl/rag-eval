"""第 2 层评测：答案质量（三个指标）。

第 1 层（eval_retrieval.py）只量了"检索找得准不准"。但检索对了 ≠ 答对了 ——
冒烟测试里就出现过：检索命中了三个相关块，模型却答错了内容。

所以这一层量三件事：

    ① 答案正确率　—— 期望要点覆盖了多少（关键词判分；加 --llm 可换 LLM 判分对照）
    ② 引用准确性　—— 它说"依据资料 1、3"，这个引用靠不靠得住
    ③ 幻觉率　　　—— 资料里没有答案的题，它编了几次

⚠️ 关于指标 ② 的判据（这是本脚本最需要说清的地方）：
    "引用准确"分两层：
      (a) 它引用的编号是否真实存在（不能引用"资料 9"，如果只检索到 5 块）
      (b) 它答出的要点，是否真的出现在它引用的那些块里
    两层都过才算准确。只查 (a) 太松——编造内容再随便挂个编号也能过。

用法：
    python tools\\eval_answer.py            # 只跑关键词判分（便宜）
    python tools\\eval_answer.py --llm      # 加一轮 LLM 判分做对照（每题多 1 次调用）
    python tools\\eval_answer.py --rewrite  # ⭐ 检索前先做查询改写（对照实验）
    python tools\\eval_answer.py --limit 3  # 只跑前 3 题（调试用）
"""

import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import factory, generator, query, retriever  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_SET = os.path.join(BASE_DIR, "eval_set.json")
DATA_DIR = os.path.join(BASE_DIR, "data")

TOP_K = 5          # 给生成端多喂一点资料（评测关心最终答案，不是检索精度）

# 模型表示"资料里没有"的常见说法 —— 用来判断它有没有拒答
REFUSE_MARKERS = [
    "资料中没有", "没有相关信息", "未提及", "没有提及", "无法回答", "无法确定",
    "资料未", "没有找到", "未找到相关", "资料里没有", "未包含", "不包含",
]


def strip_punct(s):
    """去掉空白与常见标点，减少匹配噪声。"""
    return re.sub(r"[\s，。、；：！？,.;:!?（）()\"'`]", "", s)


def point_hit(point_group, text):
    """一个要点（关键词组）是否在 text 里命中：组内任一关键词出现即算命中。"""
    t = strip_punct(text)
    return any(strip_punct(kw) in t for kw in point_group)


def coverage(expected_points, answer=None, container=None):
    """算要点覆盖率。

    - 算"整篇回答覆盖了多少要点"：传 answer
    - 算"引用的那些块里覆盖了多少要点"：传 container
    （两者选一，container 优先。这也正是引用准确性指标的判据。）
    """
    text = container if container is not None else (answer or "")
    if not expected_points:
        return None, [], []
    hit, miss = [], []
    for grp in expected_points:
        (hit if point_hit(grp, text) else miss).append(grp[0])
    return len(hit) / len(expected_points), hit, miss


def parse_citations(answer, n_sources):
    """从回答里解析出它声称引用的资料编号。

    能认出这些写法：依据资料 1、3 ／ 资料1和资料2 ／ 根据资料 1,2 ／ (资料 3)
    返回去重后的编号列表。
    """
    cites = set()
    # 找出所有"资料"附近出现的数字
    for m in re.finditer(r"资料\s*([0-9０-９][0-9０-９\s、,，和与及\-~]*[0-9０-９]?)", answer):
        seg = m.group(1)
        for num in re.findall(r"[0-9０-９]+", seg):
            n = int(num.translate(str.maketrans("０１２３４５６７８９", "0123456789")))
            if 1 <= n <= n_sources:
                cites.add(n)
    return sorted(cites)


def is_refusal(answer):
    """模型是否明确表示'资料里没有'。"""
    return any(m in answer for m in REFUSE_MARKERS)


# ---------------- LLM 判分（对照用） ----------------

JUDGE_PROMPT = """你是评测裁判。请判断【模型回答】是否覆盖了【期望要点】。

只输出 JSON，不要任何解释：
{"covered": 0到1之间的小数, "missing": ["未覆盖的要点，最多3条"], "reason": "一句话理由"}

判断标准：
- 只判断【期望要点】是否被回答覆盖，不评价措辞是否优美
- 同义表述算覆盖（如"未找到"与"不存在"是同一个意思）
- 如果回答声称"资料中没有"但期望要点存在，covered 记 0
"""


def llm_judge(question, expected_points, answer):
    """用模型当裁判算覆盖率。失败返回 None。"""
    try:
        client = generator.get_client()
        resp = client.chat.completions.create(
            model="deepseek-flash",
            messages=[{
                "role": "user",
                "content": (f"{JUDGE_PROMPT}\n\n【问题】\n{question}\n\n"
                            f"【期望要点】\n{json.dumps(expected_points, ensure_ascii=False)}\n\n"
                            f"【模型回答】\n{answer}"),
            }],
            temperature=0,
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw).replace("```", "").strip()
        s, e = raw.find("{"), raw.rfind("}")
        return json.loads(raw[s:e + 1])
    except Exception as ex:
        print(f"      （LLM 判分失败：{str(ex)[:80]}）")
        return None


def main():
    args = sys.argv[1:]
    use_llm = "--llm" in args
    use_rewrite = "--rewrite" in args
    limit = int(args[args.index("--limit") + 1]) if "--limit" in args else None

    with open(EVAL_SET, encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    if limit:
        questions = questions[:limit]

    r = factory.load()
    if r is None:
        print("没有可用索引，先跑：python main.py --build")
        return

    answerable = [q for q in questions if not q["must_refuse"]]
    refusable = [q for q in questions if q["must_refuse"]]

    print("=" * 74)
    print(f"第 2 层评测：答案质量　|　检索方式：{factory.RETRIEVER}　|　top_k={TOP_K}"
          f"{'　+ LLM 判分' if use_llm else ''}")
    print(f"题库 {len(questions)} 题（可答 {len(answerable)}，应拒答 {len(refusable)}）")
    print("=" * 74)

    rows = []
    kw_cov_sum = llm_cov_sum = 0.0
    cite_ok = cite_total = 0
    refused_ok = 0

    for q in questions:
        print(f"\n[{q['id']}] {q['question']}")
        # 检索用的查询：开改写则先改写（改写只用于检索，生成仍用原问题）
        search_q = q["question"]
        if use_rewrite:
            search_q, how = query.rewrite(q["question"])
            print(f"  改写检索词（{how}）：{search_q}")

        hits = r.retrieve(search_q, top_k=TOP_K)
        context = retriever.build_context(hits)

        # ---- 检索诊断：把实际拿到的块打出来（判断"拒答"的根因在这里）----
        print("  检索命中：")
        for i, h in enumerate(hits, 1):
            print(f"    [{i}] {h['source']}  {h['score']:.2f}  {h['text'][:50].replace(chr(10), ' ')}…")

        answer, usage = generator.generate(q["question"], context)
        print(f"  回答：{answer[:150]}{'…' if len(answer) > 150 else ''}")

        row = {"id": q["id"], "question": q["question"], "answer": answer,
               "must_refuse": q["must_refuse"], "usage": usage}

        # ---- 应拒答的题：只看有没有编 ----
        if q["must_refuse"]:
            refused = is_refusal(answer)
            refused_ok += 1 if refused else 0
            row["refused"] = refused
            print(f"  幻觉检查：{'✅ 明确说资料里没有' if refused else '❌ 没有拒答（可能编造）'}")
            rows.append(row)
            continue

        # ---- 指标 ① 答案正确率（关键词判分）----
        cov, hit, miss = coverage(q["expected_points"], answer=answer)
        kw_cov_sum += cov
        row.update({"kw_coverage": cov, "kw_hit": hit, "kw_miss": miss})
        print(f"  ① 关键词覆盖：{len(hit)}/{len(hit) + len(miss)} = {cov * 100:.0f}%"
              f"{'　缺：' + '/'.join(miss) if miss else ''}")

        # ---- 指标 ② 引用准确性 ----
        cites = parse_citations(answer, len(hits))
        cited_text = "\n".join(hits[i - 1]["text"] for i in cites) if cites else ""
        if cites:
            # (a) 编号真实存在（parse 时已过滤）(b) 要点是否真出现在引用的块里
            cov_in_cited, _, miss_in_cited = coverage(
                q["expected_points"], container=cited_text)
            ok = cov_in_cited is not None and cov_in_cited >= 0.99
            cite_total += 1
            cite_ok += 1 if ok else 0
            row.update({"cites": cites, "cite_ok": ok,
                        "coverage_in_cited": cov_in_cited})
            print(f"  ② 引用：资料 {cites} → "
                  f"{'✅ 要点都在引用块里' if ok else f'❌ 要点未全在（覆盖 {cov_in_cited * 100:.0f}%），缺：' + '/'.join(miss_in_cited)}")
        else:
            row["cites"] = []
            print("  ② 引用：⚠️ 回答里没写依据（本指标不计入）")

        # ---- 指标 ① 的 LLM 对照 ----
        if use_llm:
            j = llm_judge(q["question"], q["expected_points"], answer)
            if j:
                llm_cov_sum += float(j.get("covered", 0))
                row["llm_coverage"] = j.get("covered")
                print(f"  ①' LLM 覆盖：{float(j.get('covered', 0)) * 100:.0f}%"
                      f"　（理由：{j.get('reason', '')[:50]}）")
            time.sleep(0.3)      # 轻微限速，避免打太快

        rows.append(row)

    # ---------------- 汇总 ----------------
    n_ans = len(answerable)
    print("\n" + "=" * 74)
    print("【结果】")
    print(f"  ① 答案正确率（关键词判分）　= {kw_cov_sum / n_ans * 100:.1f}%　"
          f"（{n_ans} 题的平均要点覆盖率）")
    if use_llm:
        print(f"  ①' 答案正确率（LLM 判分）　= {llm_cov_sum / n_ans * 100:.1f}%")
        # ⭐ 逐题看两把尺子是否一致 —— 均值会互相抵消，逐题才看得出差异
        diff = [r for r in rows
                if r.get("llm_coverage") is not None
                and abs(float(r["kw_coverage"]) - float(r["llm_coverage"])) >= 0.25]
        if diff:
            print(f"\n  ⚠️ 两个判分不一致的题（差 ≥25 个点）：{len(diff)} 道")
            for r in diff:
                print(f"     [{r['id']}] {r['question'][:26]}")
                print(f"           关键词 {r['kw_coverage'] * 100:.0f}%　"
                      f"LLM {float(r['llm_coverage']) * 100:.0f}%　"
                      f"缺：{'/'.join(r.get('kw_miss', [])) or '（无）'}")
        else:
            print("\n  ✅ 两个判分逐题一致（差异均 <25 个点）——"
                  "这次关键词判分是可靠的")
    if cite_total:
        print(f"  ② 引用准确性　　　　　　　= {cite_ok}/{cite_total} = "
              f"{cite_ok / cite_total * 100:.1f}%（只算写了依据的回答）")
    else:
        print("  ② 引用准确性　　　　　　　= 无样本（回答里都没写依据）")
    n_ref = len(refusable)
    if n_ref:
        print(f"  ③ 拒答率（1 - 幻觉率）　　= {refused_ok}/{n_ref} = "
              f"{refused_ok / n_ref * 100:.1f}%")
    print("=" * 74)

    # 落盘：文件名带时间戳，避免覆盖（项目 3 的教训）
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(DATA_DIR, f"eval_answer_{stamp}.json")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "retriever": factory.RETRIEVER, "top_k": TOP_K, "llm_judge": use_llm,
            "rewrite": use_rewrite, "limit": limit, "n_questions": len(questions),
            "summary": {
                "answerable": n_ans,
                "kw_coverage_mean": kw_cov_sum / n_ans if n_ans else None,
                "llm_coverage_mean": (llm_cov_sum / n_ans) if (use_llm and n_ans) else None,
                "citation_ok": cite_ok, "citation_total": cite_total,
                "refused_ok": refused_ok, "refusable": n_ref,
            },
            "rows": rows,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n原始结果已写入：{out}")


if __name__ == "__main__":
    main()
