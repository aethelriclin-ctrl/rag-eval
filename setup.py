"""把两个项目的文档复制进 docs/，作为 RAG 的知识库。

为什么知识库用自己的项目文档：
    1. 我知道标准答案 —— 评测的前提是"我知道对的答案是什么"
    2. 面试可以现场演示 —— "问它我项目为什么有失败案例，它去查我的失败案例库"
    3. 天然包含"库里没有的答案" —— 用来测幻觉（第 2 层的核心）

用法：python setup.py
"""
import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")

# 要复制的源文件：项目 1 在 agent-eval 根目录，项目 2 在 agent-eval/agent-tool-eval
SRC_ROOT = r"D:\vscode\agent-eval"
SOURCES = [
    # (源路径, 复制到 docs 后的文件名)
    (os.path.join(SRC_ROOT, "README.md"), "项目1-README.md"),
    (os.path.join(SRC_ROOT, "失败案例.md"), "项目1-失败案例.md"),
    (os.path.join(SRC_ROOT, "模型选型报告.md"), "项目1-模型选型报告.md"),
    (os.path.join(SRC_ROOT, "cases_hard.json"), "项目1-难题库.json"),
    (os.path.join(SRC_ROOT, "agent-tool-eval", "README.md"), "项目2-README.md"),
    (os.path.join(SRC_ROOT, "agent-tool-eval", "失败案例.md"), "项目2-失败案例.md"),
    (os.path.join(SRC_ROOT, "agent-tool-eval", "幻觉率实验.md"), "项目2-幻觉实验.md"),
]


def main():
    os.makedirs(DOCS_DIR, exist_ok=True)
    copied, missing = 0, []

    for src, dst_name in SOURCES:
        if not os.path.exists(src):
            missing.append(src)
            continue
        dst = os.path.join(DOCS_DIR, dst_name)
        shutil.copy2(src, dst)
        size = os.path.getsize(dst)
        print(f"  ✅ {dst_name}  ({size} 字节)")
        copied += 1

    print(f"\n共复制 {copied} 个文件到 {DOCS_DIR}")
    if missing:
        print("\n⚠️ 以下文件没找到（可能名字不同，手动确认一下）：")
        for m in missing:
            print(f"  ❌ {m}")

    # 提示下一步
    print("\n下一步：把 docs 里每个 md 的标题和长度看一眼，"
          "确认知识库有足够内容可检索。")
    print("  验收标准：docs/ 里至少 5 个 .md 文件，总字数几千字以上。")


if __name__ == "__main__":
    main()
