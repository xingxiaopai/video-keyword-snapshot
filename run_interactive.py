#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_interactive.py —— 交互式启动器（不想记命令行参数时用这个）

直接运行，按提示把视频路径和关键字粘进去就行：

    python run_interactive.py

它最后会打印出等价的命令行，方便你以后直接复制使用。
"""

from __future__ import annotations

import os
import shlex
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROI_PRESETS = {
    "1": ("整帧识别（最全，最慢）", None),
    "2": ("底部字幕区（推荐，快很多）", "0,0.70,1,1"),
    "3": ("底部窄条（字幕在很下面时用）", "0,0.80,1,1"),
    "4": ("顶部标题区", "0,0,1,0.20"),
    "5": ("自定义区域", "custom"),
}


def clean_path(raw: str) -> str:
    """去掉用户粘贴时可能带上的引号和多余空白"""
    p = (raw or "").strip()
    while len(p) >= 2 and p[0] == p[-1] and p[0] in "\"'“”":
        p = p[1:-1].strip()
    return p


def ask(prompt: str, default: str | None = None, allow_empty: bool = False) -> str:
    hint = f"（直接回车 = {default}）" if default else ""
    while True:
        try:
            val = input(f"{prompt}{hint}\n> ").strip()
        except EOFError:
            print("\n输入结束，已退出。")
            sys.exit(0)
        if not val:
            if default is not None:
                return default
            if allow_empty:
                return ""
            print("  这一项不能为空，请重新输入。\n")
            continue
        return val


def check_deps() -> bool:
    """启动前检查依赖，缺了就给一条能直接照做的提示，而不是抛一堆 traceback"""
    missing = []
    for mod, pkg in (("cv2", "opencv-python"), ("numpy", "numpy"),
                     ("rapidocr_onnxruntime", "rapidocr-onnxruntime")):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if not missing:
        return True

    print("缺少依赖：" + "、".join(missing))
    print()
    print("当前解释器：")
    print(f"    {sys.executable}")
    print()
    print("安装依赖：")
    print(f'    "{sys.executable}" -m pip install -r requirements.txt')
    print()
    print("如果依赖其实装在另一个解释器里，任选一种方式告诉脚本：")
    print("    · 在项目目录新建 .python-path 文件，写入那个解释器的完整路径（一行）")
    print("    · 或者设置环境变量 VKWS_PYTHON 指向它")
    print("    · 或者在项目目录建 .venv 并装好依赖")
    print("之后重新双击 run.cmd 即可。")
    return False


def main() -> int:
    if not check_deps():
        return 2

    print("=" * 66)
    print("  批量视频关键字截图 · 交互式启动")
    print("=" * 66)
    print("提示：路径可以直接把文件夹从资源管理器拖进这个窗口，会自动填入。")
    print()

    # ---- 1. 视频路径 ----
    raw = ask("第 1 步 / 4：视频路径（可以是单个文件、一个文件夹、或 *.mp4）")
    video_input = clean_path(raw)
    if not video_input:
        print("路径为空，已退出。")
        return 2

    # ---- 2. 关键字 ----
    raw_kw = ask("第 2 步 / 4：要找的关键字（多个用逗号分隔，例如：优惠券,立即下单）")
    keywords = ",".join(k.strip() for k in raw_kw.replace("，", ",").split(",") if k.strip())
    if not keywords:
        print("关键字为空，已退出。")
        return 2

    # ---- 3. 识别范围 ----
    print()
    print("第 3 步 / 4：识别画面的哪个区域？")
    for k, (desc, _) in ROI_PRESETS.items():
        print(f"  {k}. {desc}")
    choice = ask("请选择", default="1")
    roi_desc, roi = ROI_PRESETS.get(choice.strip(), ROI_PRESETS["1"])
    if roi == "custom":
        print("  自定义格式：x1,y1,x2,y2，用 0~1 的相对比例表示。")
        print("  例：0,0.72,1,1 = 画面下方 28% 那条横带")
        roi = clean_path(ask("  请输入区域")) or None
        roi_desc = f"自定义 {roi}"

    # ---- 4. 输出目录与间隔 ----
    print()
    default_out = str(Path(video_input).parent / f"关键字截图_{time.strftime('%Y%m%d_%H%M')}") \
        if Path(video_input).is_file() else \
        str(Path(video_input) / f"关键字截图_{time.strftime('%Y%m%d_%H%M')}")
    out_dir = clean_path(ask("第 4 步 / 4：截图保存到哪个目录？", default=default_out))
    interval = ask("抽帧间隔秒数（越小越细但越慢，关键字一闪而过就填 0.3）", default="1.0")
    try:
        float(interval)
    except ValueError:
        interval = "1.0"

    # ---- 组装命令 ----
    argv = ["-i", video_input, "-k", keywords, "-o", out_dir, "--interval", interval]
    if roi:
        argv += ["--roi", roi]

    print()
    print("-" * 66)
    print("即将执行：")
    print(f"  视频路径   : {video_input}")
    print(f"  关键字     : {keywords}")
    print(f"  识别区域   : {roi_desc}")
    print(f"  抽帧间隔   : {interval} 秒")
    print(f"  输出目录   : {out_dir}")
    print()
    print("等价的命令行（以后可以直接复制使用）：")
    py = "python"
    print("  " + py + " video_keyword_snapshot.py " + " ".join(shlex.quote(a) for a in argv))
    print("-" * 66)
    try:
        input("\n按回车开始处理（想取消就按 Ctrl+C）...")
    except (EOFError, KeyboardInterrupt):
        print("\n已取消。")
        return 130

    print()
    import video_keyword_snapshot as vks

    code = vks.main(argv)

    report = Path(out_dir) / "report.html"
    if code == 0 and report.is_file():
        print()
        try:
            ans = input(f"是否现在打开截图报告？(Y/n) > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = "n"
        if ans in ("", "y", "yes"):
            try:
                if hasattr(os, "startfile"):
                    os.startfile(str(report))          # Windows
                else:
                    import webbrowser
                    webbrowser.open(report.resolve().as_uri())
            except Exception as e:
                print(f"打不开自动预览（{e}），请手动打开：{report.resolve()}")
    return code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        sys.exit(130)
