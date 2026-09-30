# -*- coding: utf-8 -*-
"""验证 Dashboard 在 TTY 模式下的原地刷新是否正确：
把输出喂给一个极简 ANSI 解释器，还原出最终屏幕内容。"""
import io
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import video_keyword_snapshot as vks


class MiniTerm:
    """只支持 \\n \\r \\x1b[2K \\x1b[{n}A 的极简终端"""

    def __init__(self):
        self.lines = [""]
        self.row = 0
        self.col = 0

    def feed(self, s: str) -> None:
        i = 0
        while i < len(s):
            if s.startswith("\x1b[2K", i):
                self.lines[self.row] = ""
                self.col = 0
                i += 4
            elif s.startswith("\x1b[", i):
                m = re.match(r"\x1b\[(\d*)A", s[i:])
                if m:
                    n = int(m.group(1) or 1)
                    self.row = max(0, self.row - n)
                    i += m.end()
                else:
                    m2 = re.match(r"\x1b\[[0-9;]*[A-Za-z]", s[i:])
                    i += m2.end() if m2 else 1
            elif s[i] == "\n":
                self.row += 1
                self.col = 0
                while len(self.lines) <= self.row:
                    self.lines.append("")
                i += 1
            elif s[i] == "\r":
                self.col = 0
                i += 1
            else:
                line = self.lines[self.row]
                line = line[:self.col] + s[i]
                self.lines[self.row] = line
                self.col += 1
                i += 1

    def screen(self) -> str:
        return "\n".join(l.rstrip() for l in self.lines if l.strip())


def run():
    buf = io.StringIO()
    real_stdout = sys.stdout
    sys.stdout = buf
    try:
        dash = vks.Dashboard(3, enabled=True, tty=True)
        # 第 1 个视频
        dash.handle({"phase": "start", "video": "直播回放_A.mp4", "path": "A",
                     "span": 600, "duration": 60.0})
        dash.render()
        dash.handle({"phase": "run", "video": "直播回放_A.mp4", "path": "A",
                     "done": 300, "span": 600, "cur_t": 30.0, "duration": 60.0,
                     "frames_ocr": 42, "hits": 2})
        dash.render()
        # 第 2 个视频加入
        dash.handle({"phase": "start", "video": "直播回放_B.mp4", "path": "B",
                     "span": 900, "duration": 90.0})
        dash.render()
        dash.handle({"phase": "run", "video": "直播回放_B.mp4", "path": "B",
                     "done": 90, "span": 900, "cur_t": 9.0, "duration": 90.0,
                     "frames_ocr": 12, "hits": 0})
        dash.render()
        # A 完成（面板少一行，测试多余的旧行是否被清掉）
        dash.handle({"phase": "done", "video": "直播回放_A.mp4", "path": "A",
                     "saved": 4, "elapsed": 12.3})
        dash.render()
        dash.handle({"phase": "run", "video": "直播回放_B.mp4", "path": "B",
                     "done": 800, "span": 900, "cur_t": 80.0, "duration": 90.0,
                     "frames_ocr": 150, "hits": 3})
        dash.render()
        before_finish = buf.getvalue()
        dash.finish()
    finally:
        sys.stdout = real_stdout

    term = MiniTerm()
    term.feed(buf.getvalue())
    print("=== finish() 之后的屏幕（应为空，让汇总信息从干净位置开始）===")
    print(repr(term.screen()))
    print()
    term2 = MiniTerm()
    term2.feed(before_finish)
    print("=== finish() 之前的面板实际样子 ===")
    print(term2.screen())
    print()
    print("=== 断言 ===")
    raw = buf.getvalue()
    checks = [
        ("使用了清行序列 \\x1b[2K", "\x1b[2K" in raw),
        ("使用了光标上移 \\x1b[NA", re.search(r"\x1b\[\d+A", raw) is not None),
        ("没有 \\r 残留刷新", "\r" not in raw),
        ("进度百分比正确 (A=50.0%)", "50.0%" in raw),
        ("进度百分比正确 (B=88.9%)", "88.9%" in raw),
        ("命中数显示正确", "命中 3 张" in raw),
        ("A 完成后计数为 1", "视频 1/3 个已完成" in raw),
        ("累计命中为 4", "累计命中 4 张" in raw),
        ("finish 后面板已清空", term.screen().strip() == ""),
    ]
    ok = True
    for name, passed in checks:
        print(("  [通过] " if passed else "  [失败] ") + name)
        ok = ok and passed
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
