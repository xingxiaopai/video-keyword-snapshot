# -*- coding: utf-8 -*-
"""生成测试视频：在指定时间段出现中文字幕，用于验证关键字截图脚本"""
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

OUT = Path(__file__).parent / "sample.mp4"
FPS = 10.0
DUR = 60
W, H = 1280, 720

font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 44)
font_s = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 30)

# (起始秒, 结束秒, 字幕内容)
script = [
    (0, 6,  "欢迎来到直播间"),
    (6, 12, "今天给大家介绍这款新品"),
    (12, 15, "限时优惠 立即抢购"),
    (15, 21, "先看外观设计"),
    (21, 27, "现在下单立减三百元"),
    (27, 30, "下单即送精美礼品"),
    (30, 36, "我们来看参数"),
    (36, 42, "评论区扣1参与抽奖"),
    (42, 45, "抽奖活动马上开始"),
    (45, 52, "感谢大家观看"),
    (52, 60, "下期再见"),
]

writer = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
assert writer.isOpened()

n = int(DUR * FPS)
for i in range(n):
    t = i / FPS
    img = Image.new("RGB", (W, H), (18, 24, 38))
    d = ImageDraw.Draw(img)
    # 背景网格，制造画面变化
    for gx in range(0, W, 80):
        d.line([(gx, 0), (gx, H)], fill=(26, 34, 52), width=1)
    for gy in range(0, H, 80):
        d.line([(0, gy), (W, gy)], fill=(26, 34, 52), width=1)
    # 左上角滚动计时器（保证每帧都有变化，用于测试画面去重）
    d.text((30, 30), f"REC {t:06.1f}s", font=font_s, fill=(90, 110, 140))
    # 底部字幕
    for (a, b, txt) in script:
        if a <= t < b:
            d.text((80, 610), txt, font=font, fill=(255, 255, 255))
            break
    writer.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

writer.release()
print("已生成:", OUT, OUT.stat().st_size, "bytes")
