#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video_keyword_snapshot.py —— 批量视频「关键字画面」自动截图工具

功能：
    扫描一批视频文件，按设定频率抽帧做 OCR 文字识别，
    凡是画面中出现指定关键字的帧，自动截图保存，并输出 CSV / JSON / HTML 清单。

典型用途：
    · 直播回放里找出现「下单」「抽奖」「优惠券」的画面
    · 课程 / 会议录像里定位出现某个术语、人名的画面
    · 素材库排查：哪些视频里出现了竞品名称、品牌词

依赖：opencv-python、numpy、rapidocr-onnxruntime（可选：pytesseract）
     pip install opencv-python numpy rapidocr-onnxruntime

快速开始：
    # 1) 处理一个目录下所有视频，找关键字「优惠券」和「立即下单」
    python video_keyword_snapshot.py -i "D:/videos" -k "优惠券,立即下单" -o "D:/out"

    # 2) 只识别画面底部字幕区域，速度更快、更准
    python video_keyword_snapshot.py -i "D:/videos" -k "抽奖" --roi "0,0.72,1,1"

    # 3) 用配置文件跑
    python video_keyword_snapshot.py -c config.yaml
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import difflib
import html
import json
import multiprocessing
import os
import re
import shutil
import sys
import tempfile
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty
from typing import Any, Iterable, Sequence

# 必须在 import cv2 之前设置，才能压掉 FFmpeg 解码告警（如损坏文件的 moov atom not found）
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import cv2
import numpy as np

try:
    cv2.setLogLevel(0)
except Exception:  # pragma: no cover
    pass

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

VIDEO_EXTS = {
    ".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".webm", ".m4v",
    ".mpg", ".mpeg", ".ts", ".mts", ".m2ts", ".rmvb", ".rm", ".3gp",
    ".vob", ".ogv", ".asf", ".f4v",
}

# 规范化时要去掉的标点（OCR 经常自行插入标点，导致关键字匹配不上）
_PUNCT_RE = re.compile(
    r"[\s·・.,，。、!！?？:：;；'\"“”‘’()（）\[\]【】{}<>《》\-—_~～/\\|*#@&+=^%$`]+"
)

DEFAULT_INTERVAL = 1.0        # 默认每 1 秒抽 1 帧
DEFAULT_SCENE_THRESHOLD = 0.002  # 画面变化像素占比阈值（0.2%），0 表示关闭去重
DEFAULT_MAX_SIDE = 1600       # OCR 前把长边缩到此值以内，兼顾速度与精度


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def fmt_ts(sec: float) -> str:
    """秒 -> HH:MM:SS.mmm"""
    if sec is None or sec < 0:
        sec = 0.0
    h = int(sec // 3600)
    m = int(sec % 3600 // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def fmt_ts_filename(sec: float) -> str:
    """秒 -> 00-01-23_456（用于文件名，避免冒号/点号）"""
    return fmt_ts(sec).replace(":", "-").replace(".", "_")


def fmt_dur(sec: float) -> str:
    """秒 -> 12s / 1m32s / 1h02m（用于进度条上的耗时/剩余时间）"""
    try:
        sec = max(0.0, float(sec or 0))
    except (TypeError, ValueError):
        sec = 0.0
    if sec < 60:
        return f"{sec:.0f}s"
    m, s = divmod(int(sec), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def sanitize_name(name: str, max_len: int = 60) -> str:
    """把任意字符串变成安全的文件名片段"""
    name = re.sub(r'[\\/:*?"<>|\r\n\t]+', "_", name)
    name = name.replace(" ", "_")
    name = re.sub(r"_{2,}", "_", name)
    return (name.strip(" ._")[:max_len].strip(" ._") or "kw")


@contextlib.contextmanager
def silence_native_stderr():
    """
    临时把 C 层 stderr（文件描述符 2）重定向到空设备。

    OpenCV 的 FFmpeg 后端在遇到损坏/不支持的视频时，会绕过 Python 直接往 fd 2
    打印「moov atom not found」之类的告警。cv2.setLogLevel 管不到这些，
    所以这里在打开视频的短暂窗口内屏蔽掉，避免污染输出。
    Python 层的 sys.stderr 不受影响。
    """
    devnull = saved = None
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        saved = os.dup(2)
        os.dup2(devnull, 2)
    except Exception:
        if devnull is not None:
            os.close(devnull)
        if saved is not None:
            os.close(saved)
        yield
        return
    try:
        yield
    finally:
        try:
            os.dup2(saved, 2)
        except Exception:
            pass
        for fd in (saved, devnull):
            try:
                os.close(fd)
            except Exception:
                pass


def safe_imwrite(path: Path, img: np.ndarray, quality: int = 95) -> bool:
    """写图片，兼容中文/空格路径（cv2.imwrite 在部分平台对非 ASCII 路径会失败）"""
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    params: list[int] = []
    if ext in (".jpg", ".jpeg"):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        return False
    try:
        buf.tofile(str(path))
        return True
    except Exception:
        return False


def open_capture(path: Path):
    """
    打开视频。若因中文路径导致打开失败，则把文件复制到 ASCII 临时目录再试。
    返回 (cap, tmpdir)；失败返回 (None, None)。
    """
    with silence_native_stderr():
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return cap, None
            cap.release()

        tmpdir = tempfile.mkdtemp(prefix="vkws_")
        tmppath = Path(tmpdir) / ("source" + path.suffix.lower())
        try:
            shutil.copy2(path, tmppath)
        except Exception:
            shutil.rmtree(tmpdir, ignore_errors=True)
            return None, None

        cap = cv2.VideoCapture(str(tmppath))
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return cap, tmpdir
        cap.release()
    shutil.rmtree(tmpdir, ignore_errors=True)
    return None, None


def collect_videos(inputs: Sequence[str], recursive: bool = True,
                   exts: Iterable[str] = VIDEO_EXTS) -> list[Path]:
    """把文件/目录/通配符统一展开成视频文件列表（去重、排序）"""
    import glob as _glob

    exts = {e.lower() for e in exts}
    found: list[Path] = []
    for raw in inputs:
        for token in str(raw).split(";"):
            token = token.strip().strip('"')
            if not token:
                continue
            p = Path(token)
            if p.is_dir():
                it = p.rglob("*") if recursive else p.glob("*")
                found += [f for f in it if f.is_file() and f.suffix.lower() in exts]
            elif p.is_file():
                found.append(p)
            else:
                # 当作通配符处理
                for m in _glob.glob(token, recursive=recursive):
                    mp = Path(m)
                    if mp.is_file() and mp.suffix.lower() in exts:
                        found.append(mp)
                    elif mp.is_dir():
                        it = mp.rglob("*") if recursive else mp.glob("*")
                        found += [f for f in it if f.is_file() and f.suffix.lower() in exts]

    uniq: dict[str, Path] = {}
    for f in found:
        try:
            key = str(f.resolve()).lower()
        except Exception:
            key = str(f).lower()
        uniq.setdefault(key, f)
    return sorted(uniq.values(), key=lambda x: str(x).lower())


# ---------------------------------------------------------------------------
# 关键字匹配
# ---------------------------------------------------------------------------

@dataclass
class KeywordMatcher:
    """关键字匹配器：支持多关键字、忽略大小写、正则、模糊匹配、文本规范化"""
    keywords: list[str]
    ignore_case: bool = True
    regex: bool = False
    normalize: bool = True
    fuzzy: float = 0.0          # >0 时启用模糊匹配（0~1，越大越严格），如 0.85

    def __post_init__(self) -> None:
        self.keywords = [k for k in (k.strip() for k in self.keywords) if k]
        if not self.keywords:
            raise ValueError("至少需要提供一个关键字")
        self._norm_kw = [(k, self.norm(k)) for k in self.keywords]
        self._patterns = []
        if self.regex:
            flags = re.IGNORECASE if self.ignore_case else 0
            for k in self.keywords:
                try:
                    self._patterns.append((k, re.compile(k, flags)))
                except re.error as e:
                    raise ValueError(f"正则表达式非法：{k} -> {e}") from e

    def norm(self, text: str) -> str:
        """规范化：全角转半角、去空白、去标点、（可选）转小写"""
        t = unicodedata.normalize("NFKC", text or "")
        if self.normalize:
            t = _PUNCT_RE.sub("", t)
        return t.lower() if self.ignore_case else t

    def match(self, text: str) -> list[tuple[str, float]]:
        """
        对一段文本做匹配，返回 [(命中的关键字, 相似度)]。
        精确命中相似度记 1.0；模糊命中为实际相似度。
        """
        if not text:
            return []
        hits: list[tuple[str, float]] = []

        if self.regex:
            for raw, pat in self._patterns:
                if pat.search(text):
                    hits.append((raw, 1.0))
            return hits

        norm_text = self.norm(text)
        if not norm_text:
            return []
        for raw, nkw in self._norm_kw:
            if not nkw:
                continue
            if nkw in norm_text:
                hits.append((raw, 1.0))
            elif self.fuzzy > 0:
                # 用滑动窗口比对，支持「关键字是长句的一部分但有个别字被 OCR 认错」
                L = len(nkw)
                if L and len(norm_text) >= L:
                    best = 0.0
                    for i in range(0, len(norm_text) - L + 1):
                        ratio = difflib.SequenceMatcher(None, nkw, norm_text[i:i + L]).ratio()
                        if ratio > best:
                            best = ratio
                            if best >= self.fuzzy:
                                break
                    if best >= self.fuzzy:
                        hits.append((raw, round(best, 3)))
        return hits


# ---------------------------------------------------------------------------
# OCR 引擎
# ---------------------------------------------------------------------------

@dataclass
class OcrLine:
    text: str
    score: float
    box: list  # [[x,y] * 4]


class OcrEngine:
    """OCR 引擎封装。默认 rapidocr（内置中英文模型，离线可用）"""

    def __init__(self, engine: str = "rapidocr", min_score: float = 0.5,
                 max_side: int = DEFAULT_MAX_SIDE, text_score: float | None = None):
        self.engine_name = engine.lower()
        self.min_score = float(min_score)
        self.max_side = int(max_side or 0)
        self._impl = None
        self._build()

    def _build(self) -> None:
        if self.engine_name in ("rapidocr", "rapid", "rapidocr_onnxruntime"):
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError as e:  # pragma: no cover
                raise RuntimeError(
                    "未安装 rapidocr-onnxruntime，请执行：pip install rapidocr-onnxruntime"
                ) from e
            self._impl = RapidOCR()
        elif self.engine_name in ("tesseract", "pytesseract"):
            try:
                import pytesseract  # noqa: F401
            except ImportError as e:  # pragma: no cover
                raise RuntimeError(
                    "未安装 pytesseract，请执行：pip install pytesseract 并安装 Tesseract-OCR"
                ) from e
            self._impl = "tesseract"
        else:
            raise ValueError(f"不支持的 OCR 引擎：{self.engine_name}")

    def read(self, bgr: np.ndarray) -> list[OcrLine]:
        """识别一帧图像（BGR），返回置信度达标的文本行"""
        if bgr is None or bgr.size == 0:
            return []
        h, w = bgr.shape[:2]
        scale = 1.0
        if self.max_side and max(h, w) > self.max_side:
            scale = self.max_side / float(max(h, w))
            bgr = cv2.resize(bgr, (max(1, int(round(w * scale))),
                                   max(1, int(round(h * scale)))),
                             interpolation=cv2.INTER_AREA)

        lines: list[OcrLine] = []
        if self._impl == "tesseract":
            import pytesseract
            from pytesseract import Output
            data = pytesseract.image_to_data(bgr, output_type=Output.DICT, lang="chi_sim+eng")
            n = len(data.get("text", []))
            for i in range(n):
                txt = (data["text"][i] or "").strip()
                try:
                    conf = float(data["conf"][i]) / 100.0
                except Exception:
                    conf = 0.0
                if not txt or conf < self.min_score:
                    continue
                x, y = data["left"][i], data["top"][i]
                ww, hh = data["width"][i], data["height"][i]
                box = [[x, y], [x + ww, y], [x + ww, y + hh], [x, y + hh]]
                lines.append(OcrLine(txt, conf, box))
        else:
            result, _elapse = self._impl(bgr)
            for item in (result or []):
                try:
                    box, txt, score = item[0], item[1], float(item[2])
                except Exception:
                    continue
                txt = (txt or "").strip()
                if not txt or score < self.min_score:
                    continue
                if scale != 1.0:
                    box = [[float(p[0]) / scale, float(p[1]) / scale] for p in box]
                lines.append(OcrLine(txt, score, box))
        return lines


# 进程内 OCR 引擎缓存（每个进程只加载一次模型）
_ENGINE_CACHE: dict[tuple, OcrEngine] = {}


def get_engine(cfg: dict) -> OcrEngine:
    key = (cfg.get("ocr_engine", "rapidocr"), cfg.get("min_score", 0.5),
           cfg.get("max_side", DEFAULT_MAX_SIDE))
    eng = _ENGINE_CACHE.get(key)
    if eng is None:
        eng = OcrEngine(engine=key[0], min_score=key[1], max_side=key[2])
        _ENGINE_CACHE[key] = eng
    return eng


# ---------------------------------------------------------------------------
# 数据处理结构
# ---------------------------------------------------------------------------

@dataclass
class Hit:
    """一次命中"""
    t: float                 # 秒
    frame_index: int
    keyword: str
    matched_text: str        # 命中的那一行原文
    score: float             # OCR 置信度
    match_score: float       # 匹配相似度
    full_text: str           # 该帧全部识别文本
    lines: list = field(default_factory=list)  # 所有识别行（用于画框）

    @property
    def weight(self) -> tuple:
        """用于在同一段落中挑「最佳帧」"""
        return (len(self.keyword), self.match_score, self.score, len(self.matched_text))


# ---------------------------------------------------------------------------
# 单视频处理
# ---------------------------------------------------------------------------

SIG_W, SIG_H = 320, 180


def _frame_signature(bgr: np.ndarray, rois: Sequence[tuple] | None) -> np.ndarray:
    """
    取灰度小图，用于判断相邻采样帧画面是否变化。

    若指定了 ROI，则只在 ROI 区域内计算签名（分辨率相对更高，对小字幕变化更敏感）；
    否则使用整帧缩略图。返回一维灰度数组。
    """
    h, w = bgr.shape[:2]
    parts: list[np.ndarray] = []
    if rois:
        for (x1, y1, x2, y2) in rois:
            xa, ya = int(max(0.0, x1) * w), int(max(0.0, y1) * h)
            xb, yb = int(min(1.0, x2) * w), int(min(1.0, y2) * h)
            if xb <= xa or yb <= ya:
                continue
            crop = bgr[ya:yb, xa:xb]
            ch, cw = crop.shape[:2]
            # 保持长宽比缩放到 320 宽，至少 24 高，避免细字幕被压没
            tw = min(SIG_W, cw)
            th = max(24, int(round(ch * tw / max(1, cw))))
            parts.append(cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA))
    if not parts:
        parts = [cv2.resize(bgr, (SIG_W, SIG_H), interpolation=cv2.INTER_AREA)]

    grays = [cv2.cvtColor(p, cv2.COLOR_BGR2GRAY) for p in parts]
    return np.concatenate([g.reshape(-1) for g in grays])


def _changed_ratio(prev: np.ndarray, cur: np.ndarray, pixel_delta: int = 25) -> float:
    """变化像素占比（prev/cur 为一维灰度数组）"""
    if prev is None or cur is None or prev.shape != cur.shape:
        return 1.0
    diff = np.abs(prev.astype(np.int16) - cur.astype(np.int16))
    return float(np.count_nonzero(diff > pixel_delta)) / float(diff.size)


def process_video(vpath: Path, cfg: dict) -> dict:
    """
    处理单个视频，返回结果字典：
        {video, path, ok, error, records: [Hit-like dict], stats: {...}}
    """
    out: dict[str, Any] = {
        "video": vpath.name, "path": str(vpath), "ok": False, "error": None,
        "records": [], "stats": {},
    }
    t_start = time.time()

    # 进度上报（单进程模式是 InlineProgress，多进程模式是队列代理；两者都只要求有 put_nowait）
    progress_sink = cfg.get("_progress")

    def emit(**kw: Any) -> None:
        if progress_sink is None:
            return
        kw["video"] = vpath.name
        kw["path"] = str(vpath)
        try:
            progress_sink.put_nowait(kw)
        except Exception:
            pass

    cap, tmpdir = open_capture(vpath)
    if cap is None:
        out["error"] = "无法打开视频（编码不支持或文件损坏）"
        out["elapsed_sec"] = round(time.time() - t_start, 2)
        emit(phase="done", failed=True, saved=0, span=1, done=1,
             elapsed=out["elapsed_sec"])
        return out

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not (fps > 0) or fps > 1000:
            fps = 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = (total_frames / fps) if total_frames > 0 else 0.0

        interval = float(cfg.get("interval", DEFAULT_INTERVAL))
        step = max(1, int(round(fps * interval)))

        start_sec = float(cfg.get("start", 0.0) or 0.0)
        end_sec = float(cfg.get("end", 0.0) or 0.0)
        if end_sec <= 0 or (duration and end_sec > duration):
            end_sec = duration if duration else 0.0

        first_frame = max(0, int(start_sec * fps))
        last_frame = int(end_sec * fps) if end_sec > 0 else (total_frames or 0)
        if last_frame <= first_frame:
            last_frame = first_frame + step
        last_frame = min(last_frame, total_frames) if total_frames else last_frame

        roi = cfg.get("roi")  # (x1,y1,x2,y2) 相对坐标，或 None
        rois = cfg.get("rois") or ([roi] if roi else [])
        scene_th = float(cfg.get("scene_threshold", DEFAULT_SCENE_THRESHOLD))
        save_mode = cfg.get("save_mode", "best")
        merge_gap = float(cfg.get("merge_gap", 2.0))
        min_interval = float(cfg.get("min_interval", 0.0))
        join_lines = bool(cfg.get("join_lines", True))
        matcher: KeywordMatcher = cfg["matcher"]
        engine = get_engine(cfg)

        span = max(1, last_frame - first_frame)   # 本次要扫过的总帧数（进度分母）
        emit(phase="start", span=span, done=0, cur_t=start_sec,
             duration=duration, total_frames=total_frames)

        hits: list[Hit] = []
        frames_read = 0
        frames_ocr = 0
        frames_skipped = 0
        prev_sig: np.ndarray | None = None
        last_ocr_t = -1e9
        last_emit = 0.0

        idx = first_frame
        while idx < last_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames_read += 1
            t = idx / fps

            # --- 画面变化检测：静止画面直接跳过 OCR ---
            if scene_th > 0:
                sig = _frame_signature(frame, rois)
                if prev_sig is not None and _changed_ratio(prev_sig, sig) < scene_th:
                    frames_skipped += 1
                    idx += step
                    continue
                prev_sig = sig

            # --- 抽帧节流 ---
            if min_interval > 0 and (t - last_ocr_t) < min_interval:
                idx += step
                continue

            # --- OCR（支持多个 ROI）---
            all_lines: list[OcrLine] = []
            if rois:
                h, w = frame.shape[:2]
                for (x1, y1, x2, y2) in rois:
                    xa, ya = int(max(0.0, x1) * w), int(max(0.0, y1) * h)
                    xb, yb = int(min(1.0, x2) * w), int(min(1.0, y2) * h)
                    if xb <= xa or yb <= ya:
                        continue
                    crop = frame[ya:yb, xa:xb]
                    for ln in engine.read(crop):
                        ln.box = [[float(p[0]) + xa, float(p[1]) + ya] for p in ln.box]
                        all_lines.append(ln)
            else:
                all_lines = engine.read(frame)

            frames_ocr += 1
            last_ocr_t = t

            if all_lines:
                # 逐行匹配
                matched: list[tuple[str, float, str]] = []  # (kw, sim, line_text)
                for ln in all_lines:
                    for kw, sim in matcher.match(ln.text):
                        matched.append((kw, sim, ln.text))
                # 整帧合并匹配（应对关键字被 OCR 切成多个框的情况）
                if join_lines and not matched:
                    joined = "".join(l.text for l in all_lines)
                    for kw, sim in matcher.match(joined):
                        matched.append((kw, sim, joined))

                if matched:
                    # 同一帧多关键字只记一条，取权重最高的
                    kw, sim, text = max(matched, key=lambda x: (len(x[0]), x[1]))
                    best_line_score = max(
                        (l.score for l in all_lines if l.text == text), default=0.0
                    )
                    hits.append(Hit(
                        t=t, frame_index=idx, keyword=kw, matched_text=text,
                        score=round(best_line_score, 4), match_score=sim,
                        full_text=" | ".join(l.text for l in all_lines),
                        lines=[{"text": l.text, "score": round(l.score, 4),
                                "box": l.box} for l in all_lines],
                    ))

            # 进度上报（节流：最多每 0.25 秒一次，避免刷屏）
            now = time.time()
            if now - last_emit >= 0.25:
                last_emit = now
                emit(phase="run", done=idx - first_frame, span=span, cur_t=t,
                     duration=duration, frames_ocr=frames_ocr,
                     frames_skipped=frames_skipped, hits=len(hits))

            idx += step

        # --- 合并相邻命中为「片段」---
        segments: list[list[Hit]] = []
        for h in hits:
            if segments and (h.t - segments[-1][-1].t) <= merge_gap:
                segments[-1].append(h)
            else:
                segments.append([h])

        # --- 选出要保存的帧 ---
        to_save: list[Hit] = []
        for seg in segments:
            if save_mode == "all":
                to_save.extend(seg)
            elif save_mode == "first":
                to_save.append(seg[0])
            else:  # best
                to_save.append(max(seg, key=lambda x: x.weight))

        if min_interval > 0 and len(to_save) > 1:
            filtered = [to_save[0]]
            for h in to_save[1:]:
                if (h.t - filtered[-1].t) >= min_interval:
                    filtered.append(h)
            to_save = filtered

        out["stats"] = {
            "fps": round(fps, 3), "duration_sec": round(duration, 2),
            "width": width, "height": height,
            "sample_step_frames": step, "sample_interval_sec": interval,
            "frames_read": frames_read, "frames_ocr": frames_ocr,
            "frames_skipped_static": frames_skipped,
            "hit_frames": len(hits), "segments": len(segments),
            "saved": len(to_save),
        }
        out["ok"] = True

        # --- 保存截图 ---
        if to_save:
            outdir: Path = cfg["_video_outdir"]
            for n, h in enumerate(to_save, 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, h.frame_index)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                save_img = frame.copy()
                if cfg.get("draw_boxes", False):
                    for ln in h.lines:
                        pts = np.array(ln["box"], dtype=np.int32).reshape(-1, 1, 2)
                        cv2.polylines(save_img, [pts], True, (0, 200, 0), 2)
                if cfg.get("burn_text", True):
                    save_img = _burn_caption(save_img, h)
                ext = ".png" if str(cfg.get("image_format", "png")).lower() == "png" else ".jpg"
                fname = (f"{n:04d}_{fmt_ts_filename(h.t)}_"
                         f"{sanitize_name(h.keyword, 24)}__{sanitize_name(vpath.stem, 40)}{ext}")
                fpath = outdir / fname
                if safe_imwrite(fpath, save_img, int(cfg.get("jpeg_quality", 95))):
                    out["records"].append({
                        "video": vpath.name,
                        "video_path": str(vpath),
                        "time_sec": round(h.t, 3),
                        "time_hms": fmt_ts(h.t),
                        "frame_index": h.frame_index,
                        "keyword": h.keyword,
                        "matched_text": h.matched_text,
                        "match_score": h.match_score,
                        "ocr_score": h.score,
                        "ocr_text": h.full_text,
                        "image": str(fpath),
                        "image_name": fpath.name,
                    })
    finally:
        cap.release()
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)

    out["elapsed_sec"] = round(time.time() - t_start, 2)
    emit(phase="done", failed=not out["ok"], saved=len(out["records"]),
         elapsed=out["elapsed_sec"], done=1, span=1)
    return out


def _cv_text_box(img: np.ndarray, text: str, org: tuple[int, int],
                 scale: float, color: tuple[int, int, int],
                 thickness: int = 2, bg_alpha: float = 0.55) -> None:
    """OpenCV 兜底方案：画带半透明底色的文字（不支持中文）"""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    x = max(0, min(x, img.shape[1] - tw - 8))
    y = max(th + base + 6, min(y, img.shape[0] - 6))
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 6, y - th - base - 6), (x + tw + 6, y + base + 4),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, bg_alpha, img, 1 - bg_alpha, 0, img)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness,
                cv2.LINE_AA)


# 常见中文字体候选（按优先级）
_FONT_CANDIDATES = (
    "C:/Windows/Fonts/msyh.ttc",          # 微软雅黑
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",        # 黑体
    "C:/Windows/Fonts/simsun.ttc",        # 宋体
    "C:/Windows/Fonts/Deng.ttf",          # 等线
    "/System/Library/Fonts/PingFang.ttc",  # macOS
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)
_FONT_STATE: dict[str, Any] = {"checked": False, "path": None, "cache": {}}


def _load_font(size: int):
    """按需加载字体（带缓存）。找不到可用字体时返回 None。"""
    if not _FONT_STATE["checked"]:
        _FONT_STATE["checked"] = True
        env = os.environ.get("VKWS_FONT")
        if env and Path(env).is_file():
            _FONT_STATE["path"] = env
        else:
            for p in _FONT_CANDIDATES:
                if Path(p).is_file():
                    _FONT_STATE["path"] = p
                    break
    path = _FONT_STATE["path"]
    if not path:
        return None
    cached = _FONT_STATE["cache"].get(size)
    if cached is not None:
        return cached
    try:
        from PIL import ImageFont
        font = ImageFont.truetype(path, size)
    except Exception:
        return None
    _FONT_STATE["cache"][size] = font
    return font


def burn_caption(img: np.ndarray,
                 items: Sequence[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
    """
    在画面左上角叠加半透明信息条（时间戳 / 命中关键字）。
    优先用 PIL 渲染（支持中文），找不到字体时退回 OpenCV 内置字体（仅 ASCII）。
    颜色参数为 BGR。
    """
    if not items or img is None or img.size == 0:
        return img
    h, w = img.shape[:2]
    size = max(16, int(round(w / 48.0)))
    font = _load_font(size)

    if font is None:
        for i, (text, color) in enumerate(items):
            ascii_text = text.encode("ascii", "replace").decode("ascii")
            _cv_text_box(img, ascii_text,
                         (14, size + 14 + i * int(size * 1.5)),
                         size / 30.0, color)
        return img

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return img

    pad, line_h = 9, size + 7
    text_w = 0
    for text, _c in items:
        try:
            text_w = max(text_w, int(font.getlength(text)))
        except Exception:
            text_w = max(text_w, len(text) * size)
    total_h = line_h * len(items) + pad * 2 - 7
    x0, y0 = 12, 12

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rounded_rectangle(
        [x0, y0, x0 + text_w + pad * 2, y0 + total_h], radius=6, fill=(0, 0, 0, 165)
    )
    pil = Image.alpha_composite(pil, overlay).convert("RGB")
    draw = ImageDraw.Draw(pil)
    for i, (text, color) in enumerate(items):
        b, g, r = color
        draw.text((x0 + pad, y0 + pad + i * (line_h - 7)), text, font=font, fill=(r, g, b))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _burn_caption(img: np.ndarray, h: "Hit") -> np.ndarray:
    """烧录时间戳 + 命中关键字"""
    items = [(fmt_ts(h.t), (60, 255, 120))]
    if h.keyword:
        items.append((f"命中：{h.keyword}", (0, 220, 255)))
    return burn_caption(img, items)


# ---------------------------------------------------------------------------
# 输出：CSV / JSON / HTML
# ---------------------------------------------------------------------------

def write_csv(path: Path, records: list[dict]) -> None:
    cols = ["video", "time_hms", "time_sec", "keyword", "matched_text",
            "match_score", "ocr_score", "image_name", "ocr_text", "video_path", "image"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)


def write_json(path: Path, records: list[dict], results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_hits": len(records),
        "total_videos": len(results),
        "hits": records,
        "videos": [
            {"video": r["video"], "path": r["path"], "ok": r["ok"],
             "error": r["error"], "stats": r["stats"], "hits": len(r["records"])}
            for r in results
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


_HTML_TPL = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>视频关键字截图报告</title>
<style>
  :root{{--bg:#f6f7f9;--card:#fff;--bd:#e3e6ea;--tx:#1f2328;--sub:#656d76;--acc:#0b6bcb;}}
  *{{box-sizing:border-box}}
  body{{margin:0;background:var(--bg);color:var(--tx);
    font:14px/1.6 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;}}
  header{{position:sticky;top:0;background:rgba(255,255,255,.94);backdrop-filter:blur(8px);
    border-bottom:1px solid var(--bd);padding:14px 22px;z-index:9;}}
  h1{{margin:0 0 4px;font-size:17px}}
  .meta{{color:var(--sub);font-size:12.5px}}
  .bar{{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px;align-items:center}}
  input,select{{padding:7px 11px;border:1px solid var(--bd);border-radius:8px;
    background:#fff;font-size:13px;color:var(--tx);min-width:170px}}
  input:focus,select:focus{{outline:2px solid rgba(11,107,203,.25);border-color:var(--acc)}}
  .count{{color:var(--sub);font-size:12.5px;margin-left:auto}}
  main{{padding:18px 22px 60px;display:grid;gap:16px;
    grid-template-columns:repeat(auto-fill,minmax(320px,1fr));}}
  .card{{background:var(--card);border:1px solid var(--bd);border-radius:12px;overflow:hidden;
    display:flex;flex-direction:column;transition:.15s;}}
  .card:hover{{box-shadow:0 6px 20px rgba(20,30,50,.09);transform:translateY(-2px)}}
  .thumb{{background:#eceff3;display:block;line-height:0}}
  .thumb img{{width:100%;height:auto;display:block;cursor:zoom-in}}
  .body{{padding:11px 13px 13px;display:flex;flex-direction:column;gap:6px;flex:1}}
  .row{{display:flex;gap:8px;align-items:baseline;flex-wrap:wrap}}
  .kw{{background:#e8f1fb;color:var(--acc);border-radius:6px;padding:1px 8px;
    font-size:12px;font-weight:600}}
  .ts{{font-variant-numeric:tabular-nums;font-weight:600;color:var(--tx)}}
  .vid{{color:var(--sub);font-size:12px;word-break:break-all}}
  .ocr{{color:var(--sub);font-size:12.5px;border-top:1px dashed var(--bd);padding-top:7px;
    margin-top:auto;max-height:70px;overflow:auto}}
  .empty{{grid-column:1/-1;text-align:center;color:var(--sub);padding:60px 0}}
  #lightbox{{position:fixed;inset:0;background:rgba(15,20,28,.9);display:none;
    align-items:center;justify-content:center;z-index:99;padding:24px}}
  #lightbox img{{max-width:100%;max-height:100%;border-radius:8px}}
</style>
</head>
<body>
<header>
  <h1>视频关键字截图报告</h1>
  <div class="meta">生成时间 {generated} · 关键字：{kw_list} · 共 {n_hits} 张截图 / {n_videos} 个视频</div>
  <div class="bar">
    <input id="q" placeholder="搜索关键字 / 文件名 / 识别文本">
    <select id="kwsel"><option value="">全部关键字</option>{kw_options}</select>
    <select id="vidsel"><option value="">全部视频</option>{vid_options}</select>
    <span class="count" id="count"></span>
  </div>
</header>
<main id="grid">{cards}</main>
<div id="lightbox"><img id="lbimg" alt=""></div>
<script>
const q=document.getElementById('q'),ks=document.getElementById('kwsel'),
      vs=document.getElementById('vidsel'),cards=[...document.querySelectorAll('.card')],
      count=document.getElementById('count');
function apply(){{
  const s=q.value.trim().toLowerCase(),k=ks.value,v=vs.value;let n=0;
  cards.forEach(c=>{{
    const hay=(c.dataset.kw+' '+c.dataset.video+' '+c.dataset.text).toLowerCase();
    const ok=(!s||hay.includes(s))&&(!k||c.dataset.kw===k)&&(!v||c.dataset.video===v);
    c.style.display=ok?'':'none';if(ok)n++;
  }});
  count.textContent='显示 '+n+' / '+cards.length+' 张';
}}
q.addEventListener('input',apply);ks.addEventListener('change',apply);vs.addEventListener('change',apply);
apply();
document.getElementById('grid').addEventListener('click',e=>{{
  if(e.target.tagName==='IMG'){{document.getElementById('lbimg').src=e.target.src;
    document.getElementById('lightbox').style.display='flex';}}
}});
document.getElementById('lightbox').addEventListener('click',()=>{{
  document.getElementById('lightbox').style.display='none';}});
document.addEventListener('keydown',e=>{{if(e.key==='Escape')
  document.getElementById('lightbox').style.display='none';}});
</script>
</body>
</html>
"""


def write_html(path: Path, records: list[dict], keywords: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = path.parent
    cards: list[str] = []
    for r in records:
        try:
            rel = os.path.relpath(r["image"], base).replace("\\", "/")
        except Exception:
            rel = r["image"].replace("\\", "/")
        cards.append(
            '<div class="card" data-kw="{kw}" data-video="{vid}" data-text="{txt}">'
            '<a class="thumb" href="{rel}" target="_blank"><img loading="lazy" src="{rel}" alt=""></a>'
            '<div class="body">'
            '<div class="row"><span class="kw">{kw}</span><span class="ts">{ts}</span></div>'
            '<div class="vid">{vid}</div>'
            '<div class="ocr">{ocr}</div>'
            '</div></div>'.format(
                kw=html.escape(r["keyword"]),
                vid=html.escape(r["video"]),
                txt=html.escape((r.get("ocr_text") or "")[:400]),
                rel=html.escape(rel),
                ts=html.escape(r["time_hms"]),
                ocr=html.escape((r.get("ocr_text") or "")[:300]),
            )
        )
    if not cards:
        cards.append('<div class="empty">没有命中任何关键字画面</div>')

    kws = sorted({r["keyword"] for r in records}) or list(keywords)
    vids = sorted({r["video"] for r in records})
    html_out = _HTML_TPL.format(
        generated=time.strftime("%Y-%m-%d %H:%M:%S"),
        kw_list=html.escape("、".join(keywords)),
        n_hits=len(records),
        n_videos=len({r["video"] for r in records}),
        kw_options="".join(f'<option value="{html.escape(k)}">{html.escape(k)}</option>' for k in kws),
        vid_options="".join(f'<option value="{html.escape(v)}">{html.escape(v)}</option>' for v in vids),
        cards="\n".join(cards),
    )
    path.write_text(html_out, encoding="utf-8")


# ---------------------------------------------------------------------------
# 配置解析
# ---------------------------------------------------------------------------

def parse_rois(text: str | None) -> list[tuple[float, float, float, float]]:
    """解析 ROI 字符串，如 "0,0.72,1,1" 或 "0,0,1,0.2; 0,0.75,1,1"（相对坐标 0~1）"""
    if not text:
        return []
    rois = []
    for chunk in re.split(r"[;|]", str(text)):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p for p in re.split(r"[,:\s]+", chunk) if p]
        if len(parts) != 4:
            raise ValueError(f"ROI 格式错误：{chunk}（应形如 x1,y1,x2,y2）")
        x1, y1, x2, y2 = (float(p) for p in parts)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"ROI 区域无效：{chunk}")
        rois.append((x1, y1, x2, y2))
    return rois


def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("读取 YAML 需要 PyYAML：pip install pyyaml") from e
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("配置文件顶层必须是键值对")
    return data


def build_parser() -> argparse.ArgumentParser:
    # argument_default=SUPPRESS：未出现在命令行的参数不会进入 namespace，
    # 这样配置文件里的值才能真正生效（否则会被 argparse 的默认值覆盖）。
    p = argparse.ArgumentParser(
        prog="video_keyword_snapshot",
        description="批量视频关键字画面自动截图工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        argument_default=argparse.SUPPRESS,
        epilog=(
            "示例：\n"
            "  python video_keyword_snapshot.py -i D:/videos -k 优惠券,下单 -o D:/out\n"
            "  python video_keyword_snapshot.py -i D:/videos -k 抽奖 --roi 0,0.72,1,1\n"
            "  python video_keyword_snapshot.py -c config.yaml\n"
        ),
    )
    p.add_argument("-c", "--config", help="YAML 配置文件（命令行参数优先级高于配置文件）")
    p.add_argument("-i", "--input", nargs="+",
                   help="输入视频文件 / 目录 / 通配符，可多个，也可用分号分隔")
    p.add_argument("-k", "--keywords",
                   help="关键字，多个用英文逗号分隔，如：优惠券,立即下单")
    p.add_argument("-kf", "--keyword-file",
                   help="从文本文件读取关键字（每行一个，# 开头为注释）")
    p.add_argument("-o", "--output", help="输出目录（默认 ./output）")
    p.add_argument("--interval", type=float,
                   help=f"抽帧间隔秒数，默认 {DEFAULT_INTERVAL}（越小越细但越慢，0.5 = 每秒 2 帧）")
    p.add_argument("--start", type=float, help="从第几秒开始处理，默认 0")
    p.add_argument("--end", type=float, help="处理到第几秒结束，默认 0（到结尾）")
    p.add_argument("--roi", help="只识别指定区域，相对坐标 x1,y1,x2,y2（0~1）；多个用分号分隔")
    p.add_argument("--scene-threshold", type=float,
                   help=f"画面变化阈值（变化像素占比），低于此值跳过 OCR，默认 {DEFAULT_SCENE_THRESHOLD}；0 = 关闭")
    p.add_argument("--save-mode", choices=["best", "first", "all"],
                   help="保存策略：best=每段最佳帧（默认）、first=每段首帧、all=全部命中帧")
    p.add_argument("--merge-gap", type=float,
                   help="间隔小于此秒数的命中合并为同一段，默认 2.0")
    p.add_argument("--min-interval", type=float,
                   help="两张截图之间的最小时间间隔（秒），用于防止刷屏，默认 0")
    p.add_argument("--max-side", type=int,
                   help=f"OCR 前把画面长边缩到此值以内，默认 {DEFAULT_MAX_SIDE}；0 = 不缩放")
    p.add_argument("--min-score", type=float, help="OCR 文本置信度下限，默认 0.5")
    p.add_argument("--engine", choices=["rapidocr", "tesseract"],
                   help="OCR 引擎，默认 rapidocr（离线中英文）")
    p.add_argument("--ignore-case", dest="ignore_case", action="store_true",
                   help="忽略大小写（默认开启）")
    p.add_argument("--case-sensitive", dest="ignore_case", action="store_false",
                   help="区分大小写")
    p.add_argument("--regex", action="store_true", help="把关键字当作正则表达式")
    p.add_argument("--no-normalize", dest="normalize", action="store_false",
                   help="不做文本规范化（默认会去掉空白和标点，能显著提高中文命中率）")
    p.add_argument("--fuzzy", type=float,
                   help="模糊匹配阈值 0~1（如 0.85），用于容忍 OCR 错字，默认关闭")
    p.add_argument("--no-join-lines", dest="join_lines", action="store_false",
                   help="关闭整帧合并匹配（默认开启，可匹配被 OCR 拆成多个框的关键字）")
    p.add_argument("--draw-boxes", action="store_true", help="在截图上画出 OCR 文本框")
    p.add_argument("--no-burn-text", dest="burn_text", action="store_false",
                   help="不在截图上烧录时间戳")
    p.add_argument("--image-format", choices=["png", "jpg"], help="截图格式，默认 png")
    p.add_argument("--jpeg-quality", type=int, help="JPG 质量，默认 95")
    p.add_argument("--flat", action="store_true",
                   help="所有截图放在同一目录（默认按视频分文件夹）")
    p.add_argument("--workers", type=int,
                   help="并行处理视频的进程数，默认自动（CPU 核数的一半，上限 4）")
    p.add_argument("--resume", action="store_true",
                   help="断点续跑：跳过输出目录中已记录完成的视频")
    p.add_argument("--no-recursive", dest="recursive", action="store_false",
                   help="不递归子目录")
    p.add_argument("--dry-run", action="store_true", help="只列出将要处理的视频，不做识别")
    p.add_argument("--no-report", action="store_true", help="不生成 HTML 报告")
    p.add_argument("--no-progress", action="store_true",
                   help="不显示动态进度条，改为每 30 秒输出一行进度日志（适合重定向到文件）")
    p.add_argument("--quiet", action="store_true", help="安静模式")
    return p


def load_keywords(args) -> list[str]:
    kws: list[str] = []
    raw_kw = getattr(args, "keywords", None)
    if raw_kw:
        kws += [k.strip() for k in re.split(r"[,，\n]", raw_kw) if k.strip()]
    kf_path = getattr(args, "keyword_file", None)
    if kf_path:
        kf = Path(kf_path)
        if not kf.is_file():
            raise FileNotFoundError(f"关键字文件不存在：{kf}")
        for line in kf.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                kws += [k.strip() for k in re.split(r"[,，]", line) if k.strip()]
    # 去重保序
    seen, out = set(), []
    for k in kws:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


_YAML_ALIASES = {
    "输入": "input", "input": "input", "inputs": "input", "videos": "input", "视频": "input",
    "关键字": "keywords", "keywords": "keywords", "keyword": "keywords", "关键词": "keywords",
    "输出": "output", "output": "output", "outdir": "output", "out": "output", "输出目录": "output",
    "间隔": "interval", "interval": "interval", "抽帧间隔": "interval",
    "区域": "roi", "roi": "roi", "识别区域": "roi",
    "引擎": "engine", "engine": "engine", "ocr引擎": "engine",
    "开始": "start", "start": "start", "结束": "end", "end": "end",
    "并行": "workers", "workers": "workers",
    "截图格式": "image_format", "保存策略": "save_mode", "save_mode": "save_mode",
    "画面变化阈值": "scene_threshold", "scene_threshold": "scene_threshold",
    "合并间隔": "merge_gap", "merge_gap": "merge_gap",
    "最短间隔": "min_interval", "min_interval": "min_interval",
    "模糊匹配": "fuzzy", "fuzzy": "fuzzy",
    "置信度": "min_score", "min_score": "min_score",
    "最长边": "max_side", "max_side": "max_side",
    "画框": "draw_boxes", "draw_boxes": "draw_boxes",
    "烧录时间戳": "burn_text", "burn_text": "burn_text",
    "扁平输出": "flat", "flat": "flat",
    "断点续跑": "resume", "resume": "resume",
    "递归": "recursive", "recursive": "recursive",
    "正则": "regex", "regex": "regex",
    "忽略大小写": "ignore_case", "ignore_case": "ignore_case",
    "规范化": "normalize", "normalize": "normalize",
    "合并整帧文本": "join_lines", "join_lines": "join_lines",
    "jpg质量": "jpeg_quality", "jpeg_quality": "jpeg_quality",
}


def normalize_rois(raw) -> list[tuple[float, float, float, float]]:
    """ROI 支持三种写法：字符串 "0,0.7,1,1"；[x1,y1,x2,y2]；[[x1,y1,x2,y2], ...]"""
    if raw is None or raw == "" or raw == []:
        return []
    if isinstance(raw, str):
        return parse_rois(raw)
    if isinstance(raw, (list, tuple)):
        if len(raw) == 4 and all(isinstance(x, (int, float)) for x in raw):
            return [(float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))]
        out: list[tuple[float, float, float, float]] = []
        for item in raw:
            out += normalize_rois(item)
        return out
    raise ValueError(f"无法解析 roi 配置：{raw!r}")


def merge_config(args) -> dict:
    """配置文件 + 命令行参数合并（命令行 > 配置文件 > 程序默认值）"""
    raw_cfg: dict[str, Any] = {}
    config_path = getattr(args, "config", None)
    if config_path:
        yml = load_yaml(Path(config_path))
        for k, v in yml.items():
            raw_cfg[_YAML_ALIASES.get(str(k).strip(), str(k).strip())] = v

    def pick(name, default=None):
        v = getattr(args, name, None)   # 未在命令行出现的参数不存在（SUPPRESS）
        if v is not None:
            return v
        v = raw_cfg.get(name)
        if v is not None:
            return v
        return default

    def pick_bool(name, default: bool) -> bool:
        v = getattr(args, name, None)
        if v is not None:
            return bool(v)
        v = raw_cfg.get(name)
        if v is not None:
            return bool(v)
        return default

    cfg: dict[str, Any] = {}

    # --- 输入 ---
    inputs = pick("input", [])
    if isinstance(inputs, (str, Path)):
        inputs = [str(inputs)]
    elif isinstance(inputs, (list, tuple)):
        inputs = [str(x) for x in inputs]
    else:
        inputs = []
    cfg["inputs"] = [x for x in inputs if x]

    # --- 关键字 ---
    kws = load_keywords(args)
    if not kws:
        yk = raw_cfg.get("keywords") or []
        if isinstance(yk, str):
            yk = [x.strip() for x in re.split(r"[,，\n]", yk) if x.strip()]
        elif isinstance(yk, (list, tuple)):
            yk = [str(x).strip() for x in yk]
        else:
            yk = []
        kws = [x for x in yk if x]
    if not kws:
        raise SystemExit(
            "错误：必须指定关键字（-k/--keywords、-kf/--keyword-file 或配置文件 keywords）"
        )
    cfg["keywords"] = kws

    # --- 其余参数 ---
    cfg["output"] = str(pick("output", "./output"))
    cfg["interval"] = float(pick("interval", DEFAULT_INTERVAL))
    cfg["start"] = float(pick("start", 0.0) or 0.0)
    cfg["end"] = float(pick("end", 0.0) or 0.0)
    rois = normalize_rois(pick("roi", None))
    cfg["rois"] = rois
    cfg["roi"] = rois[0] if rois else None
    cfg["scene_threshold"] = float(pick("scene_threshold", DEFAULT_SCENE_THRESHOLD))
    cfg["save_mode"] = str(pick("save_mode", "best")).lower()
    cfg["merge_gap"] = float(pick("merge_gap", 2.0))
    cfg["min_interval"] = float(pick("min_interval", 0.0))
    cfg["max_side"] = int(pick("max_side", DEFAULT_MAX_SIDE))
    cfg["min_score"] = float(pick("min_score", 0.5))
    cfg["ocr_engine"] = str(pick("engine", "rapidocr"))
    cfg["regex"] = pick_bool("regex", False)
    cfg["normalize"] = pick_bool("normalize", True)
    cfg["fuzzy"] = float(pick("fuzzy", 0.0))
    cfg["join_lines"] = pick_bool("join_lines", True)
    cfg["draw_boxes"] = pick_bool("draw_boxes", False)
    cfg["burn_text"] = pick_bool("burn_text", True)
    cfg["image_format"] = str(pick("image_format", "png")).lower()
    cfg["jpeg_quality"] = int(pick("jpeg_quality", 95))
    cfg["flat"] = pick_bool("flat", False)
    cfg["resume"] = pick_bool("resume", False)
    cfg["recursive"] = pick_bool("recursive", True)
    cfg["ignore_case"] = pick_bool("ignore_case", True)
    cfg["workers"] = int(pick("workers", 0) or 0)

    # --- 参数校验 ---
    if cfg["interval"] <= 0:
        raise ValueError("interval 必须大于 0")
    if cfg["save_mode"] not in ("best", "first", "all"):
        raise ValueError(f"save_mode 只能是 best / first / all，当前为 {cfg['save_mode']!r}")
    if cfg["image_format"] not in ("png", "jpg"):
        raise ValueError(f"image_format 只能是 png / jpg，当前为 {cfg['image_format']!r}")
    if not (0.0 <= cfg["fuzzy"] <= 1.0):
        raise ValueError("fuzzy 必须在 0~1 之间")
    if cfg["min_interval"] < 0 or cfg["merge_gap"] < 0 or cfg["scene_threshold"] < 0:
        raise ValueError("min_interval / merge_gap / scene_threshold 不能为负数")

    cfg["matcher"] = KeywordMatcher(
        keywords=cfg["keywords"],
        ignore_case=cfg["ignore_case"],
        regex=cfg["regex"],
        normalize=cfg["normalize"],
        fuzzy=cfg["fuzzy"],
    )
    return cfg


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def _worker(task: tuple[str, dict]) -> dict:
    """
    多进程工作函数（必须是模块级函数，否则 spawn 模式下无法序列化）。
    每个子进程内自行重建 matcher 与 OCR 引擎，只接收纯数据配置。
    """
    vpath_str, sub = task
    sub = dict(sub)
    sub["matcher"] = KeywordMatcher(**sub.pop("_matcher_args"))
    sub["_video_outdir"] = Path(sub["_video_outdir"])
    return process_video(Path(vpath_str), sub)


class Dashboard:
    """
    实时进度显示。

    TTY 环境下原地刷新一个多行面板：
        视频 1/3 个已完成 · 累计命中 7 张 · 已用 01:23
        [████████████░░░░░░░░░░░░]  48.2%  00:01:26/00:03:00  OCR 62 帧  命中 3 张  剩余 ~1m36s  a.mp4
        某个视频完成后，它那一行消失，上面的计数递增。

    非 TTY（输出被重定向 / 管道）时退化为按行输出，避免刷屏和乱码：
        每行一条「开始 / 心跳 / 完成」记录，心跳最多每 30 秒一次。
    """

    BAR_W = 24
    HEARTBEAT_SEC = 30.0

    def __init__(self, total_videos: int, enabled: bool = True,
                 tty: bool | None = None):
        self.total = max(1, int(total_videos))
        self.enabled = bool(enabled)
        self.tty = bool(sys.stdout.isatty()) if tty is None else bool(tty)
        if not self.enabled:
            self.tty = False
        self.states: dict[str, dict] = {}
        self.settled: set[str] = set()
        self.done = 0
        self.done_hits = 0
        self.done_failed = 0
        self.t0 = time.time()
        self.lines_drawn = 0
        self._last_beat: dict[str, float] = {}
        self._width = self._term_width()

    # ---------- 外部接口 ----------

    def handle(self, msg: dict) -> None:
        """处理一条来自 worker 的进度消息"""
        if not self.enabled or not isinstance(msg, dict):
            return
        key = str(msg.get("path") or msg.get("video") or "?")
        phase = msg.get("phase")

        if phase == "start":
            self.states[key] = dict(msg)
            self.states[key]["_started"] = time.time()
            self._last_beat[key] = time.time()   # 心跳从 30 秒后开始，不在启动瞬间刷一条
            if not self.tty:
                print(f"  [开始] {msg.get('video', '?')}  "
                      f"时长 {fmt_dur(msg.get('duration') or 0)}  "
                      f"采样 {msg.get('span', 0)} 帧", flush=True)
        elif phase == "run":
            st = self.states.get(key)
            if st is None:
                st = self.states[key] = {"_started": time.time()}
            st.update(msg)
        elif phase == "done":
            self.settle(key, msg.get("video") or "?",
                        failed=bool(msg.get("failed")),
                        saved=int(msg.get("saved") or 0),
                        elapsed=float(msg.get("elapsed") or 0.0))

    def settle(self, key: str, video: str, failed: bool = False,
               saved: int = 0, elapsed: float = 0.0) -> None:
        """
        把一个视频标记为结束（幂等：队列消息与主循环都会调用，只会算一次）。
        非 TTY 下不在这里打印，避免和最后的汇总重复输出。
        """
        key = str(key)
        if key in self.settled:
            return
        self.settled.add(key)
        self.states.pop(key, None)
        self.done += 1
        if failed:
            self.done_failed += 1
        else:
            self.done_hits += saved

    def drain(self, q) -> int:
        """把队列里的进度消息全部取出并处理，返回处理条数（非阻塞）"""
        if not self.enabled or q is None:
            return 0
        n = 0
        while True:
            try:
                msg = q.get_nowait()
            except (Empty, EOFError, OSError, ValueError):
                break
            except Exception:
                break
            self.handle(msg)
            n += 1
            if n > 5000:      # 保险丝，避免异常情况下死循环
                break
        return n

    def render(self) -> None:
        if not self.enabled:
            return
        lines = [self._header()]
        if self.states:
            for st in list(self.states.values()):
                lines.append(self._video_line(st))
        elif self.done >= self.total:
            lines.append("全部处理完毕")
        if self.tty:
            self._draw(lines)
        else:
            self._beat(lines)

    def finish(self) -> None:
        """收尾：TTY 下清掉面板，让后续汇总信息从干净的位置开始"""
        if self.enabled and self.tty and self.lines_drawn:
            sys.stdout.write(f"\x1b[{self.lines_drawn}A")
            for _ in range(self.lines_drawn):
                sys.stdout.write("\x1b[2K\n")
            sys.stdout.write(f"\x1b[{self.lines_drawn}A")
            sys.stdout.flush()
        self.lines_drawn = 0

    # ---------- 内部 ----------

    @staticmethod
    def _term_width() -> int:
        try:
            return max(60, shutil.get_terminal_size((100, 24)).columns)
        except Exception:
            return 100

    def _header(self) -> str:
        parts = [f"视频 {self.done}/{self.total} 个已完成",
                 f"累计命中 {self.done_hits} 张"]
        if self.done_failed:
            parts.append(f"失败 {self.done_failed} 个")
        parts.append(f"已用 {fmt_dur(time.time() - self.t0)}")
        return " · ".join(parts)

    def _video_line(self, st: dict) -> str:
        span = max(1, int(st.get("span") or 1))
        done = max(0, int(st.get("done") or 0))
        frac = min(1.0, done / span)
        filled = int(round(self.BAR_W * frac))
        bar = "█" * filled + "░" * (self.BAR_W - filled)
        el = max(0.0, time.time() - float(st.get("_started") or time.time()))
        # 进度太低时估算出的剩余时间没有意义，显示 -- 而不是一个假数字
        eta = f"~{fmt_dur(el / frac - el)}" if frac > 0.02 else "--"
        name = str(st.get("video") or "?")
        if len(name) > 20:
            name = name[:18] + ".."
        return (f"[{bar}] {frac * 100:5.1f}%  "
                f"{fmt_dur(st.get('cur_t') or 0)}/{fmt_dur(st.get('duration') or 0)}  "
                f"OCR {int(st.get('frames_ocr') or 0)} 帧  "
                f"命中 {int(st.get('hits') or 0)} 张  "
                f"剩余 {eta}  {name}")

    def _draw(self, lines: list[str]) -> None:
        width = self._width - 1
        lines = [ln[:width] for ln in lines]
        buf: list[str] = []
        if self.lines_drawn:
            buf.append(f"\x1b[{self.lines_drawn}A")   # 回到面板左上角
        for ln in lines:
            buf.append("\x1b[2K" + ln + "\n")          # 清行 + 写行
        extra = self.lines_drawn - len(lines)
        if extra > 0:                                  # 清掉多余的旧行
            for _ in range(extra):
                buf.append("\x1b[2K\n")
            buf.append(f"\x1b[{extra}A")
        self.lines_drawn = len(lines)
        sys.stdout.write("".join(buf))
        sys.stdout.flush()

    def _beat(self, lines: list[str]) -> None:
        """非 TTY：按行输出，每个视频最多每 30 秒一条心跳"""
        now = time.time()
        for key, st in list(self.states.items()):
            if now - self._last_beat.get(key, 0.0) >= self.HEARTBEAT_SEC:
                self._last_beat[key] = now
                print("  " + self._video_line(st), flush=True)


class InlineProgress:
    """
    单进程模式下的进度接收器：worker 与主流程在同一个进程，
    直接处理并立刻重绘，不需要队列。
    """

    def __init__(self, dashboard: Dashboard):
        self.dashboard = dashboard

    def put_nowait(self, msg: dict) -> None:
        self.dashboard.handle(msg)
        self.dashboard.render()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = merge_config(args)
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"配置错误：{e}", file=sys.stderr)
        return 2

    inputs = cfg["inputs"]
    if not inputs:
        print("错误：必须指定输入（-i/--input 或配置文件中的 input）", file=sys.stderr)
        return 2

    videos = collect_videos(inputs, recursive=cfg["recursive"])
    if not videos:
        print("没有找到任何视频文件。支持的扩展名："
              + ", ".join(sorted(VIDEO_EXTS)), file=sys.stderr)
        return 1

    quiet = bool(getattr(args, "quiet", False))
    dry_run = bool(getattr(args, "dry_run", False))
    no_report = bool(getattr(args, "no_report", False))
    no_progress = bool(getattr(args, "no_progress", False))
    if not quiet:
        print(f"关键字：{'、'.join(cfg['keywords'])}")
        print(f"共找到 {len(videos)} 个视频，抽帧间隔 {cfg['interval']}s，"
              f"OCR 引擎 {cfg['ocr_engine']}"
              + (f"，ROI {cfg['rois']}" if cfg["rois"] else "")
              + f"，保存策略 {cfg['save_mode']}"
              + (f"，画面变化阈值 {cfg['scene_threshold']}"
                 if cfg["scene_threshold"] > 0 else "，已关闭画面去重"))

    if dry_run:
        for v in videos:
            print("  -", v)
        return 0

    outdir = Path(cfg["output"])
    outdir.mkdir(parents=True, exist_ok=True)

    # 断点续跑
    done_videos: set[str] = set()
    state_path = outdir / "_state.json"
    if cfg["resume"] and state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            done_videos = {v for v, ok in state.get("done", {}).items() if ok}
        except Exception:
            done_videos = set()

    todo = []
    for v in videos:
        if cfg["resume"] and str(v) in done_videos:
            continue
        todo.append(v)
    if cfg["resume"] and len(todo) < len(videos) and not quiet:
        print(f"断点续跑：跳过已完成 {len(videos) - len(todo)} 个视频")

    if not todo:
        print("没有需要处理的视频。")
        return 0

    # 每个视频的输出目录
    work: list[tuple[Path, dict]] = []
    used_names: dict[str, int] = {}
    for v in todo:
        if cfg["flat"]:
            vdir = outdir
        else:
            base = sanitize_name(v.stem, 50)
            used_names[base] = used_names.get(base, 0) + 1
            vdir = outdir / (base if used_names[base] == 1 else f"{base}_{used_names[base]}")
        vdir.mkdir(parents=True, exist_ok=True)
        sub = dict(cfg)
        sub["_video_outdir"] = vdir
        sub.pop("matcher", None)  # 不跨进程传 matcher，由子进程自行构建
        sub["_matcher_args"] = {
            "keywords": cfg["keywords"], "ignore_case": cfg["ignore_case"],
            "regex": cfg["regex"], "normalize": cfg["normalize"], "fuzzy": cfg["fuzzy"],
        }
        work.append((v, sub))

    workers = cfg["workers"] or max(1, min(4, (os.cpu_count() or 2) // 2))
    workers = max(1, min(workers, len(work)))
    if not quiet:
        print(f"并行进程数：{workers}\n")

    t0 = time.time()
    results: list[dict] = []

    # ---- 进度显示 ----
    # 单进程：worker 与主流程同进程，用 InlineProgress 直接重绘，零额外开销。
    # 多进程：用一个 Manager 队列收集各 worker 的进度，主循环负责排空 + 重绘
    #        （multiprocessing.Queue 不能作为任务参数跨进程序列化，Manager 代理可以）。
    dashboard = Dashboard(len(work), enabled=not quiet,
                          tty=(False if no_progress else None))
    mgr = None
    if not dashboard.enabled:
        progress_sink = None
    elif workers == 1:
        progress_sink = InlineProgress(dashboard)
    else:
        mgr = multiprocessing.Manager()
        progress_sink = mgr.Queue()

    for _v, sub in work:
        sub["_progress"] = progress_sink

    # 转成纯数据任务，便于跨进程传递
    tasks = [(str(v), sub) for v, sub in work]

    def _failed_result(v: Path, e: BaseException) -> dict:
        return {"video": v.name, "path": str(v), "ok": False,
                "error": f"{type(e).__name__}: {e}", "records": [], "stats": {}}

    try:
        if workers == 1:
            dashboard.render()
            for v, task in zip(todo, tasks):
                try:
                    r = _worker(task)
                except Exception as e:
                    r = _failed_result(v, e)
                results.append(r)
                dashboard.settle(r["path"], r["video"], failed=not r["ok"],
                                 saved=len(r["records"]),
                                 elapsed=float(r.get("elapsed_sec") or 0.0))
                dashboard.render()
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_worker, task): Path(task[0]) for task in tasks}
                pending = set(futs)
                while pending:
                    finished, pending = wait(pending, timeout=0.15)
                    dashboard.drain(progress_sink)
                    for fut in finished:
                        v = futs[fut]
                        try:
                            r = fut.result()
                        except Exception as e:  # 单个视频失败不影响整体
                            r = _failed_result(v, e)
                        results.append(r)
                        dashboard.settle(r["path"], r["video"], failed=not r["ok"],
                                         saved=len(r["records"]),
                                         elapsed=float(r.get("elapsed_sec") or 0.0))
                    dashboard.render()
                # 收尾：把剩余进度消息排空，确保计数不漏
                dashboard.drain(progress_sink)
                for v in todo:
                    dashboard.settle(str(v), v.name, failed=False, saved=0, elapsed=0.0)
                dashboard.render()
    finally:
        dashboard.finish()
        if mgr is not None:
            try:
                mgr.shutdown()
            except Exception:
                pass

    # 清理因处理失败而没有产出任何截图的空目录
    if not cfg["flat"]:
        for _v, sub in work:
            d: Path = sub["_video_outdir"]
            try:
                if d.is_dir() and not any(d.iterdir()):
                    d.rmdir()
            except Exception:
                pass

    results.sort(key=lambda r: str(r["path"]).lower())
    records: list[dict] = []
    for r in results:
        records.extend(r["records"])

    # 汇总输出
    write_csv(outdir / "hits.csv", records)
    write_json(outdir / "hits.json", records, results)
    if not no_report:
        write_html(outdir / "report.html", records, cfg["keywords"])

    if cfg["resume"]:
        state = {"done": {r["path"]: bool(r["ok"]) for r in results}, "updated": time.time()}
        if state_path.is_file():
            try:
                old = json.loads(state_path.read_text(encoding="utf-8"))
                old_done = old.get("done", {})
                old_done.update(state["done"])
                state["done"] = old_done
            except Exception:
                pass
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    failed = [r for r in results if not r["ok"]]
    # 失败信息始终输出（安静模式下也不能吞掉），便于排查
    for r in failed:
        print(f"[失败] {r['video']}：{r.get('error') or '未知错误'}", file=sys.stderr)
    if not quiet:
        print("\n" + "=" * 62)
        print(f"完成：{len(results)} 个视频，命中 {len(records)} 张截图，"
              f"耗时 {time.time() - t0:.1f}s")
        total_ocr = sum(r["stats"].get("frames_ocr", 0) for r in results)
        total_skip = sum(r["stats"].get("frames_skipped_static", 0) for r in results)
        print(f"  共 OCR {total_ocr} 帧，画面无变化跳过 {total_skip} 帧")
        for r in results:
            st = r.get("stats", {})
            flag = "OK " if r["ok"] else "ERR"
            extra = (f"{st.get('saved', 0)} 张 / 命中 {st.get('hit_frames', 0)} 帧"
                     if r["ok"] else (r.get("error") or ""))
            print(f"  [{flag}] {r['video']}  {extra}")
        if failed:
            print(f"\n  {len(failed)} 个视频处理失败（其余结果不受影响）")
        print(f"\n输出目录：{outdir.resolve()}")
        print(f"  截图清单：{outdir / 'hits.csv'}")
        if not no_report:
            print(f"  可视化报告：{outdir / 'report.html'}")
        print("=" * 62)
    return 0 if not failed else 3


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        sys.exit(130)
