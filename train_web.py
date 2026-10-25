#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BeatFCN v22 训练网页控制台
给帮忙跑训练的同学用的，不需要懂命令行，浏览器里点按钮就行。

用法：
    python train_web.py
    然后浏览器打开 http://localhost:8000

功能：
    - 检查环境（Python/GPU/依赖/文件）
    - 开始训练 / 断点续训
    - 跑reval选冠军
    - 实时查看日志输出
    - 查看训练进度
"""

import os
import sys
import json
import subprocess
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# ========== 全局状态 ==========
current_process = None
current_log = []
log_lock = threading.Lock()
current_action = None  # "train" / "resume" / "reval" / None

# ========== 工具函数 ==========
def get_device():
    """自动检测设备"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
    except:
        pass
    return "cpu"

def get_batch_size(device):
    """根据设备推荐batch_size"""
    if device == "cuda":
        try:
            import torch
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if vram >= 20: return 4
            elif vram >= 12: return 2
            else: return 1
        except: return 4
    elif device == "mps": return 1
    else: return 1

def build_train_cmd(resume=False):
    """构建训练命令"""
    device = get_device()
    bs = get_batch_size(device)
    cmd = [
        sys.executable, 'beat_fcn_train_v22.py',
        '--epochs', '200', '--patience', '99999', '--tag', 'v22',
        '--device', device, '--batch-size', str(bs), '--threads', '8',
        '--data', './grid_npz_full128', '--split', './split_full128_final.json',
        '--pitch-dir', './pitch_npz_full', '--pitch-mode', 'chroma',
        '--max-frames', '30000', '--eval-songs', '200',
    ]
    if resume: cmd.append('--resume')
    return cmd

def run_command(cmd, action_name):
    """在后台线程运行命令，实时收集日志"""
    global current_process, current_log, current_action
    current_action = action_name
    with log_lock:
        current_log = [f"=== 开始: {action_name} ===", f"命令: {' '.join(cmd)}", ""]

    try:
        current_process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, universal_newlines=True
        )
        for line in current_process.stdout:
            with log_lock:
                current_log.append(line.rstrip())
                if len(current_log) > 2000:  # 限制日志长度
                    current_log = current_log[-1000:]
        current_process.wait()
        with log_lock:
            current_log.append(f"\n=== 完成: {action_name} (退出码: {current_process.returncode}) ===")
    except Exception as e:
        with log_lock:
            current_log.append(f"\n=== 错误: {e} ===")
    finally:
        current_process = None
        current_action = None

def check_environment():
    """检查环境，返回结果字典"""
    results = {"python": {}, "torch": {}, "files": [], "deps": []}

    # Python
    v = sys.version_info
    results["python"] = {
        "version": f"{v.major}.{v.minor}.{v.micro}",
        "ok": v.major == 3 and v.minor in (9, 10)
    }

    # PyTorch/GPU
    try:
        import torch
        results["torch"]["version"] = torch.__version__
        if torch.cuda.is_available():
            results["torch"]["device"] = "cuda"
            results["torch"]["gpu"] = torch.cuda.get_device_name(0)
            results["torch"]["vram"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
            results["torch"]["ok"] = True
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            results["torch"]["device"] = "mps"
            results["torch"]["gpu"] = "Apple Silicon"
            results["torch"]["ok"] = True
        else:
            results["torch"]["device"] = "cpu"
            results["torch"]["ok"] = False
    except ImportError:
        results["torch"]["ok"] = False
        results["torch"]["error"] = "PyTorch未安装"

    # 文件
    required = ['beat_fcn_train_v22.py', 'chart_utils.py', 'infer_v22.py',
                'reval_v22.py', 'split_full128_final.json']
    for f in required:
        results["files"].append({"name": f, "ok": os.path.exists(f)})
    for d in ['grid_npz_full128', 'pitch_npz_full']:
        if os.path.isdir(d):
            count = len([f for f in os.listdir(d) if f.endswith('.npz')])
            results["files"].append({"name": f"{d}/ ({count}个npz)", "ok": True})
        else:
            results["files"].append({"name": f"{d}/", "ok": False})

    # 依赖（项目自写mel提取，不需要librosa/soundfile）
    for dep in ['numpy', 'scipy', 'torch']:
        try:
            __import__(dep)
            results["deps"].append({"name": dep, "ok": True})
        except ImportError:
            results["deps"].append({"name": dep, "ok": False})

    return results

def get_progress():
    """从日志提取训练进度"""
    with log_lock:
        log = current_log.copy()
    epoch_lines = [l for l in log if '/200]' in l]
    progress = {"current_epoch": 0, "total_epochs": 200, "recent": []}
    if epoch_lines:
        last = epoch_lines[-1]
        # 提取 [x/200]
        import re
        m = re.search(r'\[(\d+)/200\]', last)
        if m:
            progress["current_epoch"] = int(m.group(1))
        progress["recent"] = [l.strip() for l in epoch_lines[-5:]]
    return progress

# ========== HTTP 处理 ==========
HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BEATFCN v22 // NEURAL INTERFACE</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&family=Noto+Sans+SC:wght@400;500;700&display=swap');
@font-face {
  font-family: 'Anthropic Serif';
  src: url('fonts/AnthropicSerif@500.woff2') format('woff2');
  font-weight: 500;
  font-style: normal;
}
@font-face {
  font-family: 'Anthropic Serif';
  src: url('fonts/AnthropicSerif@700.woff2') format('woff2');
  font-weight: 700;
  font-style: normal;
}
@font-face {
  font-family: 'Anthropic Serif';
  src: url('fonts/AnthropicSerif@900.woff2') format('woff2');
  font-weight: 900;
  font-style: normal;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
:root {
  --bg: #050508;
  --cyan: #00f0ff;
  --magenta: #ff00aa;
  --purple: #8b5cf6;
  --green: #00ff88;
  --yellow: #ffcc00;
  --red: #ff3366;
  --text: #d0d0e0;
  --text2: #707090;
  --border: rgba(0,240,255,0.2);
}
body {
  font-family: 'Inter', 'Noto Sans SC', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: var(--bg);
  color: var(--text);
  font-size: 15px;
  line-height: 1.6;
  min-height: 100vh;
  overflow-x: hidden;
  position: relative;
}
/* 动态粒子背景 */
#particles {
  position: fixed;
  top: 0; left: 0;
  width: 100%; height: 100%;
  z-index: 0;
  pointer-events: none;
}
/* 网格背景 */
body::before {
  content: '';
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background:
    linear-gradient(rgba(0,240,255,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,240,255,0.03) 1px, transparent 1px);
  background-size: 50px 50px;
  z-index: 0;
  pointer-events: none;
  animation: gridMove 20s linear infinite;
}
@keyframes gridMove {
  0% { background-position: 0 0; }
  100% { background-position: 50px 50px; }
}
/* 角落光晕 */
body::after {
  content: '';
  position: fixed;
  top: -20%; left: -10%;
  width: 60%; height: 60%;
  background: radial-gradient(circle, rgba(0,240,255,0.08), transparent 70%);
  z-index: 0;
  pointer-events: none;
}
.corner-glow {
  position: fixed;
  bottom: -20%; right: -10%;
  width: 50%; height: 50%;
  background: radial-gradient(circle, rgba(255,0,170,0.06), transparent 70%);
  z-index: 0;
  pointer-events: none;
}

/* 顶部HUD栏 */
.hud-top {
  position: relative;
  z-index: 10;
  background: rgba(5,5,8,0.9);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid var(--border);
  padding: 0 24px;
  height: 64px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.hud-top::before, .hud-top::after {
  content: '';
  position: absolute;
  bottom: -1px;
  width: 100px; height: 2px;
  background: linear-gradient(90deg, transparent, var(--cyan));
}
.hud-top::before { left: 0; }
.hud-top::after { right: 0; background: linear-gradient(90deg, var(--magenta), transparent); }

.logo-area { display: flex; align-items: center; gap: 16px; }
.logo-icon {
  width: 40px; height: 40px;
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
}
.logo-icon::before {
  content: '';
  position: absolute;
  inset: 0;
  border: 2px solid var(--cyan);
  transform: rotate(45deg);
  box-shadow: 0 0 15px rgba(0,240,255,0.4), inset 0 0 15px rgba(0,240,255,0.1);
}
.logo-icon span {
  font-family: 'Inter', sans-serif;
  font-weight: 900;
  font-size: 18px;
  color: var(--cyan);
  text-shadow: 0 0 10px var(--cyan);
  z-index: 1;
}
.logo-text {
  font-family: 'Anthropic Serif', Georgia, serif;
  font-weight: 700;
  font-size: 16px;
  letter-spacing: 3px;
  color: var(--text);
}
.logo-text .accent { color: var(--cyan); text-shadow: 0 0 10px var(--cyan); }
.logo-sub {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  color: var(--text2);
  letter-spacing: 2px;
  margin-top: 2px;
}

.nav-tabs { display: flex; gap: 2px; }
.nav-tab {
  padding: 8px 18px;
  font-size: 12px;
  font-weight: 500;
  color: var(--text2);
  cursor: pointer;
  transition: all 0.2s;
  position: relative;
  font-family: 'JetBrains Mono', monospace;
  letter-spacing: 1px;
  text-transform: uppercase;
  border: 1px solid transparent;
}
.nav-tab:hover { color: var(--cyan); background: rgba(0,240,255,0.05); }
.nav-tab.active {
  color: var(--cyan);
  background: rgba(0,240,255,0.08);
  border-color: var(--border);
  text-shadow: 0 0 8px rgba(0,240,255,0.5);
}
.nav-tab.active::before {
  content: '';
  position: absolute;
  top: 0; left: 0;
  width: 8px; height: 8px;
  border-top: 2px solid var(--cyan);
  border-left: 2px solid var(--cyan);
}
.nav-tab.active::after {
  content: '';
  position: absolute;
  bottom: 0; right: 0;
  width: 8px; height: 8px;
  border-bottom: 2px solid var(--cyan);
  border-right: 2px solid var(--cyan);
}

.hud-right { display: flex; align-items: center; gap: 16px; }
.status-module {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 6px 14px;
  border: 1px solid var(--border);
  background: rgba(0,240,255,0.03);
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  letter-spacing: 1px;
  position: relative;
}
.status-module::before {
  content: '';
  position: absolute;
  top: -1px; left: -1px;
  width: 6px; height: 6px;
  border-top: 2px solid var(--cyan);
  border-left: 2px solid var(--cyan);
}
.status-module::after {
  content: '';
  position: absolute;
  bottom: -1px; right: -1px;
  width: 6px; height: 6px;
  border-bottom: 2px solid var(--cyan);
  border-right: 2px solid var(--cyan);
}
.led {
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--text2);
  box-shadow: 0 0 8px currentColor;
}
.led.ok { background: var(--green); color: var(--green); animation: pulse 2s infinite; }
.led.run { background: var(--yellow); color: var(--yellow); animation: pulse 0.8s infinite; }
.led.err { background: var(--red); color: var(--red); }
@keyframes pulse { 0%,100%{opacity:1;box-shadow:0 0 8px currentColor;} 50%{opacity:0.4;box-shadow:0 0 2px currentColor;} }

/* 主内容 */
.main {
  position: relative;
  z-index: 5;
  max-width: 1100px;
  margin: 0 auto;
  padding: 28px 24px 80px;
}
.page { display: none; }
.page.active { display: block; animation: glitchIn 0.4s ease; }
@keyframes glitchIn {
  0% { opacity: 0; transform: translateX(-5px); filter: hue-rotate(90deg); }
  50% { transform: translateX(3px); }
  100% { opacity: 1; transform: translateX(0); filter: hue-rotate(0); }
}

/* 页面标题 */
.page-title {
  margin-bottom: 24px;
  position: relative;
  padding-left: 16px;
}
.page-title::before {
  content: '';
  position: absolute;
  left: 0; top: 4px; bottom: 4px;
  width: 3px;
  background: linear-gradient(180deg, var(--cyan), var(--magenta));
  box-shadow: 0 0 10px var(--cyan);
}
.page-title h1 {
  font-family: 'Anthropic Serif', Georgia, 'Noto Serif SC', serif;
  font-size: 24px;
  font-weight: 700;
  letter-spacing: 2px;
  margin-bottom: 4px;
}
.page-title h1 .glow { color: var(--cyan); text-shadow: 0 0 15px rgba(0,240,255,0.5); }
.page-title p {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--text2);
  letter-spacing: 1px;
}

/* HUD卡片 */
.hud-card {
  background: rgba(10,10,16,0.7);
  backdrop-filter: blur(8px);
  border: 1px solid var(--border);
  margin-bottom: 16px;
  position: relative;
}
.hud-card::before {
  content: '';
  position: absolute;
  top: -1px; left: 20px;
  width: 40px; height: 2px;
  background: var(--cyan);
  box-shadow: 0 0 8px var(--cyan);
}
.hud-card-head {
  padding: 12px 18px;
  border-bottom: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.hud-card-head .title {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 2px;
  color: var(--cyan);
  text-transform: uppercase;
}
.hud-card-head .badge {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  padding: 2px 8px;
  border: 1px solid var(--border);
  color: var(--magenta);
  letter-spacing: 1px;
}
.hud-card-body { padding: 18px; }

/* 步骤 */
.steps {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 10px;
  margin-bottom: 20px;
}
@media (max-width: 800px) { .steps { grid-template-columns: repeat(2, 1fr); } }
.step {
  background: rgba(10,10,16,0.6);
  border: 1px solid var(--border);
  padding: 16px 12px;
  text-align: center;
  transition: all 0.3s;
  position: relative;
  cursor: default;
}
.step::before {
  content: '';
  position: absolute;
  top: 0; left: 0;
  width: 100%; height: 2px;
  background: linear-gradient(90deg, transparent, var(--cyan), transparent);
  opacity: 0;
  transition: opacity 0.3s;
}
.step:hover {
  border-color: var(--cyan);
  transform: translateY(-4px);
  box-shadow: 0 8px 30px rgba(0,240,255,0.15);
}
.step:hover::before { opacity: 1; }
.step-num {
  font-family: 'Anthropic Serif', Georgia, serif;
  font-size: 28px;
  font-weight: 900;
  color: transparent;
  -webkit-text-stroke: 1px var(--cyan);
  margin-bottom: 8px;
  text-shadow: 0 0 20px rgba(0,240,255,0.3);
}
.step:hover .step-num {
  color: var(--cyan);
  -webkit-text-stroke: 0;
  text-shadow: 0 0 15px var(--cyan);
}
.step-title {
  font-weight: 600;
  font-size: 12px;
  margin-bottom: 4px;
  color: var(--text);
  letter-spacing: 1px;
}
.step-desc {
  font-size: 10px;
  color: var(--text2);
  line-height: 1.5;
  font-family: 'JetBrains Mono', monospace;
}

/* 统计 */
.stats {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
  margin-bottom: 20px;
}
.stat {
  background: rgba(10,10,16,0.6);
  border: 1px solid var(--border);
  padding: 16px 18px;
  position: relative;
  overflow: hidden;
}
.stat::after {
  content: '';
  position: absolute;
  top: 0; right: 0;
  width: 80px; height: 80px;
  background: radial-gradient(circle at top right, rgba(0,240,255,0.1), transparent);
}
.stat .num {
  font-family: 'Anthropic Serif', Georgia, serif;
  font-size: 36px;
  font-weight: 900;
  color: var(--cyan);
  text-shadow: 0 0 20px rgba(0,240,255,0.4);
  letter-spacing: 2px;
  line-height: 1;
}
.stat .label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  color: var(--text2);
  margin-top: 6px;
  text-transform: uppercase;
  letter-spacing: 2px;
}

/* 两列 */
.two-col {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}
@media (max-width: 700px) { .two-col { grid-template-columns: 1fr; } }

/* 状态 */
.state-row {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
}
.state-text {
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
  letter-spacing: 1px;
}

/* 进度 */
.prog-head {
  display: flex;
  justify-content: space-between;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--text2);
  margin-bottom: 8px;
  letter-spacing: 1px;
}
.prog-track {
  height: 10px;
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--border);
  position: relative;
  overflow: hidden;
}
.prog-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--cyan), var(--magenta));
  box-shadow: 0 0 15px rgba(0,240,255,0.5);
  transition: width 0.5s ease;
  position: relative;
}
.prog-fill::after {
  content: '';
  position: absolute;
  top: 0; right: 0; bottom: 0;
  width: 40px;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4));
  animation: shimmer 1.2s infinite;
}
@keyframes shimmer { 0%{transform:translateX(-40px);} 100%{transform:translateX(40px);} }
.prog-hint {
  margin-top: 8px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  color: var(--text2);
  letter-spacing: 1px;
}

/* 按钮 */
.btn-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
.btn {
  padding: 10px 22px;
  background: transparent;
  border: 1px solid var(--border);
  color: var(--text);
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  font-family: 'JetBrains Mono', monospace;
  letter-spacing: 1px;
  text-transform: uppercase;
  transition: all 0.2s;
  position: relative;
  overflow: hidden;
}
.btn::before {
  content: '';
  position: absolute;
  top: 0; left: -100%;
  width: 100%; height: 100%;
  background: linear-gradient(90deg, transparent, rgba(0,240,255,0.1), transparent);
  transition: left 0.4s;
}
.btn:hover::before { left: 100%; }
.btn:hover {
  border-color: var(--cyan);
  color: var(--cyan);
  box-shadow: 0 0 20px rgba(0,240,255,0.2), inset 0 0 20px rgba(0,240,255,0.05);
  text-shadow: 0 0 8px rgba(0,240,255,0.5);
}
.btn.primary {
  background: rgba(0,240,255,0.08);
  border-color: var(--cyan);
  color: var(--cyan);
  text-shadow: 0 0 8px rgba(0,240,255,0.5);
}
.btn.primary:hover {
  background: rgba(0,240,255,0.15);
  box-shadow: 0 0 30px rgba(0,240,255,0.3);
}
.btn.danger { border-color: rgba(255,51,102,0.3); color: var(--red); }
.btn.danger:hover { border-color: var(--red); box-shadow: 0 0 20px rgba(255,51,102,0.2); text-shadow: 0 0 8px rgba(255,51,102,0.5); }
.btn:disabled { opacity: 0.3; cursor: not-allowed; }

/* 提示 */
.alert-box {
  padding: 12px 16px;
  background: rgba(0,240,255,0.04);
  border: 1px solid rgba(0,240,255,0.2);
  border-left: 3px solid var(--cyan);
  font-size: 12px;
  color: var(--text2);
  line-height: 1.7;
  font-family: 'JetBrains Mono', monospace;
}

/* 表格 */
.data-table { width: 100%; border-collapse: collapse; }
.data-table td {
  padding: 9px 12px;
  border-bottom: 1px solid rgba(0,240,255,0.06);
  font-size: 12px;
}
.data-table tr:last-child td { border-bottom: none; }
.data-table tr:hover { background: rgba(0,240,255,0.03); }
.data-table td:first-child {
  color: var(--cyan);
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  width: 35%;
  letter-spacing: 1px;
}
.chip {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 2px 10px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 1px;
  border: 1px solid;
}
.chip.ok { color: var(--green); border-color: rgba(0,255,136,0.3); background: rgba(0,255,136,0.05); }
.chip.fail { color: var(--red); border-color: rgba(255,51,102,0.3); background: rgba(255,51,102,0.05); }
.chip.warn { color: var(--yellow); border-color: rgba(255,204,0,0.3); background: rgba(255,204,0,0.05); }

/* 日志 */
.log-area {
  background: #020204;
  padding: 16px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  line-height: 1.8;
  color: #505070;
  height: 440px;
  overflow-y: auto;
  position: relative;
}
.log-area::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 40px;
  background: linear-gradient(180deg, #020204, transparent);
  pointer-events: none;
  z-index: 1;
}
.log-area::-webkit-scrollbar { width: 4px; }
.log-area::-webkit-scrollbar-track { background: transparent; }
.log-area::-webkit-scrollbar-thumb { background: rgba(0,240,255,0.2); }

/* 扫描线 */
.scanline {
  position: fixed;
  top: 0; left: 0; right: 0;
  height: 100px;
  background: linear-gradient(180deg, transparent, rgba(0,240,255,0.02), transparent);
  animation: scan 6s linear infinite;
  pointer-events: none;
  z-index: 100;
}
@keyframes scan { 0%{top:-100px;} 100%{top:100vh;} }

/* 故障文字效果 */
.glitch-text {
  position: relative;
  display: inline-block;
}
.glitch-text::before, .glitch-text::after {
  content: attr(data-text);
  position: absolute;
  top: 0; left: 0;
  width: 100%;
}
.glitch-text::before {
  color: var(--magenta);
  animation: glitch1 3s infinite;
  clip-path: polygon(0 0, 100% 0, 100% 33%, 0 33%);
}
.glitch-text::after {
  color: var(--cyan);
  animation: glitch2 3s infinite;
  clip-path: polygon(0 67%, 100% 67%, 100% 100%, 0 100%);
}
@keyframes glitch1 {
  0%,90%,100% { transform: translate(0); }
  92% { transform: translate(-2px, 1px); }
  94% { transform: translate(2px, -1px); }
  96% { transform: translate(-1px, 0); }
}
@keyframes glitch2 {
  0%,90%,100% { transform: translate(0); }
  92% { transform: translate(2px, -1px); }
  94% { transform: translate(-2px, 1px); }
  96% { transform: translate(1px, 0); }
}
</style>
</head>
<body>
<canvas id="particles"></canvas>
<div class="corner-glow"></div>
<div class="scanline"></div>

<header class="hud-top">
  <div class="logo-area">
    <div class="logo-icon"><span>B</span></div>
    <div>
      <div class="logo-text">BEAT<span class="accent">FCN</span> v22</div>
      <div class="logo-sub">神经网络训练界面 // 版本 22.0</div>
    </div>
  </div>
  <nav class="nav-tabs">
    <div class="nav-tab active" onclick="switchPage('start', this)">快速开始</div>
    <div class="nav-tab" onclick="switchPage('control', this)">训练控制</div>
    <div class="nav-tab" onclick="switchPage('env', this)">环境检查</div>
    <div class="nav-tab" onclick="switchPage('log', this)">运行日志</div>
  </nav>
  <div class="hud-right">
    <div class="status-module" id="top-status"><span class="led"></span>待机</div>
  </div>
</header>

<main class="main">

  <!-- 快速开始 -->
  <section class="page active" id="page-start">
    <div class="page-title">
      <h1><span class="glow">神经网络</span>训练控制台</h1>
      <p>// 系统已初始化 // 等待操作指令</p>
    </div>
    <div class="steps">
      <div class="step"><div class="step-num">01</div><div class="step-title">环境检查</div><div class="step-desc">验证 Python / CUDA / Dataset</div></div>
      <div class="step"><div class="step-num">02</div><div class="step-title">启动训练</div><div class="step-desc">200 Epochs · AMP</div></div>
      <div class="step"><div class="step-num">03</div><div class="step-title">断点续训</div><div class="step-desc">Auto Resume from Checkpoint</div></div>
      <div class="step"><div class="step-num">04</div><div class="step-title">模型评估</div><div class="step-desc">Grid F1 · Threshold Scan</div></div>
      <div class="step"><div class="step-num">05</div><div class="step-title">交付权重</div><div class="step-desc">Best Checkpoint Export</div></div>
    </div>
    <div class="stats">
      <div class="stat"><div class="num">200</div><div class="label">Epochs</div></div>
      <div class="stat"><div class="num">~12H</div><div class="label">ETA / RTX 3090</div></div>
      <div class="stat"><div class="num">659K</div><div class="label">Parameters</div></div>
    </div>
    <div class="hud-card">
      <div class="hud-card-head"><span class="title">系统公告</span><span class="badge">注意</span></div>
      <div class="hud-card-body">
        <div class="alert-box">训练进程在后台独立运行，关闭浏览器不影响训练。随时重新打开此页面查看实时进度与日志。若训练意外中断，使用「断点续训」从最近 checkpoint 恢复，无需从头开始。</div>
      </div>
    </div>
  </section>

  <!-- 训练控制 -->
  <section class="page" id="page-control">
    <div class="page-title">
      <h1><span class="glow">训练</span>控制中心</h1>
      <p>// 启动 / 续训 / 评估 / 终止</p>
    </div>
    <div class="two-col">
      <div class="hud-card">
        <div class="hud-card-head"><span class="title">运行状态</span><span class="badge" id="state-tag">待机</span></div>
        <div class="hud-card-body">
          <div class="state-row"><span class="led" id="state-led"></span><span class="state-text" id="state-text">等待指令</span></div>
          <div class="prog-head"><span>训练进度</span><span id="prog-num">0 / 200</span></div>
          <div class="prog-track"><div class="prog-fill" id="prog-fill" style="width:0%"></div></div>
          <div class="prog-hint">// 平均每轮3-5分钟 // 后台守护进程运行中</div>
        </div>
      </div>
      <div class="hud-card">
        <div class="hud-card-head"><span class="title">操作面板</span><span class="badge">操作</span></div>
        <div class="hud-card-body">
          <div class="btn-row">
            <button class="btn primary" onclick="startTrain(false)" id="btn-train">开始训练</button>
            <button class="btn" onclick="startTrain(true)" id="btn-resume">断点续训</button>
            <button class="btn" onclick="startReval()" id="btn-reval">评估选冠军</button>
            <button class="btn danger" onclick="stopProcess()" id="btn-stop" style="display:none;">终止</button>
          </div>
          <div class="alert-box" id="tip">首次使用请先执行「环境检查」确认依赖与数据就绪。</div>
        </div>
      </div>
    </div>
    <div class="hud-card">
      <div class="hud-card-head"><span class="title">Hyperparameters</span><span class="badge">配置</span></div>
      <div class="hud-card-body">
        <table class="data-table">
          <tr><td>epochs</td><td>200（无早停，跑满）</td></tr>
          <tr><td>batch_size</td><td>auto-detect VRAM → 3090/4090 = 4</td></tr>
          <tr><td>learning_rate</td><td>5e-4，5-epoch warmup + cosine decay</td></tr>
          <tr><td>architecture</td><td>Transformer d=96, 4 layers, 4 heads, 659K params</td></tr>
          <tr><td>dataset</td><td>train 11685 / val 1174 / test 300</td></tr>
          <tr><td>checkpoint</td><td>每轮 ckpt_v22_ep{N}.pt + best beat_fcn_best_v22.pt</td></tr>
        </table>
      </div>
    </div>
  </section>

  <!-- 环境检查 -->
  <section class="page" id="page-env">
    <div class="page-title">
      <h1><span class="glow">环境</span>诊断</h1>
      <p>// 运行时验证与数据完整性检查</p>
    </div>
    <div class="hud-card">
      <div class="hud-card-body">
        <div class="btn-row"><button class="btn primary" onclick="checkEnv()">执行诊断</button></div>
        <div id="env-result" style="display:none; margin-top: 16px;">
          <table class="data-table" id="env-table"></table>
          <div class="alert-box" style="margin-top:16px;">Python 推荐 3.9 / 3.10；PyTorch 需 CUDA 版本；数据目录需包含 grid_npz_full128/ 与 pitch_npz_full/。</div>
        </div>
      </div>
    </div>
  </section>

  <!-- 运行日志 -->
  <section class="page" id="page-log">
    <div class="page-title">
      <h1><span class="glow">运行</span>日志</h1>
      <p>// 实时训练输出流</p>
    </div>
    <div class="hud-card" style="margin-bottom:0;">
      <div class="hud-card-head"><span class="title">标准输出流</span><span class="badge">实时</span></div>
      <div class="log-area" id="log-box">// 等待指令输入...</div>
    </div>
  </section>

</main>

<script>
// 粒子背景
const canvas = document.getElementById('particles');
const ctx = canvas.getContext('2d');
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
const particles = [];
for (let i = 0; i < 60; i++) {
  particles.push({
    x: Math.random() * canvas.width,
    y: Math.random() * canvas.height,
    vx: (Math.random() - 0.5) * 0.3,
    vy: (Math.random() - 0.5) * 0.3,
    size: Math.random() * 2 + 0.5,
    color: Math.random() > 0.5 ? '0,240,255' : '255,0,170'
  });
}
function animateParticles() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  particles.forEach(p => {
    p.x += p.vx; p.y += p.vy;
    if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
    if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(${p.color},0.4)`;
    ctx.fill();
  });
  // 连线
  for (let i = 0; i < particles.length; i++) {
    for (let j = i + 1; j < particles.length; j++) {
      const dx = particles[i].x - particles[j].x;
      const dy = particles[i].y - particles[j].y;
      const dist = Math.sqrt(dx*dx + dy*dy);
      if (dist < 120) {
        ctx.beginPath();
        ctx.moveTo(particles[i].x, particles[i].y);
        ctx.lineTo(particles[j].x, particles[j].y);
        ctx.strokeStyle = `rgba(0,240,255,${0.1 * (1 - dist/120)})`;
        ctx.lineWidth = 0.5;
        ctx.stroke();
      }
    }
  }
  requestAnimationFrame(animateParticles);
}
animateParticles();
window.addEventListener('resize', () => { canvas.width = window.innerWidth; canvas.height = window.innerHeight; });

let polling = false;
function switchPage(name, el) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
  document.getElementById('page-' + name).classList.add('active');
  el.classList.add('active');
}
async function api(url) { const r = await fetch(url); return r.json(); }
function setTopStatus(text, cls) {
  const el = document.getElementById('top-status');
  el.innerHTML = '<span class="led ' + (cls||'') + '"></span>' + text;
}
function setLed(cls) { document.getElementById('state-led').className = 'led ' + (cls || ''); }
function setButtons(running) {
  ['btn-train','btn-resume','btn-reval'].forEach(id => document.getElementById(id).disabled = running);
  document.getElementById('btn-stop').style.display = running ? 'inline-block' : 'none';
}
async function checkEnv() {
  setTopStatus('扫描中', 'run');
  const r = await api('/api/check');
  document.getElementById('env-result').style.display = 'block';
  let rows = '';
  rows += '<tr><td>python</td><td>' + r.python.version + ' <span class="chip ' + (r.python.ok?'ok':'warn') + '">' + (r.python.ok?'OK':'WARN') + '</span></td></tr>';
  rows += '<tr><td>pytorch</td><td>' + (r.torch.version||'-') + ' / ' + (r.torch.device||'-') + ' <span class="chip ' + (r.torch.ok?'ok':'fail') + '">' + (r.torch.ok?'OK':'FAIL') + '</span></td></tr>';
  if (r.torch.gpu) rows += '<tr><td>gpu</td><td>' + r.torch.gpu + ' (' + (r.torch.vram||'-') + ')</td></tr>';
  r.files.forEach(f => rows += '<tr><td>' + f.name + '</td><td><span class="chip ' + (f.ok?'ok':'fail') + '">' + (f.ok?'PRESENT':'MISSING') + '</span></td></tr>');
  r.deps.forEach(d => rows += '<tr><td>' + d.name + '</td><td><span class="chip ' + (d.ok?'ok':'fail') + '">' + (d.ok?'INSTALLED':'MISSING') + '</span></td></tr>');
  document.getElementById('env-table').innerHTML = rows;
  const allOk = r.python.ok && r.torch.ok && r.files.every(f=>f.ok) && r.deps.every(d=>d.ok);
  setTopStatus(allOk ? '就绪' : '有问题', allOk ? 'ok' : 'err');
}
async function startTrain(resume) {
  const label = resume ? 'RESUMING' : 'TRAINING';
  setLed('run'); document.getElementById('state-text').textContent = (label === 'TRAINING' ? '训练中' : label === 'RESUMING' ? '续训中' : '评估中') + '...';
  document.getElementById('state-tag').textContent = label === 'TRAINING' ? '训练中' : label === 'RESUMING' ? '续训中' : '评估';
  setTopStatus(label === 'TRAINING' ? '训练中' : label === 'RESUMING' ? '续训中' : '评估中', 'run'); setButtons(true);
  document.getElementById('tip').textContent = '训练进行中，切换至「运行日志」查看实时输出。';
  await api(resume ? '/api/resume' : '/api/train');
  if (!polling) { polling = true; pollLog(); }
  switchPage('log', document.querySelectorAll('.nav-tab')[3]);
}
async function startReval() {
  setLed('run'); document.getElementById('state-text').textContent = '评估中...';
  document.getElementById('state-tag').textContent = '评估';
  setTopStatus('评估中', 'run'); setButtons(true);
  document.getElementById('tip').textContent = '正在扫描所有检查点...';
  await api('/api/reval');
  if (!polling) { polling = true; pollLog(); }
  switchPage('log', document.querySelectorAll('.nav-tab')[3]);
}
async function stopProcess() {
  await api('/api/stop');
  document.getElementById('tip').textContent = '终止信号已发送，正在保存检查点...';
}
async function pollLog() {
  const r = await api('/api/log');
  const box = document.getElementById('log-box');
  box.textContent = r.log.join('\n');
  box.scrollTop = box.scrollHeight;
  if (r.action) {
    setLed('run'); document.getElementById('state-text').textContent = r.action + '...';
    const p = await api('/api/progress');
    if (p.current_epoch > 0) {
      const pct = (p.current_epoch / p.total_epochs * 100).toFixed(1);
      document.getElementById('prog-fill').style.width = pct + '%';
      document.getElementById('prog-num').textContent = p.current_epoch + ' / ' + p.total_epochs;
    }
  } else {
    polling = false; setButtons(false); setLed('ok');
    document.getElementById('state-text').textContent = '已完成';
    document.getElementById('state-tag').textContent = '完成';
    setTopStatus('待机', '');
    document.getElementById('tip').textContent = '操作完成。训练结束后请运行「评估选冠军」。';
  }
  if (polling) setTimeout(pollLog, 1000);
}
window.onload = async () => {
  const r = await api('/api/log');
  if (r.action) {
    document.getElementById('log-box').textContent = r.log.join('\n');
    setLed('run'); setButtons(true); polling = true; pollLog();
  }
};
</script>
</body>
</html>"""

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # 静默HTTP日志

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))

        elif path == '/api/check':
            result = check_environment()
            self._json(result)

        elif path == '/api/train':
            if current_action:
                self._json({"error": "已有任务在运行"})
                return
            cmd = build_train_cmd(resume=False)
            t = threading.Thread(target=run_command, args=(cmd, "训练"), daemon=True)
            t.start()
            self._json({"ok": True})

        elif path == '/api/resume':
            if current_action:
                self._json({"error": "已有任务在运行"})
                return
            cmd = build_train_cmd(resume=True)
            t = threading.Thread(target=run_command, args=(cmd, "断点续训"), daemon=True)
            t.start()
            self._json({"ok": True})

        elif path == '/api/reval':
            if current_action:
                self._json({"error": "已有任务在运行"})
                return
            cmd = [sys.executable, 'reval_v22.py']
            t = threading.Thread(target=run_command, args=(cmd, "reval选冠军"), daemon=True)
            t.start()
            self._json({"ok": True})

        elif path == '/api/stop':
            global current_process
            if current_process:
                current_process.terminate()
                self._json({"ok": True, "message": "已发送停止信号"})
            else:
                self._json({"ok": False, "message": "没有运行中的任务"})

        elif path == '/api/log':
            with log_lock:
                log = current_log.copy()
            self._json({"log": log, "action": current_action})

        elif path == '/api/progress':
            self._json(get_progress())

        else:
            self.send_response(404)
            self.end_headers()

    def _json(self, data):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1',
                    help='绑定地址（默认127.0.0.1仅本地访问；需要远程访问时设0.0.0.0，注意安全风险）')
    ap.add_argument('--port', type=int, default=8000)
    args = ap.parse_args()
    port = args.port
    print(f"\n🐋 BeatFCN v22 训练控制台已启动")
    print(f"   绑定地址: {args.host}:{port}")
    print(f"   浏览器打开: http://localhost:{port}")
    print(f"   按 Ctrl+C 退出\n")
    server = HTTPServer((args.host, port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 再见！")
        server.shutdown()

if __name__ == '__main__':
    main()
