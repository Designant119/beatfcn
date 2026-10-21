#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BeatFCN v22 训练辅助脚本
给帮忙跑训练的同学用的，不需要懂命令行，按菜单选就行。

用法：
    python train_helper.py

功能：
    1. 检查环境（Python/GPU/依赖/文件）
    2. 开始训练
    3. 断点续训
    4. 跑reval选冠军
    5. 查看训练进度
"""

import os
import sys
import subprocess
import platform

# 颜色输出（Windows 10+ 支持 ANSI）
class C:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header():
    print(f"\n{C.CYAN}{C.BOLD}" + "="*60 + C.END)
    print(f"{C.CYAN}{C.BOLD}  BeatFCN v22 训练辅助工具{C.END}")
    print(f"{C.CYAN}" + "="*60 + C.END + "\n")

def check_python():
    """检查Python版本"""
    print(f"{C.YELLOW}[检查] Python版本...{C.END}", end=" ")
    v = sys.version_info
    if v.major == 3 and v.minor >= 9:
        print(f"{C.GREEN}OK ({v.major}.{v.minor}.{v.micro}){C.END}")
        return True
    else:
        print(f"{C.YELLOW}推荐 ({v.major}.{v.minor}.{v.micro})，建议3.9+{C.END}")
        return False

def check_torch():
    """检查PyTorch和GPU"""
    print(f"{C.YELLOW}[检查] PyTorch和GPU...{C.END}")
    try:
        import torch
        print(f"  PyTorch版本: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  GPU: {C.GREEN}{torch.cuda.get_device_name(0)}{C.END}")
            print(f"  显存: {C.GREEN}{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB{C.END}")
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"  {C.YELLOW}Apple Silicon (MPS)，可以跑但慢很多{C.END}")
            return "mps"
        else:
            print(f"  {C.RED}没有可用GPU！只能用CPU，极慢，不建议{C.END}")
            return "cpu"
    except ImportError:
        print(f"  {C.RED}PyTorch未安装！请先运行: pip install torch{C.END}")
        return None

def check_files():
    """检查必要文件"""
    print(f"{C.YELLOW}[检查] 必要文件...{C.END}")
    required = [
        'beat_fcn_train_v22.py',
        'chart_utils.py',
        'infer_v22.py',
        'reval_v22.py',
        'split_full128_final.json',
    ]
    missing = []
    for f in required:
        if os.path.exists(f):
            print(f"  {C.GREEN}✓{C.END} {f}")
        else:
            print(f"  {C.RED}✗ {f} (缺失！){C.END}")
            missing.append(f)

    # 检查数据文件夹
    data_dirs = ['grid_npz_full128', 'pitch_npz_full']
    for d in data_dirs:
        if os.path.isdir(d):
            count = len([f for f in os.listdir(d) if f.endswith('.npz')])
            print(f"  {C.GREEN}✓{C.END} {d}/ ({count}个npz文件)")
        else:
            print(f"  {C.RED}✗ {d}/ (缺失！){C.END}")
            missing.append(d)

    return len(missing) == 0

def check_deps():
    """检查Python依赖"""
    print(f"{C.YELLOW}[检查] Python依赖...{C.END}")
    deps = ['numpy', 'scipy']  # 音频解码用系统 ffmpeg，不需要 librosa/soundfile
    missing = []
    for dep in deps:
        try:
            __import__(dep)
            print(f"  {C.GREEN}✓{C.END} {dep}")
        except ImportError:
            print(f"  {C.RED}✗ {dep} (未安装){C.END}")
            missing.append(dep)
    if missing:
        print(f"\n  {C.YELLOW}运行以下命令安装缺失依赖:{C.END}")
        print(f"  pip install {' '.join(missing)}")
    return len(missing) == 0

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
    """根据设备和显存推荐batch_size"""
    if device == "cuda":
        try:
            import torch
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if vram >= 20:
                return 4  # 24GB推荐4，拉满可以8
            elif vram >= 12:
                return 2
            else:
                return 1
        except:
            return 4
    elif device == "mps":
        return 1  # Mac统一内存，保守用1
    else:
        return 1

def build_train_cmd(resume=False):
    """构建训练命令"""
    device = get_device()
    bs = get_batch_size(device)
    cmd = [
        sys.executable, 'beat_fcn_train_v22.py',
        '--epochs', '200',
        '--patience', '99999',
        '--tag', 'v22',
        '--device', device,
        '--batch-size', str(bs),
        '--threads', '8',
        '--data', './grid_npz_full128',
        '--split', './split_full128_final.json',
        '--pitch-dir', './pitch_npz_full',
        '--pitch-mode', 'chroma',
        '--max-frames', '30000',
        '--eval-songs', '200',
    ]
    if resume:
        cmd.append('--resume')
    return cmd

def run_cmd(cmd, title):
    """运行命令并实时输出"""
    print(f"\n{C.CYAN}{C.BOLD}[{title}]{C.END}")
    print(f"{C.CYAN}命令: {' '.join(cmd)}{C.END}\n")
    print(f"{C.YELLOW}按 Ctrl+C 可以中断（训练会自动保存断点）{C.END}\n")
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   bufsize=1, universal_newlines=True)
        for line in process.stdout:
            print(line, end='')
        process.wait()
        return process.returncode == 0
    except KeyboardInterrupt:
        print(f"\n\n{C.YELLOW}已中断，断点已保存，可以用'断点续训'继续{C.END}")
        return False

def start_training(resume=False):
    """开始训练"""
    cmd = build_train_cmd(resume)
    title = "断点续训" if resume else "开始训练"
    success = run_cmd(cmd, title)
    if success:
        print(f"\n{C.GREEN}{C.BOLD}训练完成！{C.END}")
        print(f"{C.YELLOW}接下来请选择 '4. 跑reval选冠军'{C.END}")
    return success

def run_reval():
    """跑reval选冠军"""
    cmd = [sys.executable, 'reval_v22.py']
    success = run_cmd(cmd, "reval选冠军")
    if success:
        print(f"\n{C.GREEN}{C.BOLD}reval完成！{C.END}")
        print(f"{C.GREEN}冠军模型已保存为: beat_fcn_best_v22.pt{C.END}")
        print(f"\n{C.YELLOW}{C.BOLD}请把以下文件发给我:{C.END}")
        print(f"  1. beat_fcn_best_v22.pt (最重要)")
        print(f"  2. ckpt_v22.json (每轮分数)")
        print(f"  3. reval_v22_result.json (reval结果)")
    return success

def show_progress():
    """查看训练进度"""
    log_file = 'train_v22.log'
    if not os.path.exists(log_file):
        # 找其他日志文件
        logs = [f for f in os.listdir('.') if f.endswith('.log')]
        if logs:
            log_file = logs[0]
        else:
            print(f"{C.RED}没有找到日志文件{C.END}")
            return

    print(f"\n{C.CYAN}[训练进度] 日志文件: {log_file}{C.END}\n")
    try:
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        # 找包含 [x/200] 的行
        epoch_lines = [l for l in lines if '/200]' in l or '/200]' in l]
        if epoch_lines:
            print(f"{C.GREEN}最近5轮结果:{C.END}")
            for l in epoch_lines[-5:]:
                print(f"  {l.strip()}")
        else:
            print(f"{C.YELLOW}还没有轮次结果，训练可能刚开始{C.END}")
            # 显示最后10行
            print(f"\n{C.YELLOW}最后10行日志:{C.END}")
            for l in lines[-10:]:
                print(f"  {l.strip()}")
    except Exception as e:
        print(f"{C.RED}读取日志失败: {e}{C.END}")

def full_check():
    """完整环境检查"""
    print_header()
    print(f"{C.BOLD}=== 环境检查 ==={C.END}\n")
    check_python()
    device = check_torch()
    check_files()
    check_deps()

    if device == "cuda":
        print(f"\n{C.GREEN}{C.BOLD}✓ 环境检查通过，可以开始训练{C.END}")
    elif device == "mps":
        print(f"\n{C.YELLOW}△ 可以训练（Mac），但会慢很多{C.END}")
    else:
        print(f"\n{C.RED}✗ 没有GPU，不建议训练{C.END}")

def main():
    while True:
        print_header()
        print(f"{C.BOLD}请选择操作:{C.END}\n")
        print(f"  {C.CYAN}1.{C.END} 检查环境（第一次用先跑这个）")
        print(f"  {C.CYAN}2.{C.END} 开始训练（从头开始）")
        print(f"  {C.CYAN}3.{C.END} 断点续训（训练中断后继续）")
        print(f"  {C.CYAN}4.{C.END} 跑reval选冠军（训练完后跑）")
        print(f"  {C.CYAN}5.{C.END} 查看训练进度")
        print(f"  {C.CYAN}0.{C.END} 退出")

        choice = input(f"\n{C.BOLD}输入编号: {C.END}").strip()

        if choice == '1':
            full_check()
        elif choice == '2':
            start_training(resume=False)
        elif choice == '3':
            start_training(resume=True)
        elif choice == '4':
            run_reval()
        elif choice == '5':
            show_progress()
        elif choice == '0':
            print(f"\n{C.GREEN}再见！{C.END}\n")
            break
        else:
            print(f"\n{C.RED}无效选择，请输入0-5{C.END}")

        input(f"\n{C.YELLOW}按回车返回菜单...{C.END}")

if __name__ == '__main__':
    main()
