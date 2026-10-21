"""对 grid_npz_clean 里所有（含 train）谱面跑 Layer1 检测并缓存到 <out>/<fid>.npz（已存在则跳过）。"""
import sys, os, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gen.layer1 import detect
from extract_full import load_zip
from chart_utils import decode_audio
root, out = sys.argv[1:3]
for f in sorted(glob.glob(f'{root}/grid_npz_clean/*.npz')):
    fid = os.path.basename(f)[:-4]; c = f'{out}/{fid}.npz'
    if os.path.exists(c): continue
    y = decode_audio(load_zip(f'{root}/charts/{fid.split("_")[0]}.zip')[2])
    if y is None: continue
    b, d, on = detect(y); np.savez(c, beats=b, downs=d, on=on); print(fid, flush=True)
