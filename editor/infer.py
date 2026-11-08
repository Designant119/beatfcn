"""编辑器推理入口:模型自然采音,两种出口任选。

两级流水,故意分开、各自可独立测试:

 1. 出谱:调用仓库自带的 draft_song(不重新实现推理,推理细节归仓库管):
      python -m scripts.draft_song --audio S --out-dir D --mode phira
          --profile none --consensus 1
    不给 --tier/--density 即 unconditioned(draft_song.py:579:密度曲线不指定+控制全放开,
    模型自己听歌定密度);--profile none 跳过全部可玩性清理;--consensus 1 单采样,
    不加投票过滤。这是编辑器集成的既定默认(2026-11-08 定案)。
 2. 出口二选一:
    默认 = 换算:用编辑器的权威 BPM 图把毫秒事件换算成 [整拍,分子,分母] 候选
      (editor/ms_to_beats.py,自测 12/12),输出 editor-notes.json;
    --raw = 纯推理:不做任何换算/吸附,模型的毫秒输出原样导出成
      editor-notes-raw.json(不要 BPM 图;编辑器自己定位,吸附与否编辑器说了算)。

用法(有 torch + 权重的机器):
  python editor/infer.py --audio song.mp3 --bpm-map bpm.json --model ckpt.pt --out-dir out/   # 换算版
  python editor/infer.py --audio song.mp3 --raw --model ckpt.pt --out-dir out/                # 纯毫秒版
无 torch 的机器只测第二级:
  python editor/infer.py --from-draft out/draft-events.json --bpm-map bpm.json
  python editor/infer.py --from-draft out/draft-events.json --raw

BPM 图格式:与 Phira 谱面一致 {"BPMList":[{"bpm":174,"startTime":[0,0,1]},...]},
或直接给列表 [{"bpm":...,"startTime":...},...]。startTime 为 [整拍, 分子, 分母]。
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ms_to_beats import convert_events  # noqa: E402


def load_bpm_map(path):
    data = json.loads(Path(path).read_text())
    return data['BPMList'] if isinstance(data, dict) else data


def run_draft_song(args):
    """第一级:仓库自带出谱脚本,unconditioned + 无清理 + 单采样。"""
    cmd = [sys.executable, '-m', 'scripts.draft_song',
           '--audio', str(args.audio), '--out-dir', str(args.out_dir),
           '--mode', args.mode, '--profile', 'none', '--consensus', str(args.consensus)]
    if args.model:
        cmd += ['--model', str(args.model)]
    if args.device:
        cmd += ['--device', args.device]
    if args.mert_source:
        cmd += ['--mert-source', args.mert_source]
    print('[1/2] 出谱:', ' '.join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        raise SystemExit(f'draft_song 失败(exit {proc.returncode});上卡前先确认 torch/权重/MERT 源齐备')
    return args.out_dir / 'draft-events.json'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--audio', type=Path, help='输入音频(第一级)')
    p.add_argument('--from-draft', dest='from_draft', type=Path,
                   help='跳过第一级,直接换算已有的 draft-events.json(无 torch 环境测试用)')
    p.add_argument('--raw', action='store_true',
                   help='纯推理:毫秒输出原样导出,不换算不吸附(不需要 --bpm-map)')
    p.add_argument('--bpm-map', type=Path, help='编辑器 BPM 图 JSON(--raw 时不需要)')
    p.add_argument('--out-dir', type=Path, help='输出目录(默认: draft 文件所在目录)')
    p.add_argument('--model', type=Path, help='权重:导出模型或训练 checkpoint')
    p.add_argument('--mode', default='phira', choices=('phira', 'mania_4k'))
    p.add_argument('--device', choices=('cuda', 'mps', 'cpu'))
    p.add_argument('--mert-source', help='pinned MERT-v2 目录或 hf')
    p.add_argument('--consensus', type=int, default=1, help='采样数,1=模型单次自然发挥(默认)')
    p.add_argument('--clock-shift-ms', type=float, default=0.0,
                   help='模型钟->编辑器钟常数偏移(Phira 播放钟: mp3 约 -2ms, ogg 约 +10ms;BPM 图'
                        '与模型同钟则保持 0)')
    p.add_argument('--snap-dens', help='编辑器吸附菜单分母集合,如 "1,2,4,8,16,3,6,12,24";'
                                       '传入后数值巧合格被排除,小误差音的换算直接定案')
    args = p.parse_args()

    if not args.from_draft and not args.audio:
        p.error('需要 --audio(出谱)或 --from-draft(只导出)其一')
    if not args.raw and not args.bpm_map:
        p.error('换算模式需要 --bpm-map;或者加 --raw 走纯毫秒输出')

    if args.from_draft:
        draft_path = args.from_draft
        out_dir = args.out_dir or draft_path.parent
    else:
        out_dir = args.out_dir or Path('editor-out')
        draft_path = run_draft_song(args)

    draft = json.loads(draft_path.read_text())

    if args.raw:
        events = draft['events']
        duration_s = max(draft.get('duration_us', 0) / 1e6, 1e-6)
        types = {}
        for e in events:
            types[e['type']] = types.get(e['type'], 0) + 1
        result = dict(source=str(draft_path.name), mode=draft.get('mode'), format='raw_ms',
                      note='模型钟毫秒,未做任何换算/吸附/清理;定位与吸附由编辑器决定',
                      notes=events,
                      summary=dict(total=len(events),
                                   notes_per_second=round(len(events) / duration_s, 3), types=types))
        out_path = out_dir / 'editor-notes-raw.json'
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1))
        print(f'[2/2] 纯毫秒导出: {out_path}  共 {len(events)} 音 '
              f'({len(events) / duration_s:.2f} 音/秒) {types}')
        return
    bpm_list = load_bpm_map(args.bpm_map)
    allowed = [int(x) for x in args.snap_dens.split(',')] if args.snap_dens else None

    notes = convert_events(draft['events'], bpm_list,
                           clock_shift_us=int(args.clock_shift_ms * 1000), allowed_dens=allowed)
    sure = sum(1 for n in notes if n['onset_sure'] and n['end_sure'])
    result = dict(
        source=str(draft_path.name), mode=draft.get('mode'), clock='model(换算前已加 clock-shift)',
        bpm_map=str(args.bpm_map), snap_dens=allowed,
        notes=notes, summary=dict(total=len(notes), sure=sure, need_confirm=len(notes) - sure))
    out_path = out_dir / 'editor-notes.json'
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(f'[2/2] 换算完成: {out_path}  共 {len(notes)} 音,确定 {sure},待谱师确认 {len(notes) - sure}')


if __name__ == '__main__':
    main()
