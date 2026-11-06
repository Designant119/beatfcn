"""Audit fixed-clock capacity and real-pair nesting using the frozen train catalog.

This does not inspect validation targets, fit a model, or relabel charts.
Run from the repository with PYTHONPATH=. .venv/bin/python <this file>.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from beatfcn.model.features import CORE_US
from beatfcn.model.sample import match_f1


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "inputs/r09_positive_pairs/catalog.pt"
PLAN = ROOT / "bench/rhythm-v2/r09-single-field-cpu-fit-v1.json"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            h.update(block)
    return h.hexdigest()


def percentiles(values):
    return {str(p): float(np.percentile(values, p)) for p in (50, 95, 99, 100)}


def identify(entry):
    row = entry["row"]
    return (row["mode"], row["audio_sha256"], row["chart_id"],
            entry["example"].c0_us, entry["target_sha256"])


def heads(entry):
    return sorted({event["onset_us"] for event in entry["truth_events"]})


def main():
    actual_sha = digest(CATALOG)
    expected_sha = json.loads(PLAN.read_text())["input_sha256"][str(CATALOG.relative_to(ROOT))]
    if actual_sha != expected_sha:
        raise ValueError("Frozen training catalog identity changed")
    # This is our own identity-verified local catalog, never an external pickle.
    catalog = torch.load(CATALOG, map_location="cpu", weights_only=False)
    pairs = catalog["train_pairs"]
    unique = {}
    for pair in pairs:
        for entry in pair["entries"]:
            if entry["row"]["split"] != "train":
                raise ValueError("Non-training chart in training audit")
            unique.setdefault(identify(entry), entry)

    mode_rows = defaultdict(list)
    cases = []
    for key, entry in unique.items():
        ex = entry["example"]
        times = heads(entry)
        group_times = sorted({g["t_rel_us"] for g in ex.groups})
        if group_times != [t - ex.c0_us for t in times]:
            raise ValueError(f"Target/representation onset mismatch: {key}")
        if sum(len(g["notes"]) for g in ex.groups) != len(entry["truth_events"]):
            raise ValueError(f"Target/representation note count mismatch: {key}")
        if not all(0 <= t < CORE_US for t in group_times):
            raise ValueError(f"Out-of-core target: {key}")
        clocks = {}
        for cell_us in (10_000, 40_000, 400_000):
            buckets = defaultdict(list)
            for t in group_times:
                buckets[t // cell_us].append(t)
            occupancies = [len(buckets[i]) for i in range(CORE_US // cell_us)]
            clocks[str(cell_us)] = {
                "occupancies": occupancies,
                "overflow_groups_if_capacity": {
                    str(cap): sum(max(0, n - cap) for n in occupancies)
                    for cap in (1, 2, 4, 8)
                },
            }
            if cell_us == 40_000:
                for cell, ts in buckets.items():
                    if len(ts) > 1:
                        selected = [e for e in entry["truth_events"]
                                    if e["onset_us"] - ex.c0_us in ts]
                        cases.append({"mode": key[0], "audio_sha256": key[1],
                                      "chart_id": key[2], "c0_us": key[3],
                                      "cell": cell, "head_offsets_us": ts,
                                      "types": dict(Counter(e["type"] for e in selected))})
        holds = [e for e in entry["truth_events"] if e["type"] == "hold"]
        mode_rows[key[0]].append({
            "chart_id": key[2], "audio_sha256": key[1], "c0_us": key[3],
            "head_count": len(times), "note_count": len(entry["truth_events"]),
            "types": dict(Counter(e["type"] for e in entry["truth_events"])),
            "max_simultaneous_notes": max((len(g["notes"]) for g in ex.groups), default=0),
            "holds": len(holds),
            "holds_crossing_400ms_boundary": sum(
                (e["onset_us"] - ex.c0_us) // 400_000 <
                (e["end_us"] - ex.c0_us - 1) // 400_000 for e in holds),
            "holds_ending_after_core": sum(e["end_us"] > ex.c0_us + CORE_US for e in holds),
            "clocks": clocks,
        })

    pair_rows = []
    for pair in pairs:
        a, b = pair["entries"]
        # This orders by unique onset count, not human difficulty.
        low, high = sorted((a, b), key=lambda e: (len(heads(e)), e["row"]["chart_id"]))
        lt, ht = heads(low), heads(high)
        matched, _, _, f1 = match_f1(lt, ht, 20_000)
        pair_rows.append({
            "pair_id": pair["pair_id"], "mode": pair["mode"],
            "audio_sha256": pair["audio_sha256"], "c0_us": pair["c0_us"],
            "changed_fields": pair["changed_fields"],
            "lower_head_count_chart": low["row"]["chart_id"],
            "higher_head_count_chart": high["row"]["chart_id"],
            "lower_head_count": len(lt), "higher_head_count": len(ht),
            "lower_heads_unmatched_exact": len(set(lt) - set(ht)),
            "lower_heads_unmatched_20ms": len(lt) - matched,
            "time_f1_20ms": f1,
        })

    modes = {}
    for mode, rows in mode_rows.items():
        pr = [p for p in pair_rows if p["mode"] == mode]
        clocks = {}
        for width in ("10000", "40000", "400000"):
            ns = [n for row in rows for n in row["clocks"][width]["occupancies"]]
            clocks[width] = {
                "total_cells": len(ns), "nonempty_cells": sum(n > 0 for n in ns),
                "cells_with_multiple_distinct_heads": sum(n > 1 for n in ns),
                "heads_per_cell_percentiles": percentiles(ns),
                "overflow_heads_if_capacity": {
                    str(cap): sum(max(0, n - cap) for n in ns) for cap in (1, 2, 4, 8)},
            }
        modes[mode] = {
            "unique_chart_windows": len(rows),
            "unique_charts": len({r["chart_id"] for r in rows}),
            "unique_audio": len({r["audio_sha256"] for r in rows}),
            "distinct_heads": sum(r["head_count"] for r in rows),
            "notes": sum(r["note_count"] for r in rows),
            "max_simultaneous_notes": max(r["max_simultaneous_notes"] for r in rows),
            "hold_heads": sum(r["holds"] for r in rows),
            "holds_crossing_400ms_boundary": sum(r["holds_crossing_400ms_boundary"] for r in rows),
            "holds_ending_after_core": sum(r["holds_ending_after_core"] for r in rows),
            "clocks": clocks,
            "pairs": len(pr),
            "pairs_not_nested_exact": sum(p["lower_heads_unmatched_exact"] > 0 for p in pr),
            "pairs_not_nested_20ms": sum(p["lower_heads_unmatched_20ms"] > 0 for p in pr),
            "changed_field_count_histogram": dict(Counter(len(p["changed_fields"]) for p in pr)),
            "pairs_time_f1_percentiles": percentiles([p["time_f1_20ms"] for p in pr]),
        }
    report = {
        "schema": "beatfcn.p2_representation_audit.v1",
        "catalog": str(CATALOG.relative_to(ROOT)), "catalog_sha256": actual_sha,
        "script_sha256": digest(Path(__file__)), "train_only": True,
        "model_updates": 0, "new_paid_compute_usd": 0,
        "scope": "Selected real contrast pairs; not representative of the full difficulty distribution.",
        "limitations": [
            "Unique windows and pairs from the same song are correlated observations.",
            "Lower head count does not mean lower difficulty or lower burden.",
            "Non-nesting does not prove ambiguity after holding every condition fixed.",
            "Hiding other changed conditions is not a causal single-attribute intervention.",
            "Capacity maxima are observed requirements, not safe global hard caps.",
        ],
        "modes": modes, "pairs": pair_rows,
        "multiple_head_cases": sorted(cases, key=lambda r: (-len(r["head_offsets_us"]), r["chart_id"], r["c0_us"])),
        "windows": dict(mode_rows),
    }
    out = ROOT / "reports/p2_representation_audit.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = ["# P2 时间表示与真实配对审计", "", "仅本地读取冻结训练目录；0 次模型更新、0 新增付费计算。", "",
             f"输入 SHA256：`{actual_sha}`。统计未使用验证集。", "",
             "| 模式 | 唯一谱窗 | 歌曲 | 起音组 | 40ms 内多起音格 | 单格最大起音 | ±20ms 仍非包含的配对 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for mode, item in modes.items():
        cell = item["clocks"]["40000"]
        lines.append(f"| {mode} | {item['unique_chart_windows']} | {item['unique_audio']} | {item['distinct_heads']} | "
                     f"{cell['cells_with_multiple_distinct_heads']} | {cell['heads_per_cell_percentiles']['100']:g} | "
                     f"{item['pairs_not_nested_20ms']}/{item['pairs']} |")
    lines += ["", "## 解释", "",
              "1. 每 40ms 只允许一个起音会有确定的表示损失；缩成更细网格也须检查真实同格重数。观测最大值不能直接充当全库上限。",
              "2. 音符较少的真人谱并不普遍是音符较多谱的子集，单纯删点不能被当作完整的条件控制方案。这里按起音数排序，没有给难度重新命名。",
              "3. 配对往往同时改变多个条件。R09 单字段记忆成功可证明条件入口有表达能力，不能证明该字段的独立语义或因果作用。",
              "4. 空格、格内多个起音、同起音多音符、跨格 hold 都必须在新表示中保留；艺术 drag 不按独立声学起音筛掉。",
              "5. 这是刻意选择的训练反差样本，不代表全库分布，也不是独立泛化评测；不据此选择成功阈值。", "",
              "具体谱号、时间、类型、各网格容量和所有配对见同名 JSON。", "", "## 多起音实例", ""]
    for item in report["multiple_head_cases"][:5]:
        lines.append(f"- {item['chart_id']}，窗口 {item['c0_us']/1e6:g}s：相对起音 "
                     f"{[t/1000 for t in item['head_offsets_us']]} ms；类型 {item['types']}。")
    (ROOT / "reports/p2_representation_audit.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(out.relative_to(ROOT)), "modes": modes}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
