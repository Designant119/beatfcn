"""Explain the rejected R05 outputs without further model inference."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from beatfcn.train.control_panel import REQUEST_BANDS, response_gate

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = {"candidate": ROOT / "reports/r05_evaluate.json",
             "control": ROOT / "reports/r03_window_raw/r03-window-all.benchmark.json"}
    raw = {name: json.loads(path.read_text()) for name, path in paths.items()}
    review = json.loads((ROOT / "reports/r05_review.json").read_text())
    assert review["comparison"]["short_window_gate_passed"] is False
    factual = {name: {r["case_id"]: r for r in value["records"] if r["field"] is None}
               for name, value in raw.items()}
    assert factual["candidate"].keys() == factual["control"].keys()
    cases = []
    for row in review["results"]["candidate"]["factual"]:
        key = row["case_id"]
        before, after = factual["control"][key], factual["candidate"][key]
        assert before["truth_events"] == after["truth_events"]
        times = {name: sorted({e["onset_us"] for e in source["generated_events"]})
                 for name, source in (("before", before), ("after", after))}
        cases.append({k: row[k] for k in (
            "case_id", "chart_id", "mode", "first_onset_error_ms", "empty",
            "full_note_f1", "timing_point_f1", "prediction_count", "truth_count",
            "predicted_types", "truth_types", "reference_edit_saved_fraction",
            "timing_edit_saved_fraction")})
        cases[-1].update(first_onset_unchanged=times["before"][:1] == times["after"][:1],
                         timing_sequence_unchanged=times["before"] == times["after"])
    grouped = defaultdict(list)
    for row in raw["candidate"]["records"]:
        if row["field"] is not None:
            grouped[(row["case_id"], row["field"])].append(row)
    triplets = []
    for (case_id, field), rows in grouped.items():
        rows.sort(key=lambda r: REQUEST_BANDS[field].index(r["requested"]))
        gate = response_gate(rows, field)
        signatures = [json.dumps(r["generated_events"], sort_keys=True) for r in rows]
        triplets.append(dict(case_id=case_id, field=field, **gate,
                             identical_events=len(set(signatures)) == 1,
                             predicted_types=[r["predicted_types"] for r in rows]))
    report = dict(schema="beatfcn.r05_bad_cases.v1", cases=cases, condition_triplets=triplets,
        measured_triplets=len(triplets), identical_triplets=sum(r["identical_events"] for r in triplets),
        strictly_monotonic_triplets=sum(r["strictly_monotonic"] for r in triplets),
        all_in_band_triplets=sum(r["all_in_band"] for r in triplets),
        passed_triplets=sum(r["passed"] for r in triplets),
        unmeasured_requests=sum(r["remaining"] * 3 for r in raw["candidate"]["field_decisions"].values()),
        first_onset_unchanged_windows=sum(r["first_onset_unchanged"] for r in cases),
        positive_full_note_edit_proxy_windows=sum(r["reference_edit_saved_fraction"] > 0 for r in cases),
        new_inference_calls=0,
        source_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()})
    output = ROOT / "reports/r05_bad_cases.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    lines = ["# R05 错误分布", "", "只分析保存的事件，没有追加生成。",
        f"{len(triplets)} 组三档请求中，{report['identical_triplets']} 组逐事件完全相同；"
        f"严格单调 {report['strictly_monotonic_triplets']} 组，全部在带 {report['all_in_band_triplets']} 组，"
        f"两项同时通过 {report['passed_triplets']} 组。按预注册提前否决，剩余 {report['unmeasured_requests']} 个请求未测。", "",
        "| 片段 | 首点误差 ms | 输出/人谱音符 | 完整音符 F1 | 时间点 F1 | 输出类型 |",
        "|---|---:|---:|---:|---:|---|"]
    for r in cases:
        error = "空窗" if r["empty"] else f"{r['first_onset_error_ms']:.0f}"
        lines.append(f"| {r['case_id']} | {error} | {r['prediction_count']}/{r['truth_count']} | "
                     f"{r['full_note_f1']:.4f} | {r['timing_point_f1']:.4f} | {r['predicted_types']} |")
    lines += ["", "| 片段 / 条件 | 低、中、高档实际值 | 逐事件相同 |", "|---|---|---|"]
    for r in triplets:
        values = ", ".join("未定义" if v is None else f"{v:.3f}" for v in r["values"])
        lines.append(f"| {r['case_id']} / {r['field']} | {values} | {r['identical_events']} |")
    lines += ["", f"{report['first_onset_unchanged_windows']}/{len(cases)} 个窗口首点或空窗状态与对照相同；"
              f"完整音符编辑代理有正收益的片段为 {report['positive_full_note_edit_proxy_windows']}/{len(cases)}。",
              "该面板已足以否决本轮干预，但不能估计整库成功率。未把真历史诊断或人谱条件用于生成。"]
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("cases", "condition_triplets", "source_sha256")}))


if __name__ == "__main__":
    main()
