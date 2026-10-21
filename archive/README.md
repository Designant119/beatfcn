# archive/ — 历史文件，不要使用

- `split_full128.json_DO_NOT_USE`、`split_full128_clean.json_DO_NOT_USE`：旧版划分，有同曲泄漏。**唯一正确的划分是根目录 `split_full128_final.json`。**
- `fix_test_split.py`：由旧 split 生成 clean 版的一次性脚本（已完成）。
- `fix_split_final.py`：生成 final 版的一次性脚本，**会原地改写 split_full128_final.json，不要再运行**。
- `test_pollution_report.json`：旧版 test 污染报告。
