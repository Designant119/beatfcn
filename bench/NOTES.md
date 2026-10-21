## 14:05 interim (executor)
- logs: /workspace/bench/v22run/train.log, /workspace/bench/v23b/train.log
- v22b: ~900s/epoch (crop3000, CPU 6 threads), ep1-3 strict F1 0.221/0.209/0.210 (eval 4 songs)
- v23b: ~82s/epoch full songs (2 threads), ep17 strict 0.236 loose50ms 0.529 (eval 38 songs); occasional stalls (CPU contention)
- bench: train step T=3000 v22 3.40s vs v23 0.027s (127x), GFLOPs 69 vs 2.46 (28x), RSS 2459 vs 371MB; v22 OOM at T=6000
