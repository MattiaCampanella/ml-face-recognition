| Run Date        | Sampler   | Embedding | Margin         | mAP@1 | mAP@5| mAP@10| learning rate | Mining      |
|-----------------|-----------|-----------|----------------|-------|------|-------|---------------|-------------|
| 2026-05-13      | PK(24, 4) | 128       | 0.2 softmargin | .70*  | .60* | .54*  | 0.0003        | easy to hard| 🤡
| 2026-05-14      | PK(24, 4) | 512       | 0.2 softmargin | .69*  | .58* | .51*  | 0.0003        | easy to hard| 🤡
| 2026-05-14      | PK(24, 4) | 512       | 0.2 softmargin | .68*  | .57* | .50*  | 0.0003        | easy        | 🤡
| 2026-05-14      | PK(32, 4) | 512       | 0.1 softmargin | .72   | .62  | .56   | 0.0003        | easy to semi| 🌖
| 2026-05-15      | PK(32, 4) | 512       | 0.1 softmargin | .86   | .80  | .75   | 0.0003        | hard        | ⭐
| 2026-05-15      | PK(32, 4) | 512       | 0.2 softmargin | .82   | .74  | .69   | 0.0003        | semi to hard| ⬇️
| 2026-05-15      | PK(32, 4) | 512       | 0.1 softmargin | .87   | .81  | .77   | 0.0003        | hard        | only loss norm⬆️
| 2026-05-16      | PK(42, 4) | 512       | 0.1 softmargin | .86   | .80  | .75   | 0.0003        | hard        | 🌕
| 2026-05-15      | PK(32, 4) | 512       | 0.1 hingemargin| .25   | .16  | .12   | 0.0003        | hard        | 🤢

\* Overfitting observed

arcface:
🎯 mAP@1:      0.821421
🎯 mAP@5:      0.746180
🎯 mAP@10:     0.690484