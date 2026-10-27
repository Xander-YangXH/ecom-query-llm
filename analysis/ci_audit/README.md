# complement / irrelevant 标注审计（DEVLOG S24）

| 文件 | 内容 | 判断前能不能看 |
|---|---|---|
| `blind_sheet.xlsx` | 盲评表：「说明」「标注」「汇总」三个工作表。在「标注」的黄色列里用下拉菜单选 E / S / C / I / U | 可以 |
| `blind.csv` | 同样的 130 条，纯文本版本 | 可以 |
| `key.json` | 答案：每条的分组、官方标注、6 个模型的投票 | **判断完成前不要打开** |
| `labels_claude.csv` | Claude 的盲评结果（只看了 blind.csv） | 建议判断完成后再看 |
| `score_claude.json` | Claude 盲评的评分结果 | 同上 |

## 怎么参与

1. 复制一份 `blind_sheet.xlsx`，独立完成「标注」工作表（约 40 分钟）。
2. 计算结果（按粗粒度类别比较：你的 E 与 S 都算作“相关”）：

```bash
PYTHONPATH=src python -m ecomq.analysis.es_audit score --key analysis/ci_audit/key.json --labels <你的文件>.xlsx
```

## 抽样方法

- 6 个模型（标题版与丰富输入版各 3 个种子，不加偏置）至少 5 个判成同一个粗粒度错误类别的样本：
  - C2R：标注 complement，模型判为相关（exact / substitute），抽 30 条；
  - I2R：标注 irrelevant，模型判为相关，抽 30 条；
  - R2I：标注相关，模型判为 irrelevant，抽 30 条；
  - CI：complement 与 irrelevant 互判，抽 20 条；
  - 每个 query 最多 1 条。
- 对照组：6 个模型都判对的 complement、irrelevant 各 10 条。
- 130 条随机打乱，表中不显示标注和模型预测。
- 重新生成：

```bash
python -m ecomq.analysis.es_audit sample --mode ci --k 5 --n-err 30 --n-ci 20 --n-ctrl 10 --seed 0 ...
python -m ecomq.analysis.es_audit sheet --blind analysis/ci_audit/blind.csv --key analysis/ci_audit/key.json --title ... --output analysis/ci_audit/blind_sheet.xlsx
```
