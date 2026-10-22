# E↔S 标注审计（DEVLOG S17）

| 文件 | 内容 | 判断前能不能看 |
|---|---|---|
| `blind_sheet.xlsx` | 盲评表：「说明」「标注」「汇总」三个工作表。在「标注」的黄色列里用下拉菜单选 E / S / C / I / U | 可以 |
| `blind.csv` | 同样的 120 条，纯文本版本 | 可以 |
| `key.json` | 答案：每条的分组、官方标注、6 个模型的投票 | **判断完成前不要打开** |
| `labels_claude.csv` | Claude 的盲评结果（只看了 blind.csv） | 建议判断完成后再看，避免受影响 |
| `score_claude.json` | Claude 盲评的评分结果 | 同上 |

## 怎么参与

1. 复制一份 `blind_sheet.xlsx`，独立完成「标注」工作表（约 30 到 40 分钟）。
2. 计算结果：

```bash
PYTHONPATH=src python -m ecomq.analysis.es_audit score --key analysis/es_audit/key.json --labels <你的文件>.xlsx
```

3. 把结果文件命名为 `score_<名字>.json` 提交。多人结果可以互相比较，计算判断人之间的一致率。

## 抽样方法

- 系统性错误：测试集（2,500 个 query）中，6 个模型（标题版与丰富输入版各 3 个种子，不加偏置）至少 5 个都判成同一个错误标签的 E↔S 样本。两个方向各随机抽 50 条，每个 query 最多 1 条。
- 对照组：6 个模型都判对的 exact、substitute 各 10 条，用来衡量判断人自身与官方标注的一致程度。
- 120 条随机打乱，表中不显示标注和模型预测。
- 重新生成：`python -m ecomq.analysis.es_audit sample ...`（种子 0，参数见 DEVLOG S17）。
