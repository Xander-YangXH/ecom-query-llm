# 标签体系（英文场景）

代码中的唯一定义在 `src/ecomq/schema.py`；本文件解释每个标签的含义、边界与数据来源。

## 任务

| task | 输入 | 输出 JSON | 标签来源 |
|---|---|---|---|
| `relevance` | query + 商品标题 | `{"relevance": ...}` | ESCI 人工标注（E/S/C/I） |
| `full` | query（可选商品标题） | `{"intent", "attributes", "rewrite", "relevance"}` | 强模型合成 + 规则校验 + 人工抽检 |

把真实标注和合成标注拆成不同任务，是为了让每个指标都能说清“标签从哪来”：ESCI 上的分数可以和公开基线对比，合成数据上的分数只代表与合成规则的一致性。

## relevance（与 ESCI Task 2 一致）

| 标签 | ESCI | 定义 | 例子（query: `hdmi cable 10 ft`） |
|---|---|---|---|
| `exact` | E | 满足 query 的全部要求 | 10 英尺 HDMI 线 |
| `substitute` | S | 缺少某个要求，但可作为功能替代品 | 6 英尺 HDMI 线 |
| `complement` | C | 不满足 query，但可与满足的商品搭配使用 | HDMI 线扎带 |
| `irrelevant` | I | 无关，或不满足 query 的核心要求 | 绿茶 |

指标：准确率（单标签多分类下等于 micro-F1）、宏 F1、各类 P/R/F1、混淆矩阵。

## intent（full 任务）

| 标签 | 定义 | 例子 |
|---|---|---|
| `product_search` | 想找到并购买某类商品 | `nike mens running shoes under 100` |
| `compare` | 比较两个及以上商品或品牌（价格、质量、参数都算） | `samsung vs lg tv which is better` |
| `after_sales` | 退换货、维修、物流、订单问题 | `how do i return shoes that fell apart` |
| `inquiry` | 购买前的知识性问题，不指向具体商品 | `what length apple charging cable do i need` |
| `other` | 无明确购物意图 | `just browsing` |

`compare` 替代了早期中文版的「比价」：冒烟测试中发现「哪个更好」这类质量比较落在「比价」与「咨询」之间，边界不清（DEVLOG P8）。

## attributes（full 任务）

键：`brand, category, model, color, size, gender, material, price_min, price_max`。

- 值必须从 query（或给出的标题）中**原样复制**，大小写不敏感；价格只要求数字出现。校验不通过的合成样本直接丢弃，用来拦截幻觉属性。
- `size` 涵盖容量、尺寸、存储等规格（`256gb`、`32 oz`、`10 ft`）。

## rewrite（full 任务）

规范化后的检索串：纠正拼写、展开缩写、去掉口语成分；否定条件用 `-词` 表示（`women summer t-shirt -black`）。
