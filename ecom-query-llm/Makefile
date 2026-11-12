.PHONY: test sample-check smoke-data clean

# 未 pip install -e . 时也能直接运行
export PYTHONPATH := src
OUT := build/sample

test:
	python -m pytest -q

# 用仓库内的小样本跑一遍：校验 -> 划分 -> 转 Alpaca 格式
sample-check:
	python -m ecomq.data.validate --input data_sample/seed_sample.jsonl --output $(OUT)/valid.jsonl
	python -m ecomq.data.split --input $(OUT)/valid.jsonl --out-dir $(OUT)/splits
	python -m ecomq.data.to_alpaca --split-dir $(OUT)/splits --out-dir $(OUT)/llamafactory

# 生成冒烟测试数据到 $DATA_ROOT（val/test 与 train 重叠，指标无意义）
smoke-data:
	python -m ecomq.data.make_smoke --n 400

clean:
	rm -rf build
