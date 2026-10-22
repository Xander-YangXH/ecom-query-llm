"""标签体系、各任务的指令与输出格式、校验规则。数据处理、训练数据转换、推理、评测都从这里取定义。

任务（每条记录的 rec["task"]）：
  full       完整的 query 理解：intent + attributes + rewrite（+ 给出标题时的 relevance）。标签来自合成数据。
  relevance  query–商品相关性四分类，与 ESCI Task 2 一致。标签来自 ESCI 人工标注。
标签定义与例子见 docs/LABELS.md。
"""
import json

INTENTS = ["product_search", "compare", "after_sales", "inquiry", "other"]
RELEVANCE = ["exact", "substitute", "complement", "irrelevant"]
ESCI_TO_RELEVANCE = {"E": "exact", "S": "substitute", "C": "complement", "I": "irrelevant"}
ATTR_KEYS = ["brand", "category", "model", "color", "size", "gender", "material", "price_min", "price_max"]
PRICE_KEYS = {"price_min", "price_max"}

TASK_FIELDS = {
    "full": ("intent", "attributes", "rewrite", "relevance"),
    "relevance": ("relevance",),
}

INSTRUCTIONS = {
    "full": (
        "You are the query understanding model of an e-commerce search engine. "
        "Given a user query (and optionally a product title), output a JSON object with the fields "
        "intent, attributes, rewrite, relevance. "
        f"intent is one of: {', '.join(INTENTS)}. "
        f"attributes maps keys from [{', '.join(ATTR_KEYS)}] to values copied from the query. "
        "rewrite is a normalized search string. "
        f"relevance is one of [{', '.join(RELEVANCE)}] when a product title is given, otherwise null. "
        "Output JSON only."
    ),
    "relevance": (
        "You are a relevance judge for e-commerce search. Given a user query and a product title, "
        "classify how well the product matches the query. "
        "exact: the product satisfies all requirements of the query. "
        "substitute: the product misses some requirement but could serve as a functional substitute. "
        "complement: the product does not match the query but could be used together with a matching product. "
        "irrelevant: the product is unrelated or fails a central requirement of the query. "
        'Output JSON only, in the form {"relevance": "<label>"}.'
    ),
}


def task_of(rec: dict) -> str:
    return rec.get("task", "full")


def instruction_for(task: str) -> str:
    return INSTRUCTIONS[task]


# 商品附加字段（E6，DEVLOG S16）：记录里的 rec["product"] = {"brand", "color", "bullets"}，缺失的键不出现。
PRODUCT_FIELDS = (("brand", "brand"), ("color", "color"), ("bullets", "product details"))


def format_input(query: str, title: str | None = None, product: dict | None = None) -> str:
    text = f"query: {query}"
    if title:
        text += f"\nproduct title: {title}"
    for key, label in PRODUCT_FIELDS if product else ():
        if product.get(key):
            text += f"\n{label}: {product[key]}"
    return text


def build_user_prompt(query: str, title: str | None = None, task: str = "full", product: dict | None = None) -> str:
    """推理时的完整用户消息。必须与 LLaMA-Factory 对 Alpaca 样本的拼接一致：
    instruction 与 input 用换行连接，放在同一个 user 轮次里（不使用 system），见 DEVLOG P1。"""
    return f"{instruction_for(task)}\n{format_input(query, title, product)}"


def format_output(label: dict, task: str = "full") -> str:
    ordered = {k: label.get(k) for k in TASK_FIELDS[task]}
    return json.dumps(ordered, ensure_ascii=False)


def parse_output(text: str, task: str = "full") -> dict | None:
    """把模型输出解析为 dict；不是合法 JSON、缺字段或类型不对时返回 None（评测时按全错计）。"""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or any(k not in obj for k in TASK_FIELDS[task]):
        return None
    if "attributes" in TASK_FIELDS[task] and not isinstance(obj["attributes"], dict):
        return None
    return obj


def _validate_full(rec: dict, label: dict) -> list[str]:
    errors = []
    query, title = rec["query"], rec.get("title")
    if label.get("intent") not in INTENTS:
        errors.append(f"intent not in label set: {label.get('intent')!r}")

    attrs = label.get("attributes")
    if not isinstance(attrs, dict):
        errors.append("attributes must be a dict")
        attrs = {}
    source = (query + " " + (title or "")).lower()
    for key, value in attrs.items():
        if key not in ATTR_KEYS:
            errors.append(f"unknown attribute key: {key!r}")
            continue
        value = str(value)
        if not value:
            errors.append(f"empty value for attribute {key}")
        elif key in PRICE_KEYS:
            digits = "".join(ch for ch in value if ch.isdigit())
            if not digits or digits not in source:
                errors.append(f"price {value!r} not found in query/title")
        elif value.lower() not in source:
            errors.append(f"attribute {key}={value!r} not found in query/title")

    rewrite = label.get("rewrite")
    if not isinstance(rewrite, str) or not rewrite.strip():
        errors.append("rewrite missing or empty")

    relevance = label.get("relevance")
    if title:
        if relevance not in RELEVANCE:
            errors.append(f"title given but relevance invalid: {relevance!r}")
    elif relevance is not None:
        errors.append("relevance must be null when no title is given")
    return errors


def _validate_relevance(rec: dict, label: dict) -> list[str]:
    errors = []
    if not rec.get("title"):
        errors.append("relevance task requires a product title")
    if label.get("relevance") not in RELEVANCE:
        errors.append(f"relevance not in label set: {label.get('relevance')!r}")
    return errors


def validate_record(rec: dict) -> list[str]:
    """返回错误列表，空列表表示通过。rec 为规范格式：id/task/query/title?/label/group?/meta?。"""
    query, label, task = rec.get("query"), rec.get("label"), task_of(rec)
    if task not in TASK_FIELDS:
        return [f"unknown task: {task!r}"]
    if not isinstance(query, str) or not query.strip():
        return ["query missing or empty"]
    if not isinstance(label, dict):
        return ["label missing"]
    if task == "relevance":
        return _validate_relevance(rec, label)
    return _validate_full(rec, label)
