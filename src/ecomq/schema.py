"""标签体系、输出 JSON 格式与校验规则。数据合成、校验、评测都从这里取定义。"""
import json

INTENTS = ["商品搜索", "比价", "售后", "咨询", "其他"]
RELEVANCE = ["精确", "部分", "不相关"]
ATTR_KEYS = ["品牌", "品类", "型号", "颜色", "规格", "性别", "材质", "价格上限", "价格下限"]
PRICE_KEYS = {"价格上限", "价格下限"}

INSTRUCTION = (
    "你是电商搜索的 Query 理解模型。根据用户 query（以及可选的商品标题），"
    "输出一个 JSON，字段为 intent、attributes、rewrite、relevance。"
    f"intent 取值：{'/'.join(INTENTS)}；"
    f"attributes 的键只能取：{'、'.join(ATTR_KEYS)}；"
    "rewrite 为规范化后的检索词；"
    f"relevance 仅在给出商品标题时判断（{'/'.join(RELEVANCE)}），否则为 null。只输出 JSON。"
)

OUTPUT_FIELDS = ("intent", "attributes", "rewrite", "relevance")


def format_input(query: str, title: str | None = None) -> str:
    text = f"query: {query}"
    if title:
        text += f"\n商品标题: {title}"
    return text


def build_user_prompt(query: str, title: str | None = None) -> str:
    """推理时的完整用户消息。必须与 LLaMA-Factory 对 Alpaca 样本的拼接一致：
    instruction 与 input 用换行连接，放在同一个 user 轮次里（不使用 system）。"""
    return f"{INSTRUCTION}\n{format_input(query, title)}"


def format_output(label: dict) -> str:
    ordered = {k: label.get(k) for k in OUTPUT_FIELDS}
    return json.dumps(ordered, ensure_ascii=False)


def parse_output(text: str) -> dict | None:
    """把模型输出解析为 dict；不是合法 JSON 或字段缺失则返回 None。"""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or any(k not in obj for k in OUTPUT_FIELDS):
        return None
    if not isinstance(obj["attributes"], dict):
        return None
    return obj


def validate_record(rec: dict) -> list[str]:
    """返回错误列表，空列表表示通过。rec 为规范格式：id/query/title?/label/meta?。"""
    errors = []
    query = rec.get("query")
    title = rec.get("title")
    label = rec.get("label")
    if not isinstance(query, str) or not query.strip():
        return ["query 缺失或为空"]
    if not isinstance(label, dict):
        return ["label 缺失"]

    if label.get("intent") not in INTENTS:
        errors.append(f"intent 不在标签集合内: {label.get('intent')!r}")

    attrs = label.get("attributes")
    if not isinstance(attrs, dict):
        errors.append("attributes 必须是 dict")
        attrs = {}
    source = query + (title or "")
    for key, value in attrs.items():
        if key not in ATTR_KEYS:
            errors.append(f"未知属性键: {key!r}")
            continue
        value = str(value)
        if not value:
            errors.append(f"属性 {key} 的值为空")
        elif key in PRICE_KEYS:
            digits = "".join(ch for ch in value if ch.isdigit())
            if not digits or digits not in source:
                errors.append(f"价格 {value!r} 未在 query/标题中出现")
        elif value not in source:
            errors.append(f"属性 {key}={value!r} 未在 query/标题中出现")

    rewrite = label.get("rewrite")
    if not isinstance(rewrite, str) or not rewrite.strip():
        errors.append("rewrite 缺失或为空")

    relevance = label.get("relevance")
    if title:
        if relevance not in RELEVANCE:
            errors.append(f"给出了标题但 relevance 非法: {relevance!r}")
    elif relevance is not None:
        errors.append("没有标题时 relevance 必须为 null")
    return errors
