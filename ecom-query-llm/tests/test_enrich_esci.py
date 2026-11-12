from ecomq.data.enrich_esci import clean_bullets, coverage, enrich, product_fields


def test_clean_bullets_joins_and_truncates():
    assert clean_bullets("A  first\n\n second \n") == "A first | second"
    assert clean_bullets(None) == "" and clean_bullets("\n\n") == ""
    long = clean_bullets("\n".join(["word " * 20] * 10), max_chars=50)
    assert len(long) <= 54 and long.endswith("...")


def test_product_fields_drop_missing_values():
    assert product_fields("Sony", None, "") == {"brand": "Sony"}
    assert product_fields("nan", "  ", None) == {}
    assert product_fields(" Apple ", "Space  Gray", "a\nb") == {"brand": "Apple", "color": "Space Gray", "bullets": "a | b"}


def test_enrich_keeps_ids_and_labels():
    recs = [{"id": "1", "query": "q", "title": "t", "label": {"relevance": "exact"}, "meta": {"product_id": "P1"}},
            {"id": "2", "query": "q", "title": "t", "label": {"relevance": "irrelevant"}, "meta": {"product_id": "P2"}}]
    out = enrich(recs, {"P1": {"brand": "Sony"}})
    assert out[0]["product"] == {"brand": "Sony"} and "product" not in out[1]
    assert coverage(out) == {"n": 2, "brand": 0.5, "color": 0.0, "bullets": 0.0, "any": 0.5}
