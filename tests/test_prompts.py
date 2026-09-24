from app.rag.prompts import is_refusal, parse_citations


def test_citations_ignore_part_and_point_markers():
    assert parse_citations("x [Article 49(2)(1)] y [Հոդված 49(3)] z [Article 45(2)]") == ["49", "45"]


def test_citations_multiple_and_decimal_articles():
    assert parse_citations("[Article 12, 13] [Article 17.1] [Art. 3] [Հոդ. 60]") == ["12", "13", "17.1", "3", "60"]


def test_refusal_detection_both_languages():
    assert is_refusal("The provided articles of the Law do not address this question.")
    assert is_refusal("Օրենքի տրամադրված հոդվածները չեն անդրադառնում այս հարցին։")
    assert not is_refusal("Yes, within 30 days [Article 57].")


def test_citations_ignore_unbracketed_points():
    assert parse_citations("[Հոդված 49(2)1] [Հոդված 57(2)2)] [Article 12(2)(3-4)]") == ["49", "57", "12"]
