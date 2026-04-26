"""Tests for regex_builder.infer."""

from __future__ import annotations

import re

import pytest

from regex_builder import infer, infer_from_example, infer_from_examples


def _matches(pattern: str, text: str) -> bool:
    return bool(re.fullmatch(pattern, text))


# ---------------------------------------------------------------------------
# Type dispatch
# ---------------------------------------------------------------------------

class TestInferTypeDispatch:
    def test_str_delegates_to_single(self):
        assert infer("hello") == infer_from_example("hello")

    def test_list_delegates_to_multi(self):
        examples = ["010-1234-5678", "01012345678"]
        assert infer(examples) == infer_from_examples(examples)

    def test_invalid_type_raises_type_error(self):
        with pytest.raises(TypeError):
            infer(12345)  # type: ignore[arg-type]

    def test_empty_list_raises_value_error(self):
        with pytest.raises(ValueError):
            infer([])


# ---------------------------------------------------------------------------
# Phone numbers
# ---------------------------------------------------------------------------

class TestPhoneNumbers:
    def test_hyphenated_pattern_shape(self):
        pattern = infer_from_example("010-1234-5678")
        assert r"\d{3}" in pattern
        assert r"\-?" in pattern

    def test_hyphenated_matches_example(self):
        assert _matches(infer_from_example("010-1234-5678"), "010-1234-5678")

    def test_hyphenated_matches_no_hyphen_variant(self):
        assert _matches(infer_from_example("010-1234-5678"), "01012345678")

    def test_no_hyphen_matches_example(self):
        assert _matches(infer_from_example("01012345678"), "01012345678")

    def test_multi_example_or_combines(self):
        pattern = infer_from_examples(["010-1234-5678", "01012345678"])
        assert pattern.startswith("(?:") and pattern.endswith(")")
        assert "|" in pattern

    def test_multi_example_matches_both(self):
        pattern = infer_from_examples(["010-1234-5678", "01012345678"])
        assert _matches(pattern, "010-1234-5678")
        assert _matches(pattern, "01012345678")

    def test_does_not_match_short_number(self):
        assert not _matches(infer_from_example("010-1234-5678"), "010-123-567")


# ---------------------------------------------------------------------------
# Korean RRN (주민등록번호)
# ---------------------------------------------------------------------------

class TestKoreanRRN:
    def test_pattern_shape(self):
        pattern = infer_from_example("971021-2333333")
        assert r"\d{6}" in pattern
        assert r"\-?" in pattern
        assert r"\d{7}" in pattern

    def test_matches_example(self):
        assert _matches(infer_from_example("971021-2333333"), "971021-2333333")

    def test_matches_no_hyphen(self):
        assert _matches(infer_from_example("971021-2333333"), "9710212333333")

    def test_does_not_match_wrong_length(self):
        assert not _matches(infer_from_example("971021-2333333"), "97102-233333")


# ---------------------------------------------------------------------------
# Mixed alphanumeric (API keys)
# ---------------------------------------------------------------------------

class TestMixedAlphanumeric:
    def test_short_mixed_classified_correctly(self):
        assert infer_from_example("AIzaSy9xZ") == "[A-Za-z0-9]{9}"

    def test_mixed_matches_example(self):
        assert _matches(infer_from_example("AIzaSy9xZ"), "AIzaSy9xZ")

    def test_mixed_matches_any_alphanumeric_same_length(self):
        pattern = infer_from_example("AIzaSy9xZ")
        assert _matches(pattern, "Q3A7kBxZ1")
        assert _matches(pattern, "AAAAAAAAA")
        assert _matches(pattern, "123456789")

    def test_full_google_api_key(self):
        key = "AIzaSyCZiaXqXxaz6e9khiWaLsI5jykwkeu9Zvk"
        pattern = infer_from_example(key)
        assert pattern == f"[A-Za-z0-9]{{{len(key)}}}"
        assert _matches(pattern, key)

    def test_two_api_keys_or_combined(self):
        keys = [
            "AIzaSyCZiaXqXxaz6e9khiWaLsI5jykwkeu9Zvk",
            "AIzaSyD0fpELF01EbZSErlqqBYY3br9870sx_Q8",
        ]
        pattern = infer_from_examples(keys)
        assert pattern.startswith("(?:")
        for key in keys:
            assert re.search(pattern, key), f"pattern did not match {key!r}"

    def test_pure_digits_not_classified_as_mixed(self):
        assert infer_from_example("12345") == r"\d{5}"

    def test_pure_letters_not_classified_as_mixed(self):
        assert infer_from_example("hello") == "[A-Za-z]{5}"


# ---------------------------------------------------------------------------
# Korean hangul
# ---------------------------------------------------------------------------

class TestKoreanHangul:
    def test_two_char_name(self):
        pattern = infer_from_example("홍길")
        assert pattern == "[가-힣]{2}"
        assert _matches(pattern, "홍길")
        assert _matches(pattern, "김이")

    def test_three_char_name(self):
        assert infer_from_example("홍길동") == "[가-힣]{3}"

    def test_hangul_does_not_match_latin(self):
        assert not _matches(infer_from_example("홍길동"), "ABC")

    def test_hangul_followed_by_digit(self):
        pattern = infer_from_example("홍123")
        assert "[가-힣]{1}" in pattern
        assert r"\d{3}" in pattern
        assert _matches(pattern, "홍123")
        assert _matches(pattern, "동456")


# ---------------------------------------------------------------------------
# Email-like strings
# ---------------------------------------------------------------------------

class TestEmailLike:
    def test_simple_email_matches(self):
        pattern = infer_from_example("user@example.com")
        assert _matches(pattern, "user@example.com")

    def test_at_sign_is_literal(self):
        # re.escape('@') returns '@' in Python 3.7+ (not a regex metachar)
        assert "@" in infer_from_example("user@example.com")

    def test_dot_is_literal(self):
        assert r"\." in infer_from_example("user@example.com")

    def test_email_with_digits_in_domain(self):
        pattern = infer_from_example("abc1@mail2.org")
        assert _matches(pattern, "abc1@mail2.org")

    def test_plus_addressing(self):
        pattern = infer_from_example("user+tag@host.com")
        assert r"\+" in pattern
        assert _matches(pattern, "user+tag@host.com")


# ---------------------------------------------------------------------------
# Special characters as literals
# ---------------------------------------------------------------------------

class TestSpecialCharsLiteral:
    @pytest.mark.parametrize("char", [".", "@", "_", "/", ":", "+", "="])
    def test_special_char_is_escaped(self, char: str):
        example = f"a{char}b"
        pattern = infer_from_example(example)
        assert re.escape(char) in pattern
        assert _matches(pattern, example)

    def test_url_like_string(self):
        pattern = infer_from_example("https://example.com")
        assert _matches(pattern, "https://example.com")
        # ':' and '/' are not regex metachars — re.escape returns them bare
        assert ":" in pattern
        assert "/" in pattern

    def test_base64_equals_sign(self):
        pattern = infer_from_example("abc=")
        # '=' is not a regex metachar — re.escape returns it bare
        assert "=" in pattern


# ---------------------------------------------------------------------------
# Optional separators
# ---------------------------------------------------------------------------

class TestOptionalSeparators:
    def test_hyphen_becomes_optional(self):
        pattern = infer_from_example("12-34")
        assert r"\-?" in pattern
        assert _matches(pattern, "12-34")
        assert _matches(pattern, "1234")

    def test_space_becomes_optional(self):
        pattern = infer_from_example("hello world")
        assert r"\ ?" in pattern
        assert _matches(pattern, "hello world")
        assert _matches(pattern, "helloworld")


# ---------------------------------------------------------------------------
# Multi-example OR
# ---------------------------------------------------------------------------

class TestMultiExampleOR:
    def test_single_element_no_wrapping(self):
        assert not infer_from_examples(["abc"]).startswith("(?:")

    def test_two_examples_wrapped(self):
        result = infer_from_examples(["abc", "123"])
        assert result.startswith("(?:") and result.endswith(")")

    def test_or_matches_first(self):
        assert re.fullmatch(infer_from_examples(["abc", "123"]), "abc")

    def test_or_matches_second(self):
        assert re.fullmatch(infer_from_examples(["abc", "123"]), "123")

    def test_three_examples(self):
        pattern = infer_from_examples(["abc", "123", "홍길동"])
        assert re.fullmatch(pattern, "abc")
        assert re.fullmatch(pattern, "123")
        assert re.fullmatch(pattern, "홍길동")

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            infer_from_examples([])


# ---------------------------------------------------------------------------
# All generated patterns must compile without error
# ---------------------------------------------------------------------------

class TestPatternsAlwaysCompile:
    @pytest.mark.parametrize("example", [
        "010-1234-5678",
        "01012345678",
        "971021-2333333",
        "AIzaSy9xZ",
        "AIzaSyCZiaXqXxaz6e9khiWaLsI5jykwkeu9Zvk",
        "user@example.com",
        "홍길동",
        "https://api.example.com:8080/v1/resource",
        "hello world",
        "MixedABC123def",
        "123-45-67890",
        "1234-5678-9012-3456",
    ])
    def test_pattern_compiles(self, example: str):
        pattern = infer_from_example(example)
        try:
            re.compile(pattern)
        except re.error as exc:
            pytest.fail(f"Pattern {pattern!r} for {example!r} failed to compile: {exc}")
