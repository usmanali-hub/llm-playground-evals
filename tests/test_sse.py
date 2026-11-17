from llm_playground.sse import parse_sse


def test_basic_events_and_default_name() -> None:
    lines = ["event: a\n", "data: 1\n", "\n", "data: 2\n", "\n"]
    assert list(parse_sse(lines)) == [("a", "1"), ("message", "2")]


def test_multiline_data_comments_and_crlf() -> None:
    lines = [": keep-alive\r\n", "event: x\r\n", "data: one\r\n", "data:two\r\n", "\r\n"]
    assert list(parse_sse(lines)) == [("x", "one\ntwo")]


def test_trailing_event_without_blank_line_is_delivered() -> None:
    assert list(parse_sse(["data: tail"])) == [("message", "tail")]


def test_blank_lines_without_data_emit_nothing_and_reset_event_name() -> None:
    assert list(parse_sse(["event: lonely\n", "\n", "data: x\n", "\n"])) == [("message", "x")]


def test_only_one_leading_space_is_removed() -> None:
    assert list(parse_sse(["data:   spaced\n", "\n"])) == [("message", "  spaced")]
