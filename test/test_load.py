import re
import textwrap
from typing import Final

import pytest

from rfc5234 import Opts, ParsingStrategy, ParseEventKind, InlineStrategy, Parser, Grammar
from rfc5234.abnf import read_abnf

opts = Opts(
    use_regex=True,
    strategy=ParsingStrategy.FIRST_MATCH,
    events=ParseEventKind.VALUE,
    events_filter=re.compile("^.*[a-z].*$"),
    inline_strategy=InlineStrategy.AGGRESSIVE,
)

grammars: Final[tuple[tuple[str, str | None], ...]] = (
    ("abnf", "rulelist"),
    ("abnf_original", "rulelist"),
    ("json", "JSON-text"),
    ("simple_expr", None),
    ("toml", "toml"),
)


@pytest.mark.parametrize(("name", "entrypoint"), grammars)
def test_load(name: str, entrypoint: str | None):
    core = read_abnf("core")
    text = read_abnf(name)

    p = Parser()
    p.load(core)
    p.load(text)

    g = p.compile(opts=opts)
    if entrypoint:
        assert g.entrypoint == entrypoint


@pytest.fixture(scope="session")
def abnf_grammar() -> Grammar:
    core = read_abnf("core")
    text = read_abnf("abnf")

    p = Parser()
    p.load(core)
    p.load(text)

    return p.compile(opts=opts)


@pytest.mark.parametrize("name", [n for n, *_ in grammars])
def test_load_generic(abnf_grammar: Grammar, name: str):
    text = read_abnf(name)
    abnf_grammar.parse(text, opts=opts)


def test_left_recursion_detection():
    core = read_abnf("core")

    p = Parser()
    p.load(core)
    p.load(textwrap.dedent("""
        r1 = r2 "a"
        r2 = r3 "b"
        r3 = r1 / "c"
    """))

    with pytest.raises(ValueError) as err:
        p.compile(opts=opts)
    assert err.value.args[0].startswith("Left recursion detected:")
