import re
import textwrap
from typing import Final

import pytest

from rfc5234 import (
    Opts,
    ParsingStrategy,
    ParseEventKind,
    InlineStrategy,
    Parser,
    Grammar,
)
from rfc5234.abnf import read_abnf

_OPTS: Final[Opts] = Opts(
    use_regex=True,
    strategy=ParsingStrategy.FIRST_MATCH,
    events=ParseEventKind.VALUE,
    events_filter=re.compile("^.*[a-z].*$"),
    inline_strategy=InlineStrategy.AGGRESSIVE,
)

# some well known grammars expressed in ABNF:
# - abnf_original is the ABNF for ABNF as given by RFC 5234
#   as per https://www.rfc-editor.org/rfc/rfc5234#section-4
# - abnf is an equivalent but slighthly reordered version to avoid ambiguous matches when using FIRST_MATCH
#   it's also annotated with @inline annotations
# - json as per https://www.rfc-editor.org/rfc/rfc8259
# - toml from https://github.com/toml-lang/toml/blob/d0c77ee8a76f506c8fddbec6d5eef282f96efa90/toml.abnf
grammars: Final[tuple[tuple[str, str | None], ...]] = (
    ("abnf", "rulelist"),
    ("abnf_original", "rulelist"),
    ("json", "JSON-text"),
    ("simple_expr", None),
    ("toml", "toml"),
)


@pytest.mark.parametrize(("name", "entrypoint"), grammars)
def test_load(name: str, entrypoint: str | None):
    """
    Check loading various ABNF grmamars.  Loads and compiles ABNF grammars.
    Always loads the core ABNF specifications https://www.rfc-editor.org/rfc/rfc5234#appendix-B
    The entrypoint is automatically inferred from the rule dependencies in the grammar,
    which is why we check for whether it's been correctly inferred.

    :param name: Which ABNF grammar to load
    :param entrypoint: Which rule to expect as the entrypoint
    """
    core = read_abnf("core")
    text = read_abnf(name)

    p = Parser()
    p.load(core)
    p.load(text)

    g = p.compile(opts=_OPTS)
    if entrypoint:
        assert g.entrypoint == entrypoint


@pytest.mark.parametrize("inline_strategy", InlineStrategy)
@pytest.mark.parametrize("parsing_strategy", ParsingStrategy)
@pytest.mark.parametrize("use_regex", [True, False])
@pytest.mark.parametrize("name", [n for n, *_ in grammars])
def test_load_generic(
    inline_strategy: InlineStrategy,
    parsing_strategy: ParsingStrategy,
    use_regex: bool,
    name: str,
):
    """
    Checks loading various ABNF grammars via the ABNF grammar from RFC 5234 loaded as such a grammar.
    The ABNF grammar we use is slightly modified from the original grammer – it accepts `[CR] LF` as newline
    (instead of `CR LF`), and it reorders one particular alternative (`"=/" / "="` instead of `"=" / "=/"`
    in the definition of `defined-as`), which makes continued rules work when using ParsingStrategy.FIRST_MATCH.
    """
    core = read_abnf("core")
    text = read_abnf("abnf")

    p = Parser()
    p.load(core)
    p.load(text)

    opts = _OPTS(inline_strategy=inline_strategy, strategy=parsing_strategy, use_regex=use_regex)
    abnf_grammar: Grammar = p.compile(opts=opts)

    text = read_abnf(name)
    last = None
    for ev in abnf_grammar.parse(text, opts=opts):
        last = ev
    assert last is not None
    assert last.rule == "rulelist"
    assert last.value == text


def test_left_recursion_detection():
    """
    Checks that left recursion detection works on a simple example.
    """
    core = read_abnf("core")

    p = Parser()
    p.load(core)
    p.load(
        textwrap.dedent("""
        r1 = r2 "a"
        r2 = r3 "b"
        r3 = r1 / "c"
    """)
    )

    with pytest.raises(ValueError) as err:
        p.compile(opts=_OPTS)
    assert err.value.args[0].startswith("Left recursion detected:")


def test_valid_mutually_recursive_rules():
    p = Parser()
    p.load(
        textwrap.dedent("""
            foobar = foo / bar
            foo = "foo" [bar]
            bar = "bar" [foo]
        """)
    )
    p.compile(opts=_OPTS)


def test_something():
    core = read_abnf("core")

    p = Parser()
    p.load(core)
    p.load(
        textwrap.dedent("""
            foobar = foo / bar
            foo = "foo" [bar]
            bar = "bar" [foo]
        """)
    )
    p.compile(opts=_OPTS)
