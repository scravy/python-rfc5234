import re

import pytest

from rfc5234 import Opts, ParsingStrategy, ParseEventKind, InlineStrategy, Parser
from rfc5234.abnf import read_abnf


@pytest.mark.parametrize(
    "name", [
        "abnf",
        "abnf_original",
        "json",
        "simple_expr",
        "toml",
    ]
)
def test_load(name: str):
    opts = Opts(
        use_regex=True,
        strategy=ParsingStrategy.FIRST_MATCH,
        events=ParseEventKind.VALUE,
        events_filter=re.compile("^.*[a-z].*$"),
        inline_strategy=InlineStrategy.AGGRESSIVE,
    )
    core = read_abnf("core")
    text = read_abnf(name)

    p = Parser()
    p.load(core)
    p.load(text)

    g = p.compile(opts=opts)

    print("=" * 80)
    print(g.entrypoint)
    for rule in g:
        print(rule)
        print(" ", g[rule].expr)
        print(" ", g[rule].flags)
        for action in g[rule].actions:
            print(" ", "-", action)
