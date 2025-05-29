from pathlib import Path

from .rfc5234 import Parser, Opts, ParsingStrategy, InlineStrategy


def main():
    p = Parser()
    p.load_file(Path(__file__).parent / "_core.abnf")
    p.load_file(Path(__file__).parent / "_kaylee.abnf")

    opts = Opts(
        use_regex=True,
        strategy=ParsingStrategy.FIRST_MATCH,
        inline_strategy=InlineStrategy.AGGRESSIVE,
    )

    g = p.compile(opts=opts)

    ast = g.parse_file_ast(Path(__file__).parent / "ky.ky")
    ast.render()
