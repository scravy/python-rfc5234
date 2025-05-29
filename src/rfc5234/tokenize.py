import re
from collections.abc import Iterator
from enum import Enum
from pathlib import Path
from typing import NamedTuple, final, Final


from colorama import Back, Fore, Style

from .exc import UnexpectedCharacter


@final
class TokenDef(NamedTuple):
    pattern: re.Pattern[str]
    style: str | tuple[str, ...] = Style.RESET_ALL

    def get_style(self) -> tuple[str, ...]:
        if isinstance(self.style, tuple):
            return self.style
        return (self.style,)


_WS: Final[re.Pattern[str]] = re.compile(r"\s|;[\x09\x20-\x7E]+\r?\n")

_RULENAME: Final[re.Pattern[str]] = re.compile(r"[A-Za-z]([A-Za-z0-9-])*")

_SIMPLE_EXPR: Final[re.Pattern[str]] = re.compile(r"[ $a-zA-Z0-9()<>=*+-]+")


@final
class TokenKind(Enum):
    RULESTART = TokenDef(
        pattern=re.compile(rf"(?m:^{_RULENAME.pattern})"),
        style=(Back.BLACK, Style.BRIGHT, Fore.LIGHTWHITE_EX),
    )
    STACK_PRAGMA = TokenDef(
        pattern=re.compile(
            rf"(?m:^(;; )?@@stack (?P<stack_name>{_RULENAME.pattern}))\n"
        ),
        style=Fore.LIGHTRED_EX,
    )
    COUNTER_PRAGMA = TokenDef(
        pattern=re.compile(
            rf"(?m:^(;; )?@@counter (?P<counter_name>{_RULENAME.pattern})( (?P<counter_value>[0-9]+))?)\n"
        ),
        style=Fore.LIGHTRED_EX,
    )
    PUSH_PRAGMA = TokenDef(
        pattern=re.compile(
            rf"(?m:^(;; )?@push (?P<push_stack>{_RULENAME.pattern}) (?P<push_stack_expr>{_SIMPLE_EXPR.pattern}))\n"
        ),
    )
    POP_PRAGMA = TokenDef(
        pattern=re.compile(rf"(?m:^(;; )?@pop (?P<pop_stack>{_RULENAME.pattern}))\n"),
    )
    SET_PRAGMA = TokenDef(
        pattern=re.compile(
            rf"(?m:^(;; )?@set (?P<set_counter>{_RULENAME.pattern}) (?P<set_counter_expr>{_SIMPLE_EXPR.pattern}))\n"
        ),
    )
    REQUIRE_PRAGMA = TokenDef(
        pattern=re.compile(
            rf"(?m:^(;; )?@require (?P<require_expr>{_SIMPLE_EXPR.pattern}))\n"
        ),
    )
    FLAG_PRAGMA = TokenDef(
        pattern=re.compile(rf"(?m:^(;; )?@(?P<flag_name>{_RULENAME.pattern}))\n"),
        style=Fore.LIGHTRED_EX,
    )
    SKIP = TokenDef(
        pattern=_WS,
        style=Fore.LIGHTBLACK_EX,
    )
    CSVAL = TokenDef(
        pattern=re.compile(r"%s\x22(?P<sval>[\x20-\x21\x23-\x7E]*)\x22"),
        style=(Back.BLACK, Fore.BLUE),
    )
    CIVAL = TokenDef(
        pattern=re.compile(r"(%i)?\x22(?P<ival>[\x20-\x21\x23-\x7E]*)\x22"),
        style=(Back.BLACK, Fore.BLUE),
    )
    BINRANGE = TokenDef(
        pattern=re.compile(r"%b(?P<blo>[01]+)-(?P<bhi>[01]+)"),
        style=(Back.BLACK, Fore.MAGENTA),
    )
    BINVAL = TokenDef(
        pattern=re.compile(r"%b(?P<bval>[01]+(\.[01]+)*)"),
        style=(Back.BLACK, Fore.MAGENTA),
    )
    DECRANGE = TokenDef(
        pattern=re.compile(r"%d(?P<dlo>[0-9]+)-(?P<dhi>[0-9]+)"),
        style=(Back.BLACK, Fore.MAGENTA),
    )
    DECVAL = TokenDef(
        pattern=re.compile(r"%d(?P<dval>[0-9]+(\.[0-9]+)*)"),
        style=(Back.BLACK, Fore.MAGENTA),
    )
    HEXRANGE = TokenDef(
        pattern=re.compile(r"%x(?P<xlo>[0-9a-fA-F]+)-(?P<xhi>[0-9a-fA-F]+)"),
        style=(Back.BLACK, Fore.MAGENTA),
    )
    HEXVAL = TokenDef(
        pattern=re.compile(r"%x(?P<xval>[0-9a-fA-F]+(\.[0-9a-fA-F]+)*)"),
        style=(Back.BLACK, Fore.MAGENTA),
    )
    PROSEVAL = TokenDef(
        pattern=re.compile(r"<(?P<pval>[\x20-\x3D\x3F-\x7E]*)>"),
        style=(Back.WHITE, Fore.BLACK),
    )
    RULENAME = TokenDef(
        pattern=_RULENAME,
        style=(Back.BLACK, Fore.YELLOW),
    )
    DEFINE = TokenDef(
        pattern=re.compile(r"=/|="),
        style=(Back.BLACK, Fore.GREEN),
    )
    SLASH = TokenDef(
        pattern=re.compile(r"/"),
        style=(Back.BLACK, Fore.GREEN),
    )
    LBRACK = TokenDef(
        pattern=re.compile(r"\["),
        style=(Back.BLACK, Fore.GREEN),
    )
    RBRACK = TokenDef(
        pattern=re.compile(r"]"),
        style=(Back.BLACK, Fore.GREEN),
    )
    LPAREN = TokenDef(
        pattern=re.compile(r"\("),
        style=(Back.BLACK, Fore.GREEN),
    )
    RPAREN = TokenDef(
        pattern=re.compile(r"\)"),
        style=(Back.BLACK, Fore.GREEN),
    )
    REPEAT = TokenDef(
        pattern=re.compile(r"(?P<min>[0-9]+)?\*(?P<max>[0-9]+)?|\*|(?P<exact>[0-9]+)"),
        style=(Back.BLACK, Fore.CYAN),
    )
    MISMATCH = TokenDef(
        pattern=re.compile(r"."),
        style=(Style.BRIGHT, Back.RED, Fore.BLACK),
    )


@final
class Token(NamedTuple):
    kind: TokenKind
    value: str
    groups: dict[str, str]
    pos: int


def tokenize(s: str, /) -> Iterator[Token]:
    tok_regex: re.Pattern[str] = re.compile(
        "|".join(f"(?P<{k.name}>{k.value.pattern.pattern})" for k in TokenKind)
    )
    pos: int = 0
    m = tok_regex.match(s, pos)
    while m is not None:
        kind: TokenKind = getattr(TokenKind, m.lastgroup)  # type: ignore[arg-type]
        value = m.group()
        yield Token(
            kind,
            value,
            {k: v for k, v in m.groupdict().items() if k.islower() and v is not None},
            pos,
        )
        pos = m.end()
        m = tok_regex.match(s, pos)


def clean(tokens: Iterator[Token]):
    for token in tokens:
        match token.kind:
            case TokenKind.SKIP:
                pass
            case TokenKind.MISMATCH:
                raise UnexpectedCharacter(
                    f"Unexpected character: {token.value} at {token.pos}"
                )
            case _:
                yield token


def render_rules(path: Path, *, render_groups: bool = False) -> None:
    print(Style.BRIGHT, Fore.WHITE, path.name.upper(), Style.RESET_ALL, sep="")
    abnf_text = Path(path).read_text()
    for tok in tokenize(abnf_text):
        print(*tok.kind.value.get_style(), tok.value, end=Style.RESET_ALL, sep="")
        if render_groups and tok.groups:
            print(Fore.RED, tok.groups, end=Style.RESET_ALL, sep="")
