import itertools
from collections.abc import Iterator
from typing import NamedTuple, final

from frozenintset import FrozenIntSet

from .actions import Action, Push, Set, Pop, Require
from .defs import RuleFlag
from .exc import ParseError
from .expr import Alt, Char, Expr, Many, Ref, Seq
from .grammar import Grammar, Rule
from .simple_expr import IntExpr, BoolExpr, stack_refs, counter_refs
from .tokenize import clean, tokenize, TokenKind, Token


@final
class _Rule(NamedTuple):
    name: str
    mode: str
    expr: Expr


type _Pragma = _FlagPragma | _StackPragma | _CounterPragma | _PushPragma | _PopPragma | _SetPragma | _RequirePragma


@final
class _FlagPragma(NamedTuple):
    flag: str


@final
class _StackPragma(NamedTuple):
    stack_name: str


@final
class _CounterPragma(NamedTuple):
    counter_name: str
    counter_value: int


@final
class _PushPragma(NamedTuple):
    stack_name: str
    int_expr: str


@final
class _PopPragma(NamedTuple):
    stack_name: str


@final
class _SetPragma(NamedTuple):
    counter_name: str
    int_expr: str


@final
class _RequirePragma(NamedTuple):
    bool_expr: str


@final
class _AABNFParser:
    __slots__ = ("_it", "_current_token")

    def __init__(self) -> None:
        self._it: Iterator[Token] | None = None
        self._current_token: Token | None = None

    def _accept(self, *kinds: TokenKind) -> Token | None:
        if self._current_token is not None and self._current_token.kind in kinds:
            token = self._current_token
            self._current_token = next(self._it, None)  # type: ignore[arg-type]
            return token
        return None

    def _expect(self, kind: TokenKind) -> Token:
        if (token := self._accept(kind)) is not None:
            return token
        raise ParseError(f"Expected {kind.name}, got {token}")

    def _parse_rule(self) -> _Rule | _Pragma | None:
        if (token := self._accept(TokenKind.RULESTART)) is not None:
            name = token.value
            mode = self._expect(TokenKind.DEFINE).value
            expr = self._parse_term()
            return _Rule(name, mode, expr)
        if (token := self._accept(TokenKind.FLAG_PRAGMA)) is not None:
            flag = token.groups["flag_name"]
            return _FlagPragma(flag)
        if (token := self._accept(TokenKind.STACK_PRAGMA)) is not None:
            stack_name = token.groups["stack_name"]
            return _StackPragma(stack_name)
        if (token := self._accept(TokenKind.COUNTER_PRAGMA)) is not None:
            counter_name = token.groups["counter_name"]
            counter_value = int(token.groups.get("counter_value", None) or 0)
            return _CounterPragma(counter_name, counter_value)
        if (token := self._accept(TokenKind.PUSH_PRAGMA)) is not None:
            stack_name = token.groups["push_stack"]
            simple_expr = token.groups["push_stack_expr"]
            return _PushPragma(stack_name, simple_expr)
        if (token := self._accept(TokenKind.POP_PRAGMA)) is not None:
            stack_name = token.groups["pop_stack"]
            return _PopPragma(stack_name)
        if (token := self._accept(TokenKind.SET_PRAGMA)) is not None:
            counter_name = token.groups["set_counter"]
            simple_expr = token.groups["set_counter_expr"]
            return _SetPragma(counter_name, simple_expr)
        if (token := self._accept(TokenKind.REQUIRE_PRAGMA)) is not None:
            simple_expr = token.groups["require_expr"]
            return _RequirePragma(simple_expr)
        return None

    def _parse_term(self) -> Expr:
        match self._parse_alternations():
            case Alt(es) if all(isinstance(e, Char) for e in es):
                return Char(FrozenIntSet.union_all(e.allowed for e in es))  # type: ignore[union-attr]
            case e:
                return e

    def _parse_alternations(self) -> Expr:
        alternations = [self._parse_sequence()]
        while self._accept(TokenKind.SLASH):
            match self._parse_sequence():
                case Alt(es):
                    alternations.extend(es)
                case e:
                    alternations.append(e)
        if len(alternations) == 1:
            return alternations[0]
        return Alt.of(alternations)

    def _parse_sequence(self) -> Expr:
        exprs: list[Expr] = list()
        while (expr := self._parse_repeat()) is not None:
            # noinspection PyUnreachableCode
            match expr:
                case Seq(es):
                    exprs.extend(es)
                case _:
                    exprs.append(expr)
        return self._sequence(exprs)

    def _sequence(self, exprs: list[Expr]) -> Expr:
        # noinspection PyUnreachableCode
        match exprs:
            case []:
                raise ParseError
            case [e]:
                return e
            case es:
                return Seq.of(es)

    def _char(self, *chars: int) -> Char:
        return Char(FrozenIntSet(chars))

    def _range(self, lo: int, hi: int) -> Char:
        return Char(FrozenIntSet(range(lo, hi + 1)))

    def _parse_element(self) -> Expr | None:
        if (token := self._accept(TokenKind.RULENAME)) is not None:
            return Ref(token.value)
        elif self._accept(TokenKind.LPAREN) is not None:
            expr = self._parse_term()
            self._expect(TokenKind.RPAREN)
            return expr
        elif self._accept(TokenKind.LBRACK) is not None:
            expr = self._parse_term()
            self._expect(TokenKind.RBRACK)
            return Many(max=1, expr=expr)
        elif (token := self._accept(TokenKind.CIVAL)) is not None:
            return self._sequence([self._char(ord(c.lower()), ord(c.upper())) for c in token.groups["ival"]])
        elif (token := self._accept(TokenKind.CSVAL)) is not None:
            return self._sequence([self._char(ord(c)) for c in token.groups["sval"]])
        elif (token := self._accept(TokenKind.HEXRANGE)) is not None:
            lo = int(token.groups["xlo"], 16)
            hi = int(token.groups["xhi"], 16)
            return self._range(lo, hi)
        elif (token := self._accept(TokenKind.HEXVAL)) is not None:
            vs = [int(v, 16) for v in token.groups["xval"].split(".")]
            return self._sequence([self._char(v) for v in vs])
        elif (token := self._accept(TokenKind.DECRANGE)) is not None:
            lo = int(token.groups["dlo"], 10)
            hi = int(token.groups["dhi"], 10)
            return self._range(lo, hi)
        elif (token := self._accept(TokenKind.DECVAL)) is not None:
            vs = [int(v, 10) for v in token.groups["dval"].split(".")]
            return self._sequence([self._char(v) for v in vs])
        elif (token := self._accept(TokenKind.BINRANGE)) is not None:
            lo = int(token.groups["blo"], 2)
            hi = int(token.groups["bhi"], 2)
            return self._range(lo, hi)
        elif (token := self._accept(TokenKind.BINVAL)) is not None:
            vs = [int(v, 2) for v in token.groups["bval"].split(".")]
            return self._sequence([self._char(v) for v in vs])
        elif (token := self._accept(TokenKind.PROSEVAL)) is not None:
            raise ParseError(f"prose values can not be compiled at pos={token.pos}")
        return None

    def _parse_repeat(self) -> Expr | None:
        if (token := self._accept(TokenKind.REPEAT)) is None:
            return self._parse_element()
        if (expr := self._parse_element()) is None:
            raise ParseError
        min_: int = 0
        max_: int | None = None
        if "exact" in token.groups:
            min_ = int(token.groups["exact"])
            max_ = int(token.groups["exact"])
        else:
            if "min" in token.groups:
                min_ = int(token.groups["min"])
            if "max" in token.groups:
                max_ = int(token.groups["max"])
        if min_ > 0:
            if max_ is not None:
                max_ -= min_
            prefix = itertools.repeat(expr, min_)
            if max_ == 0:
                return Seq.of(prefix)
            return Seq.of(prefix, [Many(max=max_, expr=expr)])
        return Many(max=max_, expr=expr)

    def _parse(self) -> Iterator[_Rule | _Pragma]:
        while (rule := self._parse_rule()) is not None:
            yield rule

    def _parse_bool_expr(self, expr: str) -> BoolExpr:
        # todo
        return ...

    def _parse_int_expr(self, expr: str) -> IntExpr:
        # todo
        return ...

    def parse(self, abnf_text: str) -> Grammar:
        self._it = clean(tokenize(abnf_text))
        self._current_token = next(self._it, None)  # type: ignore[arg-type]
        stacks: set[str] = set()
        counters: dict[str, int] = dict()
        grammar = Grammar()
        flags: set[RuleFlag] = set()
        actions: list[Action] = list()
        for line in self._parse():
            match line:
                case _Rule(name, mode, expr):
                    match mode:
                        case "=" if name in grammar:
                            raise ParseError(f"redefinition of rule {name}")
                        case "=":
                            grammar[name] = Rule(name, expr, frozenset(flags), tuple(actions))
                            flags.clear()
                            actions.clear()
                        case "=/" if name not in grammar:
                            raise ParseError(f"extension of rule {name} which has not been introduced yet")
                        case "=/":
                            grammar.extend(name, expr)
                case _FlagPragma(flag):
                    flags.add(RuleFlag(flag))
                case _StackPragma(stack_name):
                    if stack_name in stacks:
                        raise ParseError(f"redeclaration of stack {stack_name}")
                    stacks.add(stack_name)
                case _CounterPragma(counter_name, counter_value):
                    if counter_name in counters:
                        raise ParseError(f"redeclaration of counter {counter_name}")
                    counters[counter_name] = counter_value
                case _PushPragma(stack_name, expr):
                    int_expr = self._parse_int_expr(expr)
                    if stack_name not in stacks:
                        raise ParseError(f"@push references stack {stack_name} which has not been declared yet")
                    if missing_stacks := stack_refs(int_expr).difference(stacks):
                        raise ParseError(f"Missing stack references in @push expression: {', '.join(missing_stacks)}")
                    if missing_counters := counter_refs(int_expr).difference(counters):
                        raise ParseError(
                            f"Missing counter references in @push expression: {', '.join(missing_counters)}"
                        )
                    actions.append(Push(stack_name, int_expr))
                case _PopPragma(stack_name):
                    if stack_name not in stacks:
                        raise ParseError(f"@pop references stack {stack_name} which has not been declared yet")
                    actions.append(Pop(stack_name))
                case _SetPragma(counter_name, expr):
                    int_expr = self._parse_int_expr(expr)
                    if counter_name not in counters:
                        raise ParseError(f"@set references counter {counter_name} which has not been declared yet")
                    if missing_stacks := stack_refs(int_expr).difference(stacks):
                        raise ParseError(f"Missing stack references in @set expression: {', '.join(missing_stacks)}")
                    if missing_counters := counter_refs(int_expr).difference(counters):
                        raise ParseError(
                            f"Missing counter references in @set expression: {', '.join(missing_counters)}"
                        )
                    actions.append(Set(counter_name, int_expr))
                case _RequirePragma(expr):
                    bool_expr = self._parse_bool_expr(expr)
                    if missing_stacks := stack_refs(bool_expr).difference(stacks):
                        raise ParseError(
                            f"Missing stack references in @require expression: {', '.join(missing_stacks)}"
                        )
                    if missing_counters := counter_refs(bool_expr).difference(counters):
                        raise ParseError(
                            f"Missing counter references in @require expression: {', '.join(missing_counters)}"
                        )
                    actions.append(Require(bool_expr))
        return grammar


def parse(text: str) -> Grammar:
    parser = _AABNFParser()
    return parser.parse(text)
