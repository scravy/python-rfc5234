import dataclasses
import itertools
import re
from collections import deque, ChainMap
from collections.abc import (
    Generator,
    Iterator,
    Set,
    Mapping,
)
from pathlib import Path
from typing import NamedTuple, final, Final, Self

from frozenintset import FrozenIntSet
from strongly_connected_components import strongly_connected_components

from .abnf import read_abnf
from .ast import AST, ASTBuilder, ASTLeaf, ASTNode
from .compile import compile_expr, CompilationContext
from .context import Context
from .defs import (
    InlineStrategy,
    Opts,
    ParseEvent,
    ParseEventKind,
    ParsingStrategy,
)
from .exc import (
    EndOfFile,
    EntrypointInferrenceFailed,
    Expected,
    InexhaustiveParse,
    LeftRecursionDetected,
    MissingReference,
    MultipleEntrypointsDefined,
    NoAlternativeMatched,
    NoParse,
    ParseError,
    RequireFailed,
)
from .expr import (
    Alt,
    Char,
    Expr,
    Many,
    Ref,
    Regex,
    Seq,
    refs,
    inline,
    all_refs,
    can_be_empty,
)
from .simple_expr import (
    And,
    BoolExpr,
    Comparison,
    ComparisonOp,
    CounterRef,
    EmptyFn,
    IntApp1,
    IntApp2,
    IntExpr,
    IntFn1,
    IntFn2,
    IntLit,
    Not,
    Or,
    SpecialRef,
    TopFn,
    eval_bool_expr,
    eval_int_expr,
)
from .tokenize import clean, tokenize, TokenKind, Token
from .util import consume


@final
@dataclasses.dataclass(frozen=True)
class _Options:
    events: ParseEventKind = ParseEventKind.START | ParseEventKind.VALUE | ParseEventKind.END
    events_filter: re.Pattern[str] | None = None
    strategy: ParsingStrategy = ParsingStrategy.LONGEST_MATCH
    use_regex: bool = False
    use_cache: bool = True
    inline_strategy: InlineStrategy = InlineStrategy.AGGRESSIVE

    def check(self, ev: ParseEvent) -> bool:
        if self.events is not None and not (self.events & ev.kind):
            return False
        if self.events_filter is not None and not self.events_filter.fullmatch(ev.rule):
            return False
        return True

    def override(self, opts: Opts | None) -> Self:
        if opts is None:
            return self
        return _Options(
            events=opts.events if opts.events is not None else self.events,
            events_filter=opts.events_filter if opts.events_filter is not None else self.events_filter,
            strategy=opts.strategy if opts.strategy is not None else self.strategy,
            use_regex=opts.use_regex if opts.use_regex is not None else self.use_regex,
            use_cache=opts.use_cache if opts.use_cache is not None else self.use_cache,
            inline_strategy=opts.inline_strategy if opts.inline_strategy is not None else self.inline_strategy,
        )


### Rules and simple expressions for rule actions/guards ###############################################################


type Action = _PushPragma | _PopPragma | _SetPragma | _RequirePragma


@final
class Rule(NamedTuple):
    expr: Expr
    flags: frozenset[str]
    actions: tuple[Action, ...]

    def eval(self, ctx: Context, pos: int):
        local_ctx = ctx.fork()
        for action in self.actions:
            match action:
                case _PushPragma(stack_name, expr):
                    local_ctx.push(stack_name, eval_int_expr(ctx, pos, expr))
                case _PopPragma(stack_name):
                    local_ctx.pop(stack_name)
                case _SetPragma(counter_name, expr):
                    local_ctx.set(counter_name, eval_int_expr(ctx, pos, expr))
                case _RequirePragma(expr):
                    if not eval_bool_expr(ctx, pos, expr):
                        raise RequireFailed
        ctx.join(local_ctx)

    def add_branch(self, expr: Expr) -> Self:
        match self.expr, expr:
            case Alt(es1), Alt(es2):
                expr = Alt.of(es1, es2)
            case Alt(es1), _:
                expr = Alt.of(es1, [expr])
            case _, Alt(es2):
                expr = Alt.of([expr], es2)
            case e1, e2:
                expr = Alt.of([e1, e2])
        return self.with_expr(expr)

    def with_expr(self, expr: Expr) -> Self:
        return Rule(expr, self.flags, self.actions)


@final
class Grammar:
    __slots__ = ("_rules", "_context", "_entrypoint", "_opts", "_hash", "_cache")

    def __init__(
        self,
        rules: dict[str, Rule],
        entrypoint: str | None,
        opts: Opts | None,
        stacks: set[str],
        counters: dict[str, int],
    ):
        self._opts: Final[_Options] = _Options().override(opts)
        entrypoint, rules, nullable = self._check_rules(self._opts, rules, entrypoint)
        self._entrypoint: Final[str] = entrypoint
        self._rules: Final[dict[str, Rule]] = self._compile(rules, nullable, self._opts)
        self._context: Final[Context] = Context(
            stacks=ChainMap({name: None for name in stacks}),
            counters=ChainMap({name: value for name, value in counters.items()}),
        )
        self._hash: Final[int] = hash(tuple(sorted((name, rule.expr) for name, rule in self._rules.items())))
        self._cache: Final[
            dict[
                tuple[int, int, int, _Options],
                tuple[int, list[ParseEvent]],
            ]
        ] = dict()

    def __hash__(self) -> int:
        return self._hash

    def __iter__(self) -> Iterator[str]:
        yield from self._rules.keys()

    def __getitem__(self, item: str) -> Rule:
        return self._rules[item]

    def __contains__(self, item) -> bool:
        return item in self._rules

    @classmethod
    def _check_rules(
        cls, opts: _Options, rules: dict[str, Rule], entrypoint: str | None
    ) -> tuple[str, dict[str, Rule], frozenset[str]]:
        g = dict()
        to_be_inlined: set[str] = set()
        force_inlined: set[str] = set()
        entrypoints: set[str] = set()
        for name, rule in rules.items():
            if opts.inline_strategy == InlineStrategy.AGGRESSIVE and "emit" not in rule.flags and not rule.actions:
                to_be_inlined.add(name)
            if opts.inline_strategy != InlineStrategy.NO_INLINE and "inline" in rule.flags:
                force_inlined.add(name)
            if "entrypoint" in rule.flags:
                entrypoints.add(name)
        if len(entrypoints) > 1:
            raise MultipleEntrypointsDefined(entrypoints)
        if entrypoint is None and len(entrypoints) == 1:
            entrypoint = next(iter(entrypoints))
        for name, rule in rules.items():
            refnames = frozenset(refs(rule.expr))
            for ref in refnames:
                if ref not in rules:
                    raise MissingReference(name, ref)
            g[name] = refnames
        sccs: tuple[frozenset[str], ...] = tuple(strongly_connected_components(g))
        for scc in sccs:
            if len(scc) > 1:
                to_be_inlined -= scc
        to_be_inlined |= force_inlined
        if entrypoint is None:
            if (r := next(iter(sccs[-1]), None)) is not None:
                entrypoint = r
        if entrypoint is None and len(rules) == 1:
            entrypoint = next(iter(rules))
        if entrypoint is None:
            raise EntrypointInferrenceFailed
        to_be_inlined -= {entrypoint}

        def resolve(rn: str) -> Expr:
            return rules[rn].expr

        nullable = cls._compute_nullability(rules, sccs)
        cls._check_left_recursion(rules, nullable)
        for ix, scc in enumerate(sccs):
            for name in scc:
                if name in to_be_inlined:
                    for jx in range(ix, len(sccs)):
                        for n in sccs[jx]:
                            rules[n] = rules[n].with_expr(inline(rules[n].expr, resolve, name))
        reachable = all_refs(resolve, entrypoint)
        return (
            entrypoint,
            {name: rule for name, rule in rules.items() if name == entrypoint or name in reachable},
            frozenset(name for name in nullable if name in reachable),
        )

    @classmethod
    def _compile(cls, rules: Mapping[str, Rule], nullable: frozenset[str], opts: _Options) -> dict[str, Rule]:
        ctx = CompilationContext.new(use_regex=opts.use_regex, nullable=nullable)
        return {name: rule.with_expr(compile_expr(ctx, rule.expr)) for name, rule in rules.items()}

    @staticmethod
    def _compute_nullability(
        rules: Mapping[str, Rule],
        sccs: tuple[frozenset[str], ...],
    ) -> set[str]:
        nullable: set[str] = set()
        for scc in sccs:
            changed = True
            while changed:
                changed = False
                for rule in scc:
                    if rule not in nullable and can_be_empty(rules[rule].expr, nullable):
                        nullable.add(rule)
                        changed = True
        return nullable

    @staticmethod
    def _check_left_recursion(rules: Mapping[str, Rule], nullable: Set[str]) -> None:
        leftmost_cache: dict[str, frozenset[str]] = dict()

        def check_leftmost(rulename: str, stack: list[str]) -> frozenset[str]:
            if rulename in stack:
                raise LeftRecursionDetected([*stack, rulename])
            if rulename in leftmost_cache:
                return leftmost_cache[rulename]

            stack.append(rulename)
            expr = rules[rulename].expr
            out: set[str] = set()

            def recurse(e: Expr) -> bool:
                match e:
                    case Ref(ruleref):
                        out.update(check_leftmost(ruleref, stack))
                        return ruleref in nullable
                    case Alt(branches):
                        return any(recurse(b) for b in branches)
                    case Seq(steps):
                        for s in steps:
                            if not recurse(s):
                                return False
                        return True  # all steps nullable
                    case Many(expr=subexpr):
                        recurse(subexpr)
                        return True
                    case Char():
                        return False
                raise RuntimeError("unreachable code")

            recurse(expr)
            stack.pop()

            out_frozen = frozenset(out)
            leftmost_cache[rulename] = out_frozen
            return out_frozen

        for rule in rules:
            check_leftmost(rule, [])

    @property
    def rules(self) -> Set[str]:
        return self._rules.keys()

    @property
    def entrypoint(self) -> str:
        return self._entrypoint

    def parse_file(
        self,
        path: str | Path,
        *,
        encoding: str = "utf8",
        opts: Opts | None = None,
    ) -> Iterator[ParseEvent]:
        yield from self.parse(
            Path(path).read_text(encoding=encoding),
            opts=opts,
        )

    def parse(self, s: str, /, *, opts: Opts | None = None) -> Iterator[ParseEvent]:
        """
        Tries to parse the complete string passed as first argument.
        If the string can not be consumed fully, an InexhaustiveParse
        exception is raised.

        Returns the parse events that occur during parsing (entering a rule, exiting a rule,
        seeing a terminal value, etc.).  Which ParseEvents should be emitted can be
        configured by the passed `opts=`.
        """
        try:
            ix = yield from self._parse(Ref(self.entrypoint), s, 0, self._opts.override(opts), self._context)
            if ix != len(s):
                raise InexhaustiveParse(f"did not exhaustively match stopped at {s[ix : ix + 10]}")
        finally:
            self._cache.clear()

    def parse_file_ast(self, path: str | Path, *, encoding: str = "utf8", opts: Opts | None = None) -> AST:
        return self.parse_ast(
            Path(path).read_text(encoding=encoding),
            opts=opts,
        )

    def parse_ast(self, s: str, *, opts: Opts | None = None) -> AST:
        opts = opts or Opts()
        opts = Opts(
            **{
                **{f: getattr(opts, f) for f in opts._fields},
                **{
                    "events": ParseEventKind.START | ParseEventKind.END,
                    "events_filter": None,
                },
            }
        )
        stack: deque[ASTBuilder] = deque()
        stack.append(ASTBuilder(self.entrypoint, 0, []))
        for evk, name, pos, _ in self.parse(s, opts=opts):
            match evk:
                case ParseEventKind.START:
                    stack.append(ASTBuilder(name, pos, []))
                case ParseEventKind.END:
                    ast = stack.pop()
                    stack[-1].children.append(ast.build(s, pos))
        return stack.pop().build(s, len(s))

    def _parse_cached(self, expr: Expr, s: str, ix: int, opts: _Options, ctx: Context) -> tuple[int, list[ParseEvent]]:
        cache_key = (id(expr), id(s), ix, opts)
        if cache_key not in self._cache:
            self._cache[cache_key] = self._parse_uncached(expr, s, ix, opts, ctx)
        return self._cache[cache_key]

    def _parse_uncached(
        self, expr: Expr, s: str, ix: int, opts: _Options, ctx: Context
    ) -> tuple[int, list[ParseEvent]]:
        return consume(self._parse(expr, s, ix, opts, ctx))

    def _parse_buffered(
        self, expr: Expr, s: str, ix: int, opts: _Options, ctx: Context
    ) -> tuple[int, list[ParseEvent]]:
        """Apply parser `expr` on `s` from offset `ix` – depending on `opts` use either cached or uncached path."""
        if opts.use_cache:
            return self._parse_cached(expr, s, ix, opts, ctx)
        else:
            return self._parse_uncached(expr, s, ix, opts, ctx)

    def _parse_ref(self, expr: Ref, s: str, ix: int, opts: _Options, ctx: Context) -> Generator[ParseEvent, None, int]:
        rule = self[expr.name]
        if "emit" in rule.flags and opts.check(ev := ParseEvent(ParseEventKind.START, expr.name, ix)):
            yield ev
        pix = ix
        ix = yield from self._parse(rule.expr, s, ix, opts, ctx)
        if rule.actions:
            rule.eval(ctx, ix)
        if "emit" in rule.flags and opts.check(ev := ParseEvent(ParseEventKind.VALUE, expr.name, ix, s[pix:ix])):
            yield ev
        if "emit" in rule.flags and opts.check(ev := ParseEvent(ParseEventKind.END, expr.name, ix)):
            yield ev
        return ix

    def _parse_alt_first_match(
        self, expr: Alt, s: str, ix: int, opts: _Options, ctx: Context
    ) -> Generator[ParseEvent, None, int]:
        errors: list[NoParse] = list()
        for e in expr.branches:
            try:
                local_ctx = ctx.fork()
                ix, rs = self._parse_buffered(e, s, ix, opts, local_ctx)
                ctx.join(local_ctx)
                yield from rs
                return ix
            except NoParse as err:
                errors.append(err)
        raise NoAlternativeMatched(f"No alternative matched; pos={ix}; trying to apply {expr}", errors)

    def _parse_alt_longest_match(
        self, expr: Alt, s: str, ix: int, opts: _Options, ctx: Context
    ) -> Generator[ParseEvent, None, int]:
        errors: list[NoParse] = list()
        current_best: int = 0
        current_best_events: list[ParseEvent] = list()
        current_best_ctx: Context = ctx
        for e in expr.branches:
            try:
                local_ctx = ctx.fork()
                r = self._parse_buffered(e, s, ix, opts, local_ctx)
                if r[0] > current_best:
                    current_best_ctx = local_ctx
                    current_best, current_best_events = r
            except NoParse as err:
                errors.append(err)
        if not current_best:
            raise NoAlternativeMatched(
                f"No alternative matched; pos={ix}; trying to apply {expr}",
                errors,
            )
        ctx.join(current_best_ctx)
        yield from current_best_events
        return current_best

    def _parse_seq(self, expr: Seq, s: str, ix: int, opts: _Options, ctx: Context) -> Generator[ParseEvent, None, int]:
        for e in expr.steps:
            ix = yield from self._parse(e, s, ix, opts, ctx)
        return ix

    def _parse_repeat(self, r: Many, s: str, ix: int, opts: _Options, ctx: Context) -> Generator[ParseEvent, None, int]:
        if r.max is not None:
            for _ in range(r.max):
                try:
                    local_ctx = ctx.fork()
                    parser = self._parse(r.expr, s, ix, opts, local_ctx)
                    ix, rs = consume(parser)
                    ctx.join(local_ctx)
                    yield from rs
                except NoParse:
                    break
        else:
            while True:
                try:
                    local_ctx = ctx.fork()
                    parser = self._parse(r.expr, s, ix, opts, local_ctx)
                    ix, rs = consume(parser)
                    ctx.join(local_ctx)
                    yield from rs
                except NoParse:
                    break
        return ix

    def _parse_char(self, expr: Char, s: str, ix: int) -> int:
        try:
            if (c := ord(s[ix])) in expr.allowed:  # type: ignore[attr-defined]
                return ix + 1
        except IndexError as err:
            raise EndOfFile(ix, expr, expr.allowed) from err
        raise Expected(ix, expr, expr.allowed, c)

    def _parse_regex(self, regex: Regex, s: str, ix: int) -> int:
        if (m := regex.pattern.match(s, ix)) is not None:
            return m.end()
        raise NoParse(f"Expected {regex.pattern.pattern} at pos={ix}")

    def _parse(self, expr: Expr, s: str, ix: int, opts: _Options, ctx: Context) -> Generator[ParseEvent, None, int]:
        match expr:
            case Ref():
                return (yield from self._parse_ref(expr, s, ix, opts, ctx))
            case Alt() if opts.strategy == ParsingStrategy.FIRST_MATCH:
                return (yield from self._parse_alt_first_match(expr, s, ix, opts, ctx))
            case Alt() if opts.strategy == ParsingStrategy.LONGEST_MATCH:
                return (yield from self._parse_alt_longest_match(expr, s, ix, opts, ctx))
            case Seq():
                return (yield from self._parse_seq(expr, s, ix, opts, ctx))
            case Many():
                return (yield from self._parse_repeat(expr, s, ix, opts, ctx))
            case Char():
                return self._parse_char(expr, s, ix)
            case Regex():
                return self._parse_regex(expr, s, ix)
        raise RuntimeError(f"Unknown expression: {expr}; pos={ix}")


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
    expr: IntExpr


@final
class _PopPragma(NamedTuple):
    stack_name: str


@final
class _SetPragma(NamedTuple):
    counter_name: str
    expr: IntExpr


@final
class _RequirePragma(NamedTuple):
    expr: BoolExpr


@final
class SimpleExpressionParser:
    def __init__(self):
        g = read_abnf("simple_expr")
        p = Parser()
        p.load(g)
        self._bool_expr_grammar = p.compile("bool-expr")
        self._int_expr_grammar = p.compile("int-expr")

    @classmethod
    def stack_refs(cls, expr: IntExpr | BoolExpr) -> frozenset[str]:
        return frozenset(cls._stack_refs(expr))

    @classmethod
    def _stack_refs(cls, expr: IntExpr | BoolExpr) -> Iterator[str]:
        match expr:
            case IntApp1(arg1=arg):
                yield from cls._stack_refs(arg)
            case IntApp2(arg1=arg1, arg2=arg2):
                yield from cls._stack_refs(arg1)
                yield from cls._stack_refs(arg2)
            case TopFn(stack_name):
                yield stack_name
            case And(left=left, right=right):
                yield from cls._stack_refs(left)
                yield from cls._stack_refs(right)
            case Or(left=left, right=right):
                yield from cls._stack_refs(left)
                yield from cls._stack_refs(right)
            case Not(expr=expr):
                yield from cls._stack_refs(expr)
            case Comparison(left=left, right=right):
                yield from cls._stack_refs(left)
                yield from cls._stack_refs(right)
            case EmptyFn(stack_name):
                yield stack_name

    @classmethod
    def counter_refs(cls, expr: IntExpr | BoolExpr) -> frozenset[str]:
        return frozenset(cls._counter_refs(expr))

    @classmethod
    def _counter_refs(cls, expr: IntExpr | BoolExpr) -> Iterator[str]:
        match expr:
            case IntApp1(arg1=arg):
                yield from cls._counter_refs(arg)
            case IntApp2(arg1=arg1, arg2=arg2):
                yield from cls._counter_refs(arg1)
                yield from cls._counter_refs(arg2)
            case CounterRef(counter_name):
                yield counter_name
            case And(left=left, right=right):
                yield from cls._counter_refs(left)
                yield from cls._counter_refs(right)
            case Or(left=left, right=right):
                yield from cls._counter_refs(left)
                yield from cls._counter_refs(right)
            case Not(expr=expr):
                yield from cls._counter_refs(expr)
            case Comparison(left=left, right=right):
                yield from cls._counter_refs(left)
                yield from cls._counter_refs(right)

    def parse_bool_expr(self, s: str, /) -> BoolExpr:
        ast = self._bool_expr_grammar.parse_ast(s)
        return self._compile_bool_ast(ast)

    def _compile_bool_ast(self, ast: AST) -> BoolExpr:
        match ast:
            case ASTNode("bool-fn", [ASTLeaf("fn-empty"), ASTLeaf("stack-ref", stack_name)]):
                return EmptyFn(stack_name)
            case ASTNode("bool-fn", [ASTLeaf("fn-not"), arg]):
                bool_expr = self._compile_bool_ast(arg)
                return Not(bool_expr)
            case ASTNode("bool-expr", [left, ASTLeaf("comparison-op", op), right]):
                left_int_expr = self._compile_int_ast(left)
                right_int_expr = self._compile_int_ast(right)
                return Comparison(ComparisonOp(op), left_int_expr, right_int_expr)
            case ASTNode("bool-expr", [left, ASTLeaf("bool-op", "and"), right]):
                left_bool_expr = self._compile_bool_ast(left)
                right_bool_expr = self._compile_bool_ast(right)
                return And(left_bool_expr, right_bool_expr)
            case ASTNode("bool-expr", [left, ASTLeaf("bool-op", "or"), right]):
                left_bool_expr = self._compile_bool_ast(left)
                right_bool_expr = self._compile_bool_ast(right)
                return Or(left_bool_expr, right_bool_expr)
            case ASTNode("bool-arg" | "bool-expr", [top]):
                return self._compile_bool_ast(top)
        raise ParseError(f"Not recognized: {ast}")

    def parse_int_expr(self, s: str, /) -> IntExpr:
        ast = self._int_expr_grammar.parse_ast(s)
        return self._compile_int_ast(ast)

    def _compile_int_ast(self, ast: AST) -> IntExpr:
        match ast:
            case ASTLeaf("counter-ref", ref):
                return CounterRef(ref)
            case ASTLeaf("special-ref", ref):
                return SpecialRef(ref)
            case ASTLeaf("int-lit", value):
                return IntLit(int(value))
            case ASTNode("int-fn", [ASTLeaf("fn-top"), ASTLeaf("stack-ref", stack_name)]):
                return TopFn(stack_name)
            case ASTNode("int-fn", [ASTLeaf("fn-arg1", fn), arg]):
                func1 = IntFn1(fn)
                expr = self._compile_int_ast(arg)
                return IntApp1(func1, expr)
            case ASTNode("int-fn", [ASTLeaf("fn-arg2", fn), arg1, arg2]):
                func2 = IntFn2(fn)
                arg1_expr = self._compile_int_ast(arg1)
                arg2_expr = self._compile_int_ast(arg2)
                return IntApp2(func2, arg1_expr, arg2_expr)
            case ASTNode("int-expr", [left, ASTLeaf("int-op", op), right]):
                left_expr = self._compile_int_ast(left)
                right_expr = self._compile_int_ast(right)
                func2 = IntFn2(op)
                return IntApp2(func2, left_expr, right_expr)
            case ASTNode("int-arg" | "int-expr", [top]):
                return self._compile_int_ast(top)
        raise ParseError(f"not recognized: {ast}")


@final
class Parser:
    __slots__ = (
        "_it",
        "_current_token",
        "_rules",
        "_stacks",
        "_counters",
        "_expression_parser",
    )

    def __init__(self) -> None:
        self._it: Iterator[Token] | None = None
        self._current_token: Token | None = None
        self._rules: dict[str, Rule] = dict()
        self._stacks: set[str] = set()
        self._counters: dict[str, int] = dict()
        self._expression_parser: SimpleExpressionParser | None = None

    @property
    def expression_parser(self):
        if self._expression_parser is None:
            self._expression_parser = SimpleExpressionParser()
        return self._expression_parser

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
            try:
                int_expr = self.expression_parser.parse_int_expr(simple_expr)
            except NoParse as err:
                raise ParseError(f"failed parsing simple expression in {token.value}") from err
            return _PushPragma(stack_name, int_expr)
        if (token := self._accept(TokenKind.POP_PRAGMA)) is not None:
            stack_name = token.groups["pop_stack"]
            return _PopPragma(stack_name)
        if (token := self._accept(TokenKind.SET_PRAGMA)) is not None:
            counter_name = token.groups["set_counter"]
            simple_expr = token.groups["set_counter_expr"]
            try:
                int_expr = self.expression_parser.parse_int_expr(simple_expr)
            except NoParse as err:
                raise ParseError(f"failed parsing simple expression in {token.value}") from err
            return _SetPragma(counter_name, int_expr)
        if (token := self._accept(TokenKind.REQUIRE_PRAGMA)) is not None:
            simple_expr = token.groups["require_expr"]
            try:
                bool_expr = self.expression_parser.parse_bool_expr(simple_expr)
            except NoParse as err:
                raise ParseError(f"failed parsing simple expression in {token.value}") from err
            return _RequirePragma(bool_expr)
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

    def load(self, abnf_text: str) -> None:
        self._it = clean(tokenize(abnf_text))
        self._current_token = next(self._it, None)  # type: ignore[arg-type]
        new_rules: set[str] = set()
        flags: set[str] = set()
        actions: list[Action] = list()
        for line in self._parse():
            match line:
                case _Rule(name, mode, expr):
                    match mode:
                        case "=" if name in new_rules:
                            raise ParseError(f"redefinition of rule {name}")
                        case "=":
                            self._rules[name] = Rule(expr, frozenset(flags), tuple(actions))
                            flags.clear()
                            actions.clear()
                            new_rules.add(name)
                        case "=/" if name not in new_rules:
                            raise ParseError(f"extension of rule {name} which has not been introduced yet")
                        case "=/":
                            self._rules[name] = self._rules[name].add_branch(expr)
                case _FlagPragma(flag):
                    flags.add(flag)
                case _StackPragma(stack_name):
                    if stack_name in self._stacks:
                        raise ParseError(f"redeclaration of stack {stack_name}")
                    self._stacks.add(stack_name)
                case _CounterPragma(counter_name, counter_value):
                    if counter_name in self._counters:
                        raise ParseError(f"redeclaration of counter {counter_name}")
                    self._counters[counter_name] = counter_value
                case _PushPragma(stack_name, expr) as action:
                    if stack_name not in self._stacks:
                        raise ParseError(f"@push references stack {stack_name} which has not been declared yet")
                    if missing_stacks := self.expression_parser.stack_refs(expr).difference(self._stacks):
                        raise ParseError(f"Missing stack references in @push expression: {', '.join(missing_stacks)}")
                    if missing_counters := self.expression_parser.counter_refs(expr).difference(self._counters):
                        raise ParseError(
                            f"Missing counter references in @push expression: {', '.join(missing_counters)}"
                        )
                    actions.append(action)
                case _PopPragma(stack_name) as action:
                    if stack_name not in self._stacks:
                        raise ParseError(f"@pop references stack {stack_name} which has not been declared yet")
                    actions.append(action)
                case _SetPragma(counter_name, expr) as action:
                    if counter_name not in self._counters:
                        raise ParseError(f"@set references counter {counter_name} which has not been declared yet")
                    if missing_stacks := self.expression_parser.stack_refs(expr).difference(self._stacks):
                        raise ParseError(f"Missing stack references in @set expression: {', '.join(missing_stacks)}")
                    if missing_counters := self.expression_parser.counter_refs(expr).difference(self._counters):
                        raise ParseError(
                            f"Missing counter references in @set expression: {', '.join(missing_counters)}"
                        )
                    actions.append(action)
                case _RequirePragma(expr) as action:
                    if missing_stacks := self.expression_parser.stack_refs(expr).difference(self._stacks):
                        raise ParseError(
                            f"Missing stack references in @require expression: {', '.join(missing_stacks)}"
                        )
                    if missing_counters := self.expression_parser.counter_refs(expr).difference(self._counters):
                        raise ParseError(
                            f"Missing counter references in @require expression: {', '.join(missing_counters)}"
                        )
                    actions.append(action)

    def load_file(self, path: str | Path, *, encoding: str = "utf8"):
        self.load(Path(path).read_text(encoding=encoding))

    def compile(
        self,
        entrypoint: str | None = None,
        opts: Opts | None = None,
    ) -> Grammar:
        g = Grammar(self._rules, entrypoint, opts, self._stacks, self._counters)
        return g
