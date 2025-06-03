from collections.abc import Iterator
from itertools import chain
from typing import final, NamedTuple, Self, Final, NewType

from strongly_connected_components import strongly_connected_components

from .actions import Action
from .defs import RuleFlag, InlineStrategy
from .exc import MissingReference, MultipleEntrypointsDefined, EntrypointInferrenceFailed
from .expr import Expr, Alt, refs, can_be_empty, inline


@final
class Rule(NamedTuple):
    name: str
    expr: Expr
    flags: frozenset[RuleFlag] = frozenset()
    actions: tuple[Action, ...] = tuple()

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
        return Rule(self.name, expr, self.flags, self.actions)


CanonicalRuleName = NewType("CanonicalRuleName", str)


def canonical(rulename: str) -> CanonicalRuleName:
    return CanonicalRuleName(rulename.lower())


@final
class Grammar:
    __slots__ = (
        "_rules",
        "_entrypoint",
        "_sccs",
        "_nullable",
    )

    def __init__(self) -> None:
        self._entrypoint: CanonicalRuleName | None = None
        self._rules: Final[dict[CanonicalRuleName, Rule]] = dict()
        self._sccs: tuple[frozenset[CanonicalRuleName], ...] | None = None
        self._nullable: frozenset[CanonicalRuleName] | None = None

    def _invalidate(self):
        self._sccs = None
        self._nullable = None

    @property
    def sccs(self) -> tuple[frozenset[CanonicalRuleName], ...]:
        if self._sccs is None:
            g = dict()
            for name, rule in self._rules.items():
                refnames = frozenset(canonical(ref) for ref in refs(rule.expr))
                for ref in refnames:
                    if ref not in self._rules:
                        raise MissingReference(name, ref)
                g[name] = refnames
            self._sccs = tuple(strongly_connected_components(g))
        return self._sccs

    @property
    def nullable(self) -> frozenset[CanonicalRuleName]:
        if self._nullable is None:
            nullable: set[CanonicalRuleName] = set()
            for scc in self.sccs:
                changed = True
                while changed:
                    changed = False
                    for rule in scc:
                        if rule not in nullable and can_be_empty(self._rules[rule].expr, nullable):
                            nullable.add(rule)
                            changed = True
            self._nullable = frozenset(nullable)
        return self._nullable

    def is_nullable(self, rule: str) -> bool:
        return canonical(rule) in self.nullable

    @property
    def is_trivial(self) -> bool:
        return len(self._rules) == 1

    @property
    def is_empty(self) -> bool:
        return not self._rules

    @property
    def entrypoint(self) -> str | None:
        entrypoint = self._entrypoint
        if entrypoint is None:
            if self.is_empty:
                return None
            if self.is_trivial:
                # trivial
                entrypoint = next(iter(self._rules))
            else:
                # infer entrypoint from strongly connected components
                entrypoint = next(iter(self.sccs[-1]), None)
        if entrypoint is None:
            return None
        return self[entrypoint].name

    @entrypoint.setter
    def entrypoint(self, name: str) -> None:
        self._entrypoint = canonical(name)

    @property
    def _entrypoint_safe(self) -> CanonicalRuleName:
        if (entrypoint := self.entrypoint) is None:
            raise EntrypointInferrenceFailed
        return canonical(entrypoint)

    def inline(self, inline_strategy: InlineStrategy) -> None:
        to_be_inlined: set[CanonicalRuleName] = set()
        force_inlined: set[CanonicalRuleName] = set()
        for name, rule in self._rules.items():
            if any(canonical(ref) == name for ref in refs(rule.expr)):
                # a rule that directly references itself can not be inlined
                continue
            if rule.actions:
                # rules with actions can not be inlined in the current implementation
                # (would need to attach actions to expressions)
                continue
            if inline_strategy == InlineStrategy.AGGRESSIVE and RuleFlag.EMIT not in rule.flags:
                to_be_inlined.add(name)
            if inline_strategy != InlineStrategy.NO_INLINE and RuleFlag.INLINE in rule.flags:
                force_inlined.add(name)
        sccs = self.sccs
        for scc in sccs:
            if len(scc) > 1:
                # mutually recursive rules are never inlined, unless explicitly annotated with @inline
                to_be_inlined -= scc
        to_be_inlined |= force_inlined
        to_be_inlined -= {self._entrypoint_safe}
        for ix, scc in enumerate(sccs):
            for name in scc:
                if name in to_be_inlined:
                    # inline rule at all occurrences of it's name
                    # all rules that can reference it come after
                    for jx in range(ix, len(sccs)):
                        for n in sccs[jx]:
                            self._rules[n] = self._rules[n].with_expr(inline(self._rules[n].expr, self.get, name))
        # dependency graph has changed
        self._invalidate()

    @property
    def unused(self) -> frozenset[CanonicalRuleName]:
        return frozenset(self._rules.keys()) - frozenset(chain.from_iterable(self.sccs))

    def reachable(self, start: str) -> frozenset[CanonicalRuleName]:
        def _all_refs(name: CanonicalRuleName, visited: set[CanonicalRuleName]) -> Iterator[CanonicalRuleName]:
            visited.add(name)
            expr = self._rules[name].expr
            for ref in refs(expr):
                cref = canonical(ref)
                yield cref
                if cref not in visited:
                    yield from _all_refs(cref, visited)

        return frozenset(_all_refs(canonical(start), set()))

    def remove_unused(self) -> None:
        if not self.is_trivial:
            for name in self.unused:
                del self._rules[name]
            # dependency graph does not change, no need to invalidate

    def remove_unreachable(self) -> None:
        reachable_from_entrypoint = self.reachable(self._entrypoint_safe)
        unreachable = [name for name in self._rules if name not in reachable_from_entrypoint]
        for name in unreachable:
            del self._rules[name]
        self._invalidate()

    def __iter__(self) -> Iterator[str]:
        for rule in self._rules.values():
            yield rule.name

    def __getitem__(self, rulename: str) -> Rule:
        return self._rules[canonical(rulename)]

    def add(self, rule: Rule) -> None:
        self[rule.name] = rule

    def get(self, rulename: str) -> Expr:
        return self[rulename].expr

    def extend(self, rulename: str, expr: Expr):
        self[rulename] = self[rulename].add_branch(expr)

    def __copy__(self) -> Self:
        g = Grammar()
        g._rules.update(self._rules)
        return g

    def __setitem__(self, name: str, rule: Rule | Expr) -> None:
        new_rule: Rule = rule if isinstance(rule, Rule) else Rule(name, rule)
        new_name = canonical(name)
        if canonical(new_rule.name) != new_name:
            raise ValueError("rule.name must case insensitively match name it is registered as")
        if new_name in self._rules:
            old_rule = self._rules[new_name]
            if self._sccs is not None:
                old_refs = frozenset(refs(old_rule.expr))
                new_refs = frozenset(refs(new_rule.expr))
                if old_refs != new_refs:
                    self._sccs = None
        if RuleFlag.ENTRYPOINT in new_rule.flags:
            if self._entrypoint is not None and self._entrypoint != new_name:
                raise MultipleEntrypointsDefined([self._rules[self._entrypoint].name, new_rule.name])
            self._entrypoint = new_name
        self._rules[new_name] = new_rule
        self._invalidate()

    def __contains__(self, name: str) -> bool:
        return canonical(name) in self._rules
