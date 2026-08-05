# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import annotations

import copy
import enum
import logging
import re
import urllib.parse
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Generic, Optional, TYPE_CHECKING, TypeVar

import sqlglot
from flask import current_app, has_app_context
from jinja2 import nodes, Template
from sqlglot import exp
from sqlglot.dialects.dialect import (
    Dialect,
    Dialects,
    DialectType,
)
from sqlglot.dialects.singlestore import SingleStore
from sqlglot.errors import ParseError
from sqlglot.generator import Generator
from sqlglot.optimizer.pushdown_predicates import (
    pushdown_predicates,
)
from sqlglot.optimizer.scope import (
    Scope,
    ScopeType,
    traverse_scope,
)

from superset.exceptions import QueryClauseValidationException, SupersetParseError
from superset.sql.dialects import DB2, Dremio, Firebolt, OpenSearch, Pinot, Vertica

if TYPE_CHECKING:
    from superset.models.core import Database


logger = logging.getLogger(__name__)


def _check_script_length(script: str, engine: str | None) -> None:
    """
    Reject scripts whose UTF-8 byte length exceeds the configured maximum
    before they reach sqlglot. Sits at every code path in this module that
    hands a string to ``sqlglot.parse`` or ``sqlglot.parse_one`` so the
    bound cannot be bypassed by a direct caller.

    The check is in bytes, not Unicode code points, because the
    threat model is parser memory and CPU on the encoded payload that
    sqlglot ingests.
    """
    # Imported lazily to avoid a circular import (``superset.config`` pulls in
    # ``superset.jinja_context``, which imports this module).
    from superset import config

    # The live app config wins when a Flask app context is active (honoring any
    # operator override); otherwise (Alembic migrations, scripts, isolated unit
    # tests) fall back to the documented default declared in ``superset.config``
    # so the bound stays sourced from configuration rather than duplicated here.
    max_length = (
        current_app.config.get("SQL_MAX_PARSE_LENGTH", config.SQL_MAX_PARSE_LENGTH)
        if has_app_context()
        else config.SQL_MAX_PARSE_LENGTH
    )

    if max_length is None:
        return
    if (byte_length := len(script.encode("utf-8"))) > max_length:
        raise SupersetParseError(
            script,
            engine,
            message=(
                f"SQL script length ({byte_length} bytes) exceeds the "
                f"configured maximum of {max_length} bytes."
            ),
        )


# mapping between DB engine specs and sqlglot dialects
SQLGLOT_DIALECTS = {
    "base": Dialects.DIALECT,
    "ascend": Dialects.HIVE,
    "awsathena": Dialects.ATHENA,
    "bigquery": Dialects.BIGQUERY,
    "datastore": Dialects.BIGQUERY,
    "clickhouse": Dialects.CLICKHOUSE,
    "clickhousedb": Dialects.CLICKHOUSE,
    "cockroachdb": Dialects.POSTGRES,
    "couchbase": Dialects.MYSQL,
    # "crate": ???
    # "databend": ???
    "databricks": Dialects.DATABRICKS,
    "db2": DB2,
    # "denodo": ???
    "dremio": Dremio,
    "drill": Dialects.DRILL,
    "druid": Dialects.DRUID,
    "duckdb": Dialects.DUCKDB,
    # "dynamodb": ???
    # "elasticsearch": ???
    # "exa": ???
    # "firebird": ???
    "firebolt": Firebolt,
    "gsheets": Dialects.SQLITE,
    "hana": Dialects.POSTGRES,
    "hive": Dialects.HIVE,
    # "ibmi": ???
    "impala": Dialects.HIVE,
    # "kustosql": ???
    # "kylin": ???
    "mariadb": Dialects.MYSQL,
    "motherduck": Dialects.DUCKDB,
    "mssql": Dialects.TSQL,
    "mysql": Dialects.MYSQL,
    "netezza": Dialects.POSTGRES,
    "oceanbase": Dialects.MYSQL,
    # "ocient": ???
    "odelasticsearch": OpenSearch,
    "oracle": Dialects.ORACLE,
    "parseable": Dialects.POSTGRES,
    "pinot": Pinot,
    "postgresql": Dialects.POSTGRES,
    "presto": Dialects.PRESTO,
    "pydoris": Dialects.DORIS,
    "redshift": Dialects.REDSHIFT,
    "risingwave": Dialects.RISINGWAVE,
    "shillelagh": Dialects.SQLITE,
    "singlestoredb": SingleStore,
    "snowflake": Dialects.SNOWFLAKE,
    # "solr": ???
    "spark": Dialects.SPARK,
    "sqlite": Dialects.SQLITE,
    "starrocks": Dialects.STARROCKS,
    "superset": Dialects.SQLITE,
    # "taosws": ???
    "teradatasql": Dialects.TERADATA,
    "trino": Dialects.TRINO,
    "vertica": Vertica,
    # "ydb" is a plugin dialect (ydb-sqlglot-plugin) auto-discovered via entry_points,
    # hence a string name rather than a class reference like the built-in dialects.
    "yql": "ydb",
}


def has_aggregate(expression: str, engine: str = "base") -> bool:
    """
    Return True if the SQL expression contains an aggregate function, ignoring
    only an aggregate that is *itself* windowed (``SUM(x) OVER (...)``), which
    doesn't collapse rows and is just as invalid under a GROUP BY as a plain
    column. A plain aggregate nested inside a windowed one
    (``SUM(SUM(x)) OVER ()``) still counts.

    Deliberately permissive so a valid query is never wrongly blocked: an
    aggregate inside a scalar subquery still counts, and it fails open (returns
    True) on a parse error or an unmodelled function (``exp.Anonymous``) that
    might itself be an aggregate.
    """
    dialect = SQLGLOT_DIALECTS.get(engine)
    try:
        parsed = sqlglot.parse_one(f"SELECT {expression}", dialect=dialect)
    except Exception:
        return True
    if parsed.find(exp.Anonymous):
        return True
    return any(
        not isinstance(agg.parent, exp.Window) for agg in parsed.find_all(exp.AggFunc)
    )


class LimitMethod(enum.Enum):
    """
    Limit methods.

    This is used to determine how to add a limit to a SQL statement.
    """

    FORCE_LIMIT = enum.auto()
    WRAP_SQL = enum.auto()
    FETCH_MANY = enum.auto()


class CTASMethod(enum.Enum):
    TABLE = enum.auto()
    VIEW = enum.auto()


def _normalized_generator(
    dialect_name: DialectType,
    *,
    pretty: bool,
    comments: bool,
) -> Generator:
    """
    Generator that preserves multi-argument DISTINCT expressions.

    Build a sqlglot generator that preserves user-written multi-argument
    DISTINCT expressions verbatim. Postgres, Presto, Trino, and DuckDB
    set ``MULTI_ARG_DISTINCT = False`` to emulate the unsupported
    ``COUNT(DISTINCT a, b)`` idiom via a ``CASE WHEN`` row-expression, which
    silently corrupts user-defined aggregates that natively accept multiple
    arguments. Superset's sanitize / format paths normalize user SQL — they
    do not transpile — so the emulation is undesirable here.
    """
    dialect = Dialect.get_or_raise(dialect_name)
    normalized_cls = type(
        f"Normalized{dialect.generator_class.__name__}",
        (dialect.generator_class,),
        {"MULTI_ARG_DISTINCT": True},
    )
    return normalized_cls(dialect=dialect, pretty=pretty, comments=comments)


class RLSMethod(enum.Enum):
    """
    Methods for enforcing RLS.
    """

    AS_PREDICATE = enum.auto()
    AS_SUBQUERY = enum.auto()


class RLSTransformer:
    """
    AST transformer to apply RLS rules.
    """

    def __init__(
        self,
        catalog: str | None,
        schema: str | None,
        rules: dict[Table, list[exp.Expression]],
    ) -> None:
        self.catalog = catalog
        self.schema = schema
        self.rules = rules

    def get_predicate(self, table_node: exp.Table) -> exp.Expression | None:
        """
        Get the combined RLS predicate for a table.
        """
        table = Table(
            table_node.name,
            table_node.db if table_node.db else self.schema,
            table_node.catalog if table_node.catalog else self.catalog,
        )
        if predicates := self.rules.get(table):
            return sqlglot.and_(*predicates)

        return None


class RLSAsPredicateTransformer(RLSTransformer):
    """
    Apply Row Level Security role as a predicate.

    This transformer will apply any RLS predicates to the relevant tables. For example,
    given the RLS rule:

        table: some_table
        clause: id = 42

    If a user subject to the rule runs the following query:

        SELECT foo FROM some_table WHERE bar = 'baz'

    The query will be modified to:

        SELECT foo FROM some_table WHERE bar = 'baz' AND id = 42

    This approach is probably less secure than using subqueries, so it's only used for
    databases without support for subqueries.
    """

    def __call__(self, node: exp.Expression) -> exp.Expression:
        if not isinstance(node, exp.Table):
            return node

        predicate = self.get_predicate(node)
        if not predicate:
            return node

        # Qualify columns with the parsed alias node, not ``node.alias``: that property
        # strips the quoting, and a string qualifier is emitted verbatim, so an alias
        # holding SQL would land inside the predicate. ``FROM t AS (c1, c2)`` has no
        # alias name at all; qualify with the table there, since an unqualified column
        # could otherwise resolve against an enclosing scope.
        table_alias = node.args.get("alias")
        qualifier = (table_alias and table_alias.this) or node.this
        for column in predicate.find_all(exp.Column):
            column.set("table", qualifier.copy())

        if isinstance(node.parent, exp.From):
            select = node.parent.parent
            if where := select.args.get("where"):
                predicate = exp.And(
                    this=predicate,
                    expression=exp.Paren(this=where.this),
                )
            select.set("where", exp.Where(this=predicate))

        elif isinstance(node.parent, exp.Join):
            join = node.parent
            if on := join.args.get("on"):
                predicate = exp.And(
                    this=predicate,
                    expression=exp.Paren(this=on),
                )
            join.set("on", predicate)

        return node


#: Prefix used when *naming* injected CTEs.  Nothing about the rewrite's correctness
#: depends on this prefix being reserved: collisions are resolved by picking another
#: name, and "is this reference inside a CTE we injected?" is answered by node
#: identity, never by the name.  Answering it by name would hand the caller an
#: opt-out -- a user CTE called ``__rls_0_orders`` would make every reference inside
#: it look already filtered, which is a complete bypass.
RLS_CTE_PREFIX = "__rls_"

#: ``exp.Table`` arguments that describe the relation the *reference* produces rather
#: than the physical table it names.  These stay at the reference site; every other
#: argument travels, untouched, into the CTE body with the table node.  A new sqlglot
#: argument therefore defaults to "travels with the table", which is the
#: semantics-preserving choice, because it stays attached to the same node.
_RLS_SITE_ARGS = ("alias", "joins", "laterals", "pivots")

#: Node types that can carry a ``WITH`` clause and are a query.
_RLS_QUERY_NODES = (exp.Select, exp.SetOperation, exp.Subquery)


class RLSUnsupportedError(Exception):
    """
    The statement cannot be row-filtered safely, so it must not run.

    Raised rather than returning a statement whose filtering cannot be vouched for.
    """


def _rls_real_reads(ast: exp.Expression) -> Iterable[tuple[exp.Table, Scope]]:
    """
    Yield every real (non-CTE) table read, paired with the scope it resolves in.

    This is *exactly* the enumeration and classifier the authorisation layer uses:
    ``extract_tables_from_statement`` walks ``scope.sources.values()`` and drops the
    entries ``is_cte`` recognises.  The rewrite drives target selection and its output
    self-check off this same helper, so the set of reads that are authorised, the set
    that are filtered, and the set that are verified are provably the same set -- a
    read cannot be authorised by one enumeration and skipped by another.

    ``scope.tables`` is deliberately *not* used: it omits a table that carries a join
    (``FROM (orders CROSS JOIN pub) sub`` surfaces only ``pub`` there), and classifying
    by ``scope.sources[alias_or_name]`` misreads a real table aliased to a CTE's key as
    the CTE.  Both drop authorised reads, i.e. leak.
    """
    for scope in traverse_scope(ast):
        for source in scope.sources.values():
            if isinstance(source, exp.Table) and not is_cte(source, scope):
                yield source, scope


def _rls_inside_injected_cte(node: exp.Expression, injected: frozenset[int]) -> bool:
    """Is ``node`` inside a CTE this rewrite created?  By identity, never by name."""
    if not injected:
        return False
    ancestor = node.parent
    while ancestor is not None:
        if id(ancestor) in injected:
            return True
        ancestor = ancestor.parent
    return False


def _rls_collect_targets(
    ast: exp.Expression,
    lookup: Callable[[exp.Table], list[exp.Expression] | None],
) -> list[tuple[exp.Table, list[exp.Expression]]]:
    """
    Every real table read in the statement that has a predicate.

    Enumerated through ``_rls_real_reads`` -- the authorisation layer's own view of the
    statement -- so a read the caller was authorised against is a read this hoists.
    """
    out: list[tuple[exp.Table, list[exp.Expression]]] = []
    seen: set[int] = set()
    for node, _scope in _rls_real_reads(ast):
        # A correlated ``LATERAL`` registers the same outer ``exp.Table`` node in two
        # scopes' ``sources``, so the identical node is yielded twice.  Hoist each
        # physical node once; otherwise it is wrapped in a CTE that reads the CTE that
        # reads it -- redundant, and unsafe for a non-idempotent predicate.
        if id(node) in seen:
            continue
        if predicates := lookup(node):
            seen.add(id(node))
            out.append((node, predicates))
    return out


def _rls_with_host(ast: exp.Expression) -> exp.Expression | None:
    """
    The node a ``WITH`` can attach to so it is in scope for the whole statement.

    It must be the **outermost** query.  Attach it to a nested one and the predicate
    re-enters an enclosing scope, which is the whole property this rewrite exists to
    provide.
    """
    if isinstance(ast, exp.Subquery) and isinstance(
        ast.this, (exp.Select, exp.SetOperation)
    ):
        # A ``Subquery`` renders its own ``WITH`` *before* its parentheses, which is
        # not valid anywhere it appears, so put the ``WITH`` inside them.
        return ast.this
    if isinstance(ast, _RLS_QUERY_NODES):
        return ast
    # ``INSERT INTO t <query>``, ``CREATE ... AS <query>``, ``COPY (<query>) TO ...``
    children = [
        child
        for child in (ast.args.get("this"), ast.args.get("expression"))
        if isinstance(child, _RLS_QUERY_NODES)
    ]
    return _rls_with_host(children[0]) if len(children) == 1 else None


def _rls_taken_names(ast: exp.Expression) -> set[str]:
    names = {(cte.alias or "").lower() for cte in ast.find_all(exp.CTE)}
    for table in ast.find_all(exp.Table):
        names.add(table.name.lower())
        if table.alias:
            names.add(str(table.alias).lower())
    return names


def _rls_sanitise(name: str) -> str:
    return "".join(c if (c.isalnum() or c == "_") else "_" for c in name)[:40] or "t"


def _rls_prepend_ctes(
    host: exp.Expression, ctes: Iterable[tuple[str, exp.Select]]
) -> list[exp.CTE]:
    """
    Prepend, never append.

    A non-recursive ``WITH`` item may only reference *preceding* items, so an appended
    CTE is a forward reference: Trino rejects it, and PostgreSQL resolves the name to
    the base table instead, which would silently undo the filter.
    """
    nodes = [
        exp.CTE(this=body, alias=exp.TableAlias(this=exp.to_identifier(name)))
        for name, body in ctes
    ]
    if existing := host.args.get("with_"):
        existing.set("expressions", nodes + existing.expressions)
    else:
        host.set("with_", exp.With(expressions=nodes))
    return nodes


def _rls_assert_all_filtered(
    ast: exp.Expression,
    lookup: Callable[[exp.Table], list[exp.Expression] | None],
    injected: frozenset[int] = frozenset(),
) -> None:
    """
    Post-condition, checked against the authorisation layer's own enumeration.

    Every real read the authorisation layer surfaces (``_rls_real_reads``, the same
    walk ``extract_tables_from_statement`` uses) that still has a predicate and is not
    now enclosed in a CTE this rewrite injected is a read the caller was authorised
    against and the rewrite failed to hoist.  Refuse rather than emit it unfiltered.

    This does *not* go through ``_rls_collect_targets``: a systematic mistake in target
    selection therefore cannot also silence its own check.  Should selection ever
    regress -- miss a joined table, misclassify an aliased read -- this still sees the
    read the authorisation layer authorised and fails closed.
    """
    leftover = [
        node
        for node, _scope in _rls_real_reads(ast)
        if not _rls_inside_injected_cte(node, injected) and lookup(node)
    ]
    if leftover:
        names = ", ".join(node.sql() for node in leftover)
        raise RLSUnsupportedError(
            f"unfiltered reference(s) survived the row-level security rewrite: {names}"
        )


def _rls_assert_ctes_emit_intact(
    ast: exp.Expression, dialect: DialectType, names: set[str]
) -> None:
    """
    Refuse if a CTE we injected does not serialise back to what we built.

    A generator may decorate a node we did not ask it to decorate.  One instance:
    inside a ``WITH RECURSIVE``, sqlglot's Trino generator synthesises a column-alias
    list for *every* CTE in the clause from that CTE's projection list, so a
    ``SELECT *`` body is emitted as ``__rls_0_t("*")`` -- a one-element alias list
    against a relation with however many columns the table has.  The engine rejects
    that today, but the
    validity of a security rewrite must not rest on an arity coincidence: against a
    single-column table the list would match and the statement's shape would be partly
    the caller's to choose.

    Checked by round-tripping rather than by naming the dialects that do it, so a
    different generator quirk in the same position is caught too.  Only the injected
    CTEs are inspected: what the *caller's* own CTEs serialise to is not this
    function's business.
    """
    if not names:
        return
    with_node = ast.args.get("with_") if hasattr(ast, "args") else None
    if not isinstance(with_node, exp.With) or not with_node.args.get("recursive"):
        # Only a recursive WITH is known to provoke this, and re-parsing every statement
        # to check for a quirk that has never appeared elsewhere is not worth the cost.
        return

    try:
        reparsed = sqlglot.parse_one(ast.sql(dialect=dialect), dialect=dialect)
    except Exception as ex:  # noqa: BLE001
        raise RLSUnsupportedError(
            f"the filtered statement does not parse back ({type(ex).__name__}), so its "
            "row-level security cannot be vouched for"
        ) from ex

    for cte in reparsed.find_all(exp.CTE):
        if cte.alias in names and cte.args.get("alias", exp.TableAlias()).args.get(
            "columns"
        ):
            raise RLSUnsupportedError(
                f"the row-level security CTE {cte.alias} is serialised with a column "
                "alias list this rewrite did not create, so the statement that would "
                "execute is not the statement that was built; this happens for a "
                "protected table inside a WITH RECURSIVE clause on some dialects"
            )


def _rls_unwrap_redundant_paren(
    ast: exp.Expression, targets: list[tuple[exp.Table, list[exp.Expression]]]
) -> exp.Expression:
    """
    Strip a redundant outermost parenthesisation so a ``WITH`` can attach.

    ``(SELECT ...)`` parses as a ``Subquery`` and there is nowhere valid to put the
    clause: outside the parentheses a ``Subquery`` renders its ``WITH`` before its own
    ``(``, and inside them some engines reject ``(WITH ... SELECT ...)`` outright.  At
    statement level those parentheses carry no meaning, so remove them rather than emit
    SQL the engine refuses.  Only when there is something to filter, and only when the
    parentheses hold nothing but the query: an alias or an attached join at this
    position is not a shape this can flatten.
    """
    if (
        targets
        and isinstance(ast, exp.Subquery)
        and isinstance(ast.this, (exp.Select, exp.SetOperation))
        and not any(ast.args.get(arg) for arg in _RLS_SITE_ARGS)
    ):
        return ast.this
    return ast


def _rls_hoist_target(
    node: exp.Table,
    predicates: list[exp.Expression],
    ctes: dict[tuple[str, str], tuple[str, exp.Select]],
    taken: set[str],
    counter: int,
    dialect: DialectType,
) -> int:
    """
    Hoist one protected reference into a filtered CTE and rebind it in place.

    ``ctes`` and ``taken`` are updated in place; the next unused CTE counter is
    returned so the caller can thread it across references.
    """
    original_name = node.name

    # Split the container: site arguments stay, everything else travels with the table
    # node into the CTE body, unread and uninterpreted.
    site = {key: node.args.get(key) for key in _RLS_SITE_ARGS}
    body_table = node.copy()
    for key in _RLS_SITE_ARGS:
        body_table.set(key, None)

    # One CTE per (physical-relation signature, predicate).  Two references to the same
    # table share a CTE -- one filtered scan instead of N -- while
    # ``t FOR VERSION AS OF 1`` and ``FOR VERSION AS OF 2`` are different physical
    # relations and must not.
    predicate = sqlglot.and_(*predicates)
    key = (body_table.sql(dialect=dialect), predicate.sql(dialect=dialect))
    if key not in ctes:
        while True:
            name = f"{RLS_CTE_PREFIX}{counter}_{_rls_sanitise(original_name)}"
            counter += 1
            if name.lower() not in taken:
                break
        taken.add(name.lower())
        ctes[key] = (
            name,
            exp.Select(
                expressions=[exp.Star()],
                from_=exp.From(this=body_table),
                where=exp.Where(this=predicate),
            ),
        )
    cte_name = ctes[key][0]

    # Rebind the name.  Nothing else about the node is read or rebuilt.
    for arg in list(node.args):
        if arg not in _RLS_SITE_ARGS:
            node.set(arg, None)
    node.set("this", exp.to_identifier(cte_name))
    for arg, value in site.items():
        if value:
            node.set(arg, value)

    # An unaliased reference lends its bare name to column qualifiers
    # (``SELECT orders.id FROM orders``).  Renaming it would strand those, so bind the
    # old name back as an alias.
    if not node.args.get("alias"):
        node.set(
            "alias",
            exp.TableAlias(this=exp.Identifier(this=original_name, quoted=True)),
        )
    return counter


def apply_rls_as_cte(
    ast: exp.Expression,
    lookup: Callable[[exp.Table], list[exp.Expression] | None],
    dialect: DialectType = None,
) -> exp.Expression:
    """
    Filter every protected reference by hoisting a CTE and rebinding the name.

    For example, given the RLS rule ``some_table`` -> ``id = 42`` and the query::

        SELECT foo FROM some_table WHERE bar = 'baz'

    the statement is rewritten to::

        WITH __rls_0_some_table AS (SELECT * FROM some_table WHERE id = 42)
        SELECT foo FROM __rls_0_some_table WHERE bar = 'baz'

    The predicate is *not* placed at the reference site.  It goes into the body of a
    CTE on the outermost query, where there is no enclosing scope, so an unqualified
    column in the predicate can only resolve against the relation it is meant to
    filter -- or the statement fails.  Placed at the reference site instead, the same
    unqualified column resolves outward into whatever scope encloses the reference,
    and a caller who supplies a same-named column turns the filter into a tautology.

    Nothing is replaced.  ``exp.Table`` conflates a physical table with a ``FROM``
    item: besides the name it carries the alias, the alias column list, its own
    ``joins`` and ``laterals``, ``TABLESAMPLE``, pivots and a time-travel version.
    Constructing a replacement node means rebuilding whichever of those the rewriter
    happens to know about, so the rest is dropped or swallowed, and a replacement is
    not re-walked -- one predicate is emitted for N relations.  Here only the *name*
    fields change, in place, so the walk stays total and nothing attached can be lost.

    A CTE reference is left untouched: it is not a table read, and the base-table read
    it stands for is filtered where the ``WITH`` item defines it.  This is why a query
    that reads a real table and also references a same-named CTE, such as
    ``WITH orders AS (SELECT id FROM orders) SELECT * FROM orders``, filters only the
    real read inside the ``WITH`` body and leaves both ``orders`` references bound to
    it.

    Reads are enumerated exactly as the authorisation layer enumerates them
    (``_rls_real_reads`` / ``extract_tables_from_statement``), so a table the caller was
    authorised against is a table this hoists.  The output is then re-checked against
    that same enumeration (``_rls_assert_all_filtered``): if any authorised read is
    still exposed the statement is refused, never emitted unfiltered.

    A statement whose outermost node cannot carry a ``WITH`` -- ``UPDATE``/``DELETE``
    /``MERGE`` that read a protected table in a predicate subquery -- is *refused*
    rather than filtered.  Filtering in place there would need the predicate injected
    into the enclosing ``WHERE``/``ON`` (the ``AS_PREDICATE`` method's job), which
    cannot be done without the scope-escape risks this method exists to avoid; failing
    closed is the safe choice and callers on such engines use ``AS_PREDICATE``.

    A hoisted CTE is materialised or treated as an optimiser fence by some engines, so
    predicate push-down across it can differ from the previous inline-subquery form.
    This changes query plans, not results.
    """
    targets = _rls_collect_targets(ast, lookup)
    ast = _rls_unwrap_redundant_paren(ast, targets)

    host = _rls_with_host(ast)

    if host is None:
        if targets:
            raise RLSUnsupportedError(
                f"{type(ast).__name__} cannot carry a WITH clause, so "
                f"{len(targets)} protected reference(s) cannot be filtered"
            )
        return ast

    taken = _rls_taken_names(ast)
    ctes: dict[tuple[str, str], tuple[str, exp.Select]] = {}
    counter = 0
    for node, predicates in targets:
        counter = _rls_hoist_target(node, predicates, ctes, taken, counter, dialect)

    injected: frozenset[int] = frozenset()
    if ctes:
        injected = frozenset(
            id(node) for node in _rls_prepend_ctes(host, list(ctes.values()))
        )

    _rls_assert_all_filtered(ast, lookup, injected)
    _rls_assert_ctes_emit_intact(ast, dialect, {name for name, _ in ctes.values()})

    return ast


@dataclass(eq=True, frozen=True)
class Table:
    """
    A fully qualified SQL table conforming to [[catalog.]schema.]table.
    """

    table: str
    schema: str | None = None
    catalog: str | None = None

    def __str__(self) -> str:
        """
        Return the fully qualified SQL table name.

        Should not be used for SQL generation, only for logging and debugging, since the
        quoting is not engine-specific.
        """
        return ".".join(
            urllib.parse.quote(part, safe="").replace(".", "%2E")
            for part in [self.catalog, self.schema, self.table]
            if part
        )

    def __eq__(self, other: Any) -> bool:
        return str(self) == str(other)

    def qualify(
        self,
        *,
        catalog: str | None = None,
        schema: str | None = None,
    ) -> Table:
        """
        Return a new Table with the given schema and/or catalog, if not already set.
        """
        return Table(
            table=self.table,
            schema=self.schema or schema,
            catalog=self.catalog or catalog,
        )


@dataclass(eq=True, frozen=True)
class Partition:
    """
    Partition object, with two attribute keys:
    is_partitioned_table and partition_column,
    used to provide partition information
    Here is an example of an object:
    Partition(is_partitioned_table=True, partition_column=("month", "day"))
    """

    is_partitioned_table: bool
    partition_column: tuple[str, ...] | None = None

    def __str__(self) -> str:
        """
        Return a string representation of the Partition object.
        """
        partition_column_str = (
            ", ".join(map(str, self.partition_column))
            if self.partition_column
            else "None"
        )
        return (
            f"Partition(is_partitioned_table={self.is_partitioned_table}, "
            f"partition_column=[{partition_column_str}])"
        )


# To avoid unnecessary parsing/formatting of queries, the statement has the concept of
# an "internal representation", which is the AST of the SQL statement. For most of the
# engines supported by Superset this is `sqlglot.exp.Expression`, but there is a special
# case: KustoKQL uses a different syntax and there are no Python parsers for it, so we
# store the AST as a string (the original query), and manipulate it with regular
# expressions.
InternalRepresentation = TypeVar("InternalRepresentation")

# The base type. This helps type checking the `split_query` method correctly, since each
# derived class has a more specific return type (the class itself). This will no longer
# be needed once Python 3.11 is the lowest version supported. See PEP 673 for more
# information: https://peps.python.org/pep-0673/
TBaseSQLStatement = TypeVar("TBaseSQLStatement")  # pylint: disable=invalid-name


class BaseSQLStatement(Generic[InternalRepresentation]):
    """
    Base class for SQL statements.

    The class should be instantiated with a string representation of the script and, for
    efficiency reasons, optionally with a pre-parsed AST. This is useful with
    `sqlglot.parse`, which will split a script in multiple already parsed statements.

    The `engine` parameters comes from the `engine` attribute in a Superset DB engine
    spec.
    """

    def __init__(
        self,
        statement: str | None = None,
        engine: str = "base",
        ast: InternalRepresentation | None = None,
    ):
        if ast:
            self._parsed = ast
        elif statement:
            self._parsed = self._parse_statement(statement, engine)
        else:
            raise ValueError("Either statement or ast must be provided")

        self.engine = engine
        self.tables = self._extract_tables_from_statement(self._parsed, self.engine)

    @classmethod
    def split_script(
        cls: type[TBaseSQLStatement],
        script: str,
        engine: str,
    ) -> list[TBaseSQLStatement]:
        """
        Split a script into multiple instantiated statements.

        This is a helper function to split a full SQL script into multiple
        `BaseSQLStatement` instances. It's used by `SQLScript` when instantiating the
        statements within a script.
        """
        raise NotImplementedError()

    @classmethod
    def _parse_statement(
        cls,
        statement: str,
        engine: str,
    ) -> InternalRepresentation:
        """
        Parse a string containing a single SQL statement, and returns the parsed AST.

        Derived classes should not assume that `statement` contains a single statement,
        and MUST explicitly validate that. Since this validation is parser dependent the
        responsibility is left to the children classes.
        """
        raise NotImplementedError()

    @classmethod
    def _extract_tables_from_statement(
        cls,
        parsed: InternalRepresentation,
        engine: str,
    ) -> set[Table]:
        """
        Extract all table references in a given statement.
        """
        raise NotImplementedError()

    def format(self, comments: bool = True) -> str:
        """
        Format the statement, optionally ommitting comments.
        """
        raise NotImplementedError()

    def get_settings(self) -> dict[str, str | bool]:
        """
        Return any settings set by the statement.

        For example, for this statement:

            sql> SET foo = 'bar';

        The method should return `{"foo": "'bar'"}`. Note the single quotes.
        """
        raise NotImplementedError()

    def is_select(self) -> bool:
        """
        Check if the statement is a `SELECT` statement.
        """
        raise NotImplementedError()

    def is_mutating(self) -> bool:
        """
        Check if the statement mutates data (DDL/DML).

        :return: True if the statement mutates data.
        """
        raise NotImplementedError()

    def is_destructive(self) -> bool:
        """
        Check if the statement is destructive DDL (DROP, TRUNCATE, ALTER).

        :return: True if the statement is destructive DDL.
        """
        raise NotImplementedError()

    def optimize(self) -> BaseSQLStatement[InternalRepresentation]:
        """
        Return optimized statement.
        """
        raise NotImplementedError()

    def check_functions_present(self, functions: set[str]) -> bool:
        """
        Check if any of the given functions are present in the script.

        :param functions: List of functions to check for
        :return: True if any of the functions are present
        """
        raise NotImplementedError()

    def check_tables_present(
        self, tables: set[str], default_schema: str | None = None
    ) -> bool:
        """
        Check if any of the given tables are present in the statement.

        :param tables: Set of table names to check for (case-insensitive)
        :param default_schema: Schema unqualified references resolve to at
            runtime (e.g. the session ``search_path`` / selected schema)
        :return: True if any of the tables are present
        """
        raise NotImplementedError()

    def changes_search_path(self) -> bool:
        """
        Check if the statement changes the session ``search_path``.

        Defaults to ``False``; engines whose statements can rebind unqualified
        schema resolution override this.

        :return: True if the statement changes the session ``search_path``
        """
        return False

    def get_disallowed_tables(
        self,
        tables: set[str],
        default_schema: str | None = None,
        schema_indeterminate: bool = False,
    ) -> set[str]:
        """
        Return the subset of ``tables`` referenced by this statement.

        :param tables: Set of table names to check for (case-insensitive)
        :param default_schema: Schema unqualified references resolve to at
            runtime (e.g. the session ``search_path`` / selected schema)
        :param schema_indeterminate: When True, unqualified references are
            matched against schema-qualified entries too (see
            :meth:`SQLStatement.get_disallowed_tables`)
        :return: The matched entries, in their original denylist form
        """
        raise NotImplementedError()

    def get_limit_value(self) -> int | None:
        """
        Get the limit value of the statement.
        """
        raise NotImplementedError()

    def set_limit_value(
        self,
        limit: int,
        method: LimitMethod = LimitMethod.FORCE_LIMIT,
    ) -> None:
        """
        Add a limit to the statement.
        """
        raise NotImplementedError()

    def has_cte(self) -> bool:
        """
        Check if the statement has a CTE.

        :return: True if the statement has a CTE at the top level.
        """
        raise NotImplementedError()

    def as_cte(self, alias: str = "__cte") -> BaseSQLStatement[InternalRepresentation]:
        """
        Rewrite the statement as a CTE.

        :param alias: The alias to use for the CTE.
        :return: A new BaseSQLStatement[InternalRepresentation] with the CTE.
        """
        raise NotImplementedError()

    def as_create_table(
        self,
        table: Table,
        method: CTASMethod,
    ) -> BaseSQLStatement[InternalRepresentation]:
        """
        Rewrite the statement as a `CREATE TABLE AS` statement.

        :param table: The table to create.
        :param method: The method to use for creating the table.
        :return: A new BaseSQLStatement[InternalRepresentation] with the CTE.
        """
        raise NotImplementedError()

    def has_subquery(self) -> bool:
        """
        Check if the statement has a subquery.

        :return: True if the statement has a subquery at the top level.
        """
        raise NotImplementedError()

    def parse_predicate(self, predicate: str) -> InternalRepresentation:
        """
        Parse a predicate string into an AST.

        :param predicate: The predicate to parse.
        :return: The parsed predicate.
        """
        raise NotImplementedError()

    def apply_rls(
        self,
        catalog: str | None,
        schema: str | None,
        predicates: dict[Table, list[InternalRepresentation]],
        method: RLSMethod,
    ) -> None:
        """
        Apply relevant RLS rules to the statement inplace.

        :param catalog: The default catalog for non-qualified table names
        :param schema: The default schema for non-qualified table names
        :param method: The method to use for applying the rules.
        """
        raise NotImplementedError()

    def __str__(self) -> str:
        return self.format()


class SQLStatement(BaseSQLStatement[exp.Expression]):
    """
    A SQL statement.

    This class is used for all engines with dialects that can be parsed using sqlglot.
    """

    # Function names that mutate server-side state but appear in the AST as
    # plain function calls inside a non-mutating wrapper. Used by
    # ``is_mutating()`` to classify e.g. PostgreSQL large-object writers.
    # Names are uppercased for comparison.
    _MUTATING_FUNCTION_NAMES: frozenset[str] = frozenset(
        {
            "LO_FROM_BYTEA",
            "LO_EXPORT",
            "LO_IMPORT",
            "LO_PUT",
            "LO_CREATE",
            "LO_CREAT",
            "LOWRITE",
            "LO_TRUNCATE",
            "LO_TRUNCATE64",
            "LO_UNLINK",
            # PostgreSQL sequence mutators. `SELECT setval('seq', N)` and
            # `SELECT nextval('seq')` look like reads but change sequence state
            # for every subsequent caller. (`currval` only reads the session's
            # last value, so it is intentionally not listed.)
            "SETVAL",
            "NEXTVAL",
        }
    )

    # PostgreSQL constructs that sqlglot represents as an opaque ``exp.Command``
    # (no structured AST). Each can mutate server state or wrap a DML body that
    # would otherwise be detected by node-type matching. Used by
    # ``is_mutating()``.
    _POSTGRES_MUTATING_COMMAND_NAMES: frozenset[str] = frozenset(
        {
            "DO",  # PL/pgSQL anonymous block
            "PREPARE",  # PREPARE u AS UPDATE ... ; EXECUTE u
            "EXECUTE",  # body is the prepared DML
            "CALL",  # procedure body may mutate
            "COPY",  # server-side file ingest into a table
            "GRANT",
            "REVOKE",
            # Only the command-fallback forms (e.g. SET ROLE / SET SESSION
            # AUTHORIZATION, which change the effective user) reach here as an
            # exp.Command. Structured `SET search_path = ...` /
            # `SET statement_timeout = ...` parse as exp.Set and are NOT matched
            # by this command-name path.
            "SET",
            "RESET",  # RESET ROLE / RESET ALL reverts SET; same class as SET
            "REFRESH",  # REFRESH MATERIALIZED VIEW
            "REINDEX",
            "VACUUM",
            # DDL head-tokens that sqlglot falls back to exp.Command for
            # whenever the body uses syntax it does not model
            # (CREATE EXTENSION/FUNCTION...LANGUAGE C/PUBLICATION/etc.,
            # ALTER ROLE/SYSTEM/..., DROP EXTENSION/RULE/...). Well-formed
            # CREATE TABLE/ALTER TABLE/DROP TABLE are already caught by the
            # node-type tuple; these entries close the fallback path.
            "CREATE",
            "ALTER",
            "DROP",
            "LOAD",  # LOAD '/path/lib.so' dlopens a shared library on the PG host
            # NOTE: `SHOW` is intentionally NOT included. It is a read (mutates
            # nothing), so classifying it as mutating would be wrong for every
            # is_mutating()/has_mutation() consumer (the commit decision, the
            # "only SELECT allowed" validators, limit handling), not just the
            # read-only gate. Gating information-disclosure reads such as
            # `SHOW server_version` belongs in a denylist (DISALLOWED_SQL_FUNCTIONS
            # already blocks version()/pg_read_file), not in the mutation check.
        }
    )

    # Dialects where `SELECT ... INTO target` is CTAS (creates a table, and so
    # mutates schema). Elsewhere the same syntax assigns into a variable and is
    # a read: Oracle PL/SQL `SELECT ... INTO v` and MySQL `SELECT ... INTO @v`
    # parse into an identical `exp.Select` with an `into` arg, so the dialect is
    # the only signal that distinguishes the mutating form from the read form.
    _SELECT_INTO_CTAS_DIALECTS: frozenset[Dialects] = frozenset(
        {
            Dialects.POSTGRES,
            Dialects.REDSHIFT,
            Dialects.TSQL,
        }
    )

    def __init__(
        self,
        statement: str | None = None,
        engine: str = "base",
        ast: exp.Expression | None = None,
    ):
        self._dialect = SQLGLOT_DIALECTS.get(engine)
        super().__init__(statement, engine, ast)

    @classmethod
    def _parse(cls, script: str, engine: str) -> list[exp.Expression]:
        """
        Parse helper.

        When the base dialect (engine="base" or unknown engines) fails to parse SQL
        containing backtick-quoted identifiers, we fall back to MySQL dialect which
        supports backticks natively. This handles cases like "Other" database type
        where users may have MySQL-compatible syntax with backtick-quoted table names.
        """
        _check_script_length(script, engine)
        dialect = SQLGLOT_DIALECTS.get(engine)
        try:
            statements = sqlglot.parse(script, dialect=dialect)
        except sqlglot.errors.ParseError as ex:
            # If parsing fails with base dialect (or no dialect for unknown engines)
            # and the script contains backticks, retry with MySQL dialect which
            # supports backtick-quoted identifiers
            if (dialect is None or dialect == Dialects.DIALECT) and "`" in script:
                logger.warning(
                    "Parsing with base dialect failed for engine %r; "
                    "script contains backticks, falling back to MySQL dialect",
                    engine,
                )
                try:
                    statements = sqlglot.parse(script, dialect=Dialects.MYSQL)
                except sqlglot.errors.ParseError:
                    # If MySQL dialect also fails, raise the original error
                    pass
                else:
                    return statements

            kwargs = (
                {
                    "highlight": ex.errors[0]["highlight"],
                    "line": ex.errors[0]["line"],
                    "column": ex.errors[0]["col"],
                }
                if ex.errors
                else {}
            )
            raise SupersetParseError(script, engine, **kwargs) from ex
        except sqlglot.errors.SqlglotError as ex:
            raise SupersetParseError(
                script,
                engine,
                message="Unable to parse script",
            ) from ex

        # `sqlglot` will parse comments after the last semicolon as a separate
        # statement; move them back to the last token in the last real statement
        if len(statements) > 1 and isinstance(statements[-1], exp.Semicolon):
            last_statement = statements.pop()
            target = statements[-1]
            for node in statements[-1].walk():
                if hasattr(node, "comments"):  # pragma: no cover
                    target = node

            target.comments = target.comments or []
            target.comments.extend(last_statement.comments)

        return statements

    @classmethod
    def split_script(
        cls,
        script: str,
        engine: str,
    ) -> list[SQLStatement]:
        return [
            cls(ast=ast, engine=engine) for ast in cls._parse(script, engine) if ast
        ]

    @classmethod
    def _parse_statement(
        cls,
        statement: str,
        engine: str,
    ) -> exp.Expression:
        """
        Parse a single SQL statement.
        """
        statements = cls.split_script(statement, engine)
        if len(statements) != 1:
            raise SupersetParseError(
                statement,
                engine,
                message="SQLStatement should have exactly one statement",
            )

        return statements[0]._parsed  # pylint: disable=protected-access

    @classmethod
    def _extract_tables_from_statement(
        cls,
        parsed: exp.Expression,
        engine: str,
    ) -> set[Table]:
        """
        Find all referenced tables.
        """
        dialect = SQLGLOT_DIALECTS.get(engine)
        return extract_tables_from_statement(parsed, dialect)

    def is_select(self) -> bool:
        """
        Check if the statement is a `SELECT` statement.
        """
        return isinstance(self._parsed, exp.Select)

    def is_mutating(self) -> bool:
        """
        Check if the statement mutates data (DDL/DML).

        :return: True if the statement mutates data.
        """
        mutating_nodes = (
            exp.Insert,
            exp.Update,
            exp.Delete,
            exp.Merge,
            exp.Create,
            exp.Drop,
            exp.TruncateTable,
            exp.Alter,
            # sqlglot has structured nodes for these DML/DCL forms in
            # PostgreSQL and other dialects; without them an opaque exp.Command
            # check would still miss the structured-parse path.
            exp.Copy,  # COPY <table> FROM/TO (server-side file ingest)
            exp.Grant,
            exp.Revoke,
            # COMMENT ON TABLE/COLUMN/etc. writes to system catalog pg_description.
            exp.Comment,
        )

        if self._parsed.find(*mutating_nodes):
            return True

        # `SELECT ... INTO new_table FROM ...` parses as `exp.Select` with an
        # `into` arg (Postgres-style CTAS variant). It creates a new table and
        # therefore mutates schema. Only treat it as mutating for dialects where
        # the syntax is CTAS; elsewhere it assigns into a variable (a read).
        if (
            self._dialect in self._SELECT_INTO_CTAS_DIALECTS
            and isinstance(self._parsed, exp.Select)
            and self._parsed.args.get("into")
        ):
            return True

        # Function calls that mutate server-side state without an enclosing
        # mutating AST node. Notable example: PostgreSQL large-object writers
        # (`lo_export` writes to the server filesystem, `lo_from_bytea`/
        # `lo_create`/`lo_put`/`lo_import`/`lowrite` mutate the pg_largeobject
        # catalog). These appear as plain function calls inside an `exp.Select`
        # and would otherwise pass the read-only gate. Every name in
        # _MUTATING_FUNCTION_NAMES is PostgreSQL-specific, so the walk is gated
        # on the dialect: other engines may expose read-only functions/UDFs with
        # the same names, and flagging those would wrongly block read-only
        # queries. Each parses as an `exp.Anonymous`, whose `.name` is the bare
        # function identifier. The walk is restricted to `exp.Anonymous` rather
        # than the broader `exp.Func`, because for built-in function nodes (e.g.
        # `exp.Upper`) `.name` returns the first argument's text, not the
        # function name, so `SELECT upper('lo_export')` would otherwise be
        # misclassified as mutating.
        if self._dialect == Dialects.POSTGRES and any(
            function.name.upper() in self._MUTATING_FUNCTION_NAMES
            for function in self._parsed.find_all(exp.Anonymous)
        ):
            return True

        # depending on the dialect (Oracle, MS SQL) the `ALTER` is parsed as a
        # command, not an expression - check at root level
        if isinstance(self._parsed, exp.Command) and self._parsed.name == "ALTER":
            return True  # pragma: no cover

        # PostgreSQL constructs that sqlglot represents as an opaque
        # `exp.Command` rather than a structured AST. Each of these can mutate
        # state or wrap a DML body that would otherwise be detected. The
        # `.name` attribute on `exp.Command` preserves the source-case of the
        # head keyword (so `create extension ...` would yield `'create'`),
        # which means the set lookup must be case-insensitive.
        if (
            self._dialect == Dialects.POSTGRES
            and isinstance(self._parsed, exp.Command)
            and self._parsed.name.upper() in self._POSTGRES_MUTATING_COMMAND_NAMES
        ):
            return True

        # Postgres runs DMLs prefixed by `EXPLAIN ANALYZE`, see
        # https://www.postgresql.org/docs/current/sql-explain.html
        if (
            self._dialect == Dialects.POSTGRES
            and isinstance(self._parsed, exp.Command)
            and self._parsed.name == "EXPLAIN"
            and self._parsed.expression.name.upper().startswith("ANALYZE ")
        ):
            analyzed_sql = self._parsed.expression.name[len("ANALYZE ") :]
            return SQLStatement(
                statement=analyzed_sql,
                engine=self.engine,
            ).is_mutating()

        return False

    def is_destructive(self) -> bool:
        """
        Check if the statement is destructive DDL (DROP, TRUNCATE, ALTER).

        Unlike ``is_mutating()``, this excludes non-destructive DML
        (INSERT, UPDATE, DELETE, MERGE) and CREATE.

        :return: True if the statement is destructive DDL.
        """
        destructive_nodes = (
            exp.Drop,
            exp.TruncateTable,
            exp.Alter,
        )

        for node_type in destructive_nodes:
            if self._parsed.find(node_type):
                return True

        # Handle ALTER parsed as Command (Oracle, MS SQL dialects)
        if isinstance(self._parsed, exp.Command) and self._parsed.name == "ALTER":
            return True  # pragma: no cover

        return False

    def format(self, comments: bool = True) -> str:
        """
        Pretty-format the SQL statement.
        """
        return _normalized_generator(
            self._dialect,
            pretty=True,
            comments=comments,
        ).generate(self._parsed, copy=True)

    def get_settings(self) -> dict[str, str | bool]:
        """
        Return the settings for the SQL statement.

            >>> statement = SQLStatement("SET foo = 'bar'")
            >>> statement.get_settings()
            {"foo": "'bar'"}

        """
        return {
            eq.this.sql(
                dialect=self._dialect,
                comments=False,
            ): eq.expression.sql(comments=False)
            for set_item in self._parsed.find_all(exp.SetItem)
            for eq in set_item.find_all(exp.EQ)
        }

    def optimize(self) -> SQLStatement:
        """
        Return optimized statement.
        """
        # only optimize statements that have a custom dialect
        if not self._dialect:
            return SQLStatement(ast=self._parsed.copy(), engine=self.engine)

        optimized = pushdown_predicates(self._parsed, dialect=self._dialect)

        return SQLStatement(ast=optimized, engine=self.engine)

    def check_functions_present(self, functions: set[str]) -> bool:
        """
        Check if any of the given functions are present in the script.

        :param functions: List of functions to check for
        :return: True if any of the functions are present
        """
        # Build the set of SQL-level function names present in the AST. For
        # Anonymous nodes the name is stored directly; for named Func nodes we
        # use sql_name(). We also add dialect parser aliases so that functions
        # that are normalised by a dialect (e.g. VERSION() -> CurrentVersion in
        # Postgres, sql_name = CURRENT_VERSION) can still be matched by their
        # original SQL name.
        dialect_cls = Dialect.get_or_raise(self._dialect) if self._dialect else None
        parser_cls = getattr(dialect_cls, "parser_class", None) if dialect_cls else None
        parser_functions: dict[str, Any] = (
            getattr(parser_cls, "FUNCTIONS", {}) if parser_cls is not None else {}
        )

        present: set[str] = set()
        for function in self._parsed.find_all(exp.Func):
            sql_name = function.sql_name()
            if sql_name != "ANONYMOUS":
                present.add(sql_name)
                # Add any dialect-level aliases that resolve to the same class
                # (e.g. 'VERSION' -> CurrentVersion when dialect is Postgres).
                func_type = type(function)
                for key, builder in parser_functions.items():
                    try:
                        if builder.__self__ is func_type:
                            present.add(key)
                    except AttributeError:
                        pass
            else:
                present.add(function.name.upper())

        # MySQL `@@<name>` syntax (also Oracle/SQL-Server `@@name`) parses as
        # `exp.SessionParameter`, which is *not* a subclass of `exp.Func`, so
        # the walk above misses it. Include those names so denylist entries
        # like `version` or `hostname` match `SELECT @@version`.
        for param in self._parsed.find_all(exp.SessionParameter):
            present.add(param.name.upper())

        return any(function.upper() in present for function in functions)

    def check_tables_present(
        self, tables: set[str], default_schema: str | None = None
    ) -> bool:
        """
        Check if any of the given tables are present in the statement.

        Denylist entries may be bare (``pg_stat_activity``) or
        schema-qualified (``information_schema.tables``). Bare entries
        match by table name regardless of schema; qualified entries
        require the schema to match too. This lets us block all access
        to ``information_schema`` without also blocking any
        user-authored table that happens to be named ``tables``.

        :param tables: Set of table names to check for (case-insensitive)
        :param default_schema: Schema unqualified references resolve to at
            runtime (e.g. the session ``search_path`` / selected schema)
        :return: True if any of the given tables is referenced
        """
        return bool(self.get_disallowed_tables(tables, default_schema))

    def changes_search_path(self) -> bool:
        """
        Return True if the statement changes the session ``search_path``.

        A ``SET search_path = ...`` makes unqualified references in later
        statements resolve to a schema other than the caller's
        ``default_schema``, so denylist matching against ``default_schema``
        alone becomes unreliable once such a statement is present.
        """
        # `SET search_path = schema` (and the `TO`/`SESSION`/`LOCAL` variants)
        # parse as a structured exp.Set, surfaced by get_settings(). Strip any
        # identifier quoting so `SET "search_path" = ...` (equivalent to the
        # unquoted form in Postgres) is still recognized.
        if any(key.strip('"').lower() == "search_path" for key in self.get_settings()):
            return True
        # `set_config('search_path', ...)` rebinds the search path through a
        # function call rather than a SET statement, so it never reaches
        # get_settings() and must be detected on the parsed tree.
        for func in self._parsed.find_all(exp.Anonymous):
            if (
                func.name.lower() == "set_config"
                and func.expressions
                and isinstance(func.expressions[0], exp.Literal)
                and func.expressions[0].name.lower() == "search_path"
            ):
                return True
        # Exotic forms (e.g. `SET search_path TO "$user", public`) fall back to
        # an opaque exp.Command. Match the leading setting name rather than
        # scanning the whole expression, so `SET ROLE my_search_path_role`
        # (whose value merely contains the substring) is not misclassified.
        parsed = self._parsed
        if isinstance(parsed, exp.Command) and parsed.name.upper() == "SET":
            tokens = str(parsed.expression).replace("=", " ").split()
            while tokens and tokens[0].upper() in {"SESSION", "LOCAL"}:
                tokens.pop(0)
            return bool(tokens) and tokens[0].strip('"').lower() == "search_path"
        return False

    def get_disallowed_tables(
        self,
        tables: set[str],
        default_schema: str | None = None,
        schema_indeterminate: bool = False,
    ) -> set[str]:
        """
        Return the subset of ``tables`` referenced by this statement.

        Matching mirrors :meth:`check_tables_present`: bare entries match by
        table name regardless of schema, while schema-qualified entries
        require the schema to match too. Entries are returned in their
        original denylist form so callers can report exactly which
        denylisted tables were hit.

        A reference without an explicit schema is resolved against
        ``default_schema`` when one is supplied, so an unqualified ``tables``
        run under ``search_path = information_schema`` still matches the
        ``information_schema.tables`` entry, while the same name under a
        user schema does not.

        :param tables: Set of table names to check for (case-insensitive)
        :param default_schema: Schema unqualified references resolve to at
            runtime (e.g. the session ``search_path`` / selected schema)
        :param schema_indeterminate: When True, the effective ``search_path``
            cannot be pinned to ``default_schema`` (e.g. the script contains a
            ``SET search_path``), so an unqualified reference is matched
            against the bare-name portion of schema-qualified entries too, to
            avoid bypassing a qualified denylist entry
        :return: The matched entries, in their original denylist form
        """
        fallback = default_schema.lower() if default_schema else None
        present_bare: set[str] = set()
        present_qualified: set[str] = set()
        present_unqualified: set[str] = set()
        for t in self.tables:
            bare = t.table.lower()
            present_bare.add(bare)
            if t.schema:
                present_qualified.add(f"{t.schema.lower()}.{bare}")
                # Also index the fully-qualified (catalog.schema.table) form so a
                # three-part denylist entry can match; without this, qualified
                # entries deeper than schema.table would silently never match.
                if t.catalog:
                    present_qualified.add(
                        f"{t.catalog.lower()}.{t.schema.lower()}.{bare}"
                    )
            else:
                present_unqualified.add(bare)
                # An unqualified reference can only be resolved against a known
                # default schema. When ``default_schema`` is None (the runtime
                # search_path is unknown to us), a qualified denylist entry is
                # matched only via the ``schema_indeterminate`` bare-name
                # fallback below, never here, this is an inherent limit of static
                # analysis without the live search_path.
                if fallback:
                    present_qualified.add(f"{fallback}.{bare}")
        found: set[str] = set()
        for entry in tables:
            needle = entry.lower()
            if "." in needle:
                if needle in present_qualified:
                    found.add(entry)
                elif (
                    schema_indeterminate
                    and needle.rsplit(".", 1)[1] in present_unqualified
                ):
                    found.add(entry)
            elif needle in present_bare:
                found.add(entry)
        return found

    def get_limit_value(self) -> int | None:
        """
        Parse a SQL query and return the `LIMIT` or `TOP` value, if present.
        """
        if limit_node := self._parsed.args.get("limit"):
            literal = limit_node.args.get("expression") or getattr(
                limit_node, "this", None
            )
            if isinstance(literal, exp.Literal) and literal.is_int:
                return int(literal.name)

        return None

    def set_limit_value(
        self,
        limit: int,
        method: LimitMethod = LimitMethod.FORCE_LIMIT,
    ) -> None:
        """
        Modify the `LIMIT` or `TOP` value of the SQL statement inplace.
        """
        if method == LimitMethod.FORCE_LIMIT:
            self._parsed.args["limit"] = exp.Limit(
                expression=exp.Literal(this=str(limit), is_string=False)
            )
        elif method == LimitMethod.WRAP_SQL:
            self._parsed = exp.Select(
                expressions=[exp.Star()],
                limit=exp.Limit(
                    expression=exp.Literal(this=str(limit), is_string=False)
                ),
                from_=exp.From(this=exp.Subquery(this=self._parsed.copy())),
            )
        else:  # method == LimitMethod.FETCH_MANY
            pass

    def has_cte(self) -> bool:
        """
        Check if the statement has a CTE.

        :return: True if the statement has a CTE at the top level.
        """
        return bool(self._parsed.args.get("with_"))

    def as_cte(self, alias: str = "__cte") -> SQLStatement:
        """
        Rewrite the statement as a CTE.

        This is needed by MS SQL when the query includes CTEs. In that case the CTEs
        need to be moved to the top of the query when we wrap it as a subquery when
        building charts.

        :param alias: The alias to use for the CTE.
        :return: A new SQLStatement with the CTE.
        """
        existing_ctes = self._parsed.args["with_"].expressions if self.has_cte() else []
        self._parsed.args["with_"] = None
        new_cte = exp.CTE(
            this=self._parsed.copy(),
            alias=exp.TableAlias(this=exp.Identifier(this=alias)),
        )
        return SQLStatement(
            ast=exp.With(expressions=[*existing_ctes, new_cte], this=None),
            engine=self.engine,
        )

    def as_create_table(self, table: Table, method: CTASMethod) -> SQLStatement:
        """
        Rewrite the statement as a `CREATE TABLE AS` statement.

        :param table: The table to create.
        :param method: The method to use for creating the table.
        :return: A new SQLStatement with the create table statement.
        """
        table_expr = exp.Table(
            this=exp.Identifier(this=table.table, quoted=True),
            db=exp.Identifier(this=table.schema, quoted=True) if table.schema else None,
            catalog=exp.Identifier(this=table.catalog, quoted=True)
            if table.catalog
            else None,
        )
        create_table = exp.Create(
            this=table_expr,
            kind=method.name,
            expression=self._parsed.copy(),
        )

        return SQLStatement(ast=create_table, engine=self.engine)

    def has_subquery(self) -> bool:
        """
        Check if the statement has a subquery.

        Covers explicit subqueries, set operations (``UNION``/``INTERSECT``/
        ``EXCEPT``), and any nested ``SELECT`` regardless of the top-level node
        type (e.g. when wrapped in parentheses or a set operation).

        :return: True if the statement has a subquery.
        """
        return (
            self.is_set_operation()
            or bool(self._parsed.find(exp.Subquery))
            or any(
                select != self._parsed for select in self._parsed.find_all(exp.Select)
            )
        )

    def is_set_operation(self) -> bool:
        """
        Check if the statement is a top-level set operation (UNION/INTERSECT/EXCEPT).
        """
        return isinstance(self._parsed, exp.SetOperation)

    def parse_predicate(self, predicate: str) -> exp.Expression:
        """
        Parse a predicate string into an AST.

        :param predicate: The predicate to parse.
        :return: The parsed predicate.
        """
        _check_script_length(predicate, self.engine)
        return sqlglot.parse_one(predicate, dialect=self._dialect)

    def apply_rls(
        self,
        catalog: str | None,
        schema: str | None,
        predicates: dict[Table, list[exp.Expression]],
        method: RLSMethod,
    ) -> None:
        """
        Apply relevant RLS rules to the statement inplace.

        :param catalog: The default catalog for non-qualified table names
        :param schema: The default schema for non-qualified table names
        :param method: The method to use for applying the rules.
        """
        if not predicates:
            return

        if method == RLSMethod.AS_SUBQUERY:
            # ``AS_SUBQUERY`` hoists the predicate into a filtered CTE on the outermost
            # query and rebinds each protected reference to it, so a CTE reference that
            # merely shares a base table's name is left untouched while the base read it
            # stands for is still filtered.

            def lookup(node: exp.Table) -> list[exp.Expression] | None:
                return predicates.get(
                    Table(
                        node.name,
                        node.db if node.db else schema,
                        node.catalog if node.catalog else catalog,
                    )
                )

            self._parsed = apply_rls_as_cte(self._parsed, lookup, dialect=self._dialect)
            return

        if method != RLSMethod.AS_PREDICATE:
            raise ValueError(f"Invalid RLS method: {method}")

        transformer = RLSAsPredicateTransformer(catalog, schema, predicates)
        self._parsed = self._parsed.transform(transformer)


class KQLSplitState(enum.Enum):
    """
    State machine for splitting a KQL script.

    The state machine keeps track of whether we're inside a string or not, so we
    don't split the script in a semi-colon that's part of a string.
    """

    OUTSIDE_STRING = enum.auto()
    INSIDE_SINGLE_QUOTED_STRING = enum.auto()
    INSIDE_DOUBLE_QUOTED_STRING = enum.auto()
    INSIDE_MULTILINE_STRING = enum.auto()


class KQLTokenType(enum.Enum):
    """
    Token types for KQL.
    """

    STRING = enum.auto()
    WORD = enum.auto()
    NUMBER = enum.auto()
    SEMICOLON = enum.auto()
    WHITESPACE = enum.auto()
    OTHER = enum.auto()


def classify_non_string_kql(text: str) -> list[tuple[KQLTokenType, str]]:
    """
    Classify non-string KQL.
    """
    tokens: list[tuple[KQLTokenType, str]] = []
    for m in re.finditer(r"[A-Za-z_][A-Za-z_0-9]*|\d+|\s+|.", text):
        tok = m.group(0)
        if tok == ";":
            tokens.append((KQLTokenType.SEMICOLON, tok))
        elif tok.isdigit():
            tokens.append((KQLTokenType.NUMBER, tok))
        elif re.match(r"[A-Za-z_][A-Za-z_0-9]*", tok):
            tokens.append((KQLTokenType.WORD, tok))
        elif re.match(r"\s+", tok):
            tokens.append((KQLTokenType.WHITESPACE, tok))
        else:
            tokens.append((KQLTokenType.OTHER, tok))

    return tokens


def tokenize_kql(kql: str) -> list[tuple[KQLTokenType, str]]:
    """
    Turn a KQL script into a flat list of tokens.
    """

    state = KQLSplitState.OUTSIDE_STRING
    tokens: list[tuple[KQLTokenType, str]] = []
    buffer = ""
    script = kql

    for i, ch in enumerate(script):
        if state == KQLSplitState.OUTSIDE_STRING:
            if ch in {"'", '"'}:
                if buffer:
                    tokens.extend(classify_non_string_kql(buffer))
                    buffer = ""
                state = (
                    KQLSplitState.INSIDE_SINGLE_QUOTED_STRING
                    if ch == "'"
                    else KQLSplitState.INSIDE_DOUBLE_QUOTED_STRING
                )
                buffer = ch
            elif ch == "`" and script[i - 2 : i] == "``":
                state = KQLSplitState.INSIDE_MULTILINE_STRING
                buffer = "```"
            else:
                buffer += ch
        else:
            buffer += ch
            end_str = (
                (
                    state == KQLSplitState.INSIDE_SINGLE_QUOTED_STRING
                    and ch == "'"
                    and script[i - 1] != "\\"
                )
                or (
                    state == KQLSplitState.INSIDE_DOUBLE_QUOTED_STRING
                    and ch == '"'
                    and script[i - 1] != "\\"
                )
                or (
                    state == KQLSplitState.INSIDE_MULTILINE_STRING
                    and ch == "`"
                    and script[i - 2 : i] == "``"
                )
            )
            if end_str:
                tokens.append((KQLTokenType.STRING, buffer))
                buffer = ""
                state = KQLSplitState.OUTSIDE_STRING

    if buffer:
        tokens.extend(classify_non_string_kql(buffer))

    return tokens


def split_kql(kql: str) -> list[str]:
    """
    Split a KQL script into statements on semicolons,
    ignoring those inside strings.
    """
    tokens = tokenize_kql(kql)
    stmts_tokens: list[list[tuple[KQLTokenType, str]]] = []
    current: list[tuple[KQLTokenType, str]] = []

    for ttype, val in tokens:
        if ttype == KQLTokenType.SEMICOLON:
            if current:
                stmts_tokens.append(current)
                current = []
        else:
            current.append((ttype, val))

    if current:
        stmts_tokens.append(current)

    return ["".join(val for _, val in stmt) for stmt in stmts_tokens]


class KustoKQLStatement(BaseSQLStatement[str]):
    """
    Special class for Kusto KQL.

    Kusto KQL is a SQL-like language, but it's not supported by sqlglot. Queries look
    like this:

        StormEvents
        | summarize PropertyDamage = sum(DamageProperty) by State
        | join kind=innerunique PopulationData on State
        | project State, PropertyDamagePerCapita = PropertyDamage / Population
        | sort by PropertyDamagePerCapita

    See https://learn.microsoft.com/en-us/azure/data-explorer/kusto/query/ for more
    details about it.
    """

    def __init__(
        self,
        statement: str | None = None,
        engine: str = "kustokql",
        ast: str | None = None,
    ):
        super().__init__(statement, engine, ast)

    @classmethod
    def split_script(
        cls,
        script: str,
        engine: str,
    ) -> list[KustoKQLStatement]:
        """
        Split a script at semi-colons.

        Since we don't have a parser, we use a simple state machine based function. See
        https://learn.microsoft.com/en-us/azure/data-explorer/kusto/query/scalar-data-types/string
        for more information.
        """
        return [
            cls(statement, engine, statement.strip()) for statement in split_kql(script)
        ]

    @classmethod
    def _parse_statement(
        cls,
        statement: str,
        engine: str,
    ) -> str:
        if engine != "kustokql":
            raise SupersetParseError(
                statement,
                engine,
                message=f"Invalid engine: {engine}",
            )

        statements = split_kql(statement)
        if len(statements) != 1:
            raise SupersetParseError(
                statement,
                engine,
                message="KustoKQLStatement should have exactly one statement",
            )

        return statements[0].strip()

    @classmethod
    def _extract_tables_from_statement(
        cls,
        parsed: str,
        engine: str,
    ) -> set[Table]:
        """
        Extract all tables referenced in the statement.

            StormEvents
            | where InjuriesDirect + InjuriesIndirect > 50
            | join (PopulationData) on State
            | project State, Population, TotalInjuries = InjuriesDirect + InjuriesIndirect

        """  # noqa: E501
        logger.warning(
            "Kusto KQL doesn't support table extraction. This means that data access "
            "roles will not be enforced by Superset in the database."
        )
        return set()

    def format(self, comments: bool = True) -> str:
        """
        Pretty-format the SQL statement.
        """
        return self._parsed.strip()

    def get_settings(self) -> dict[str, str | bool]:
        """
        Return the settings for the SQL statement.

            >>> statement = KustoKQLStatement("set querytrace;")
            >>> statement.get_settings()
            {"querytrace": True}

        """
        set_regex = r"^set\s+(?P<name>\w+)(?:\s*=\s*(?P<value>\w+))?$"
        if match := re.match(set_regex, self._parsed, re.IGNORECASE):
            return {match.group("name"): match.group("value") or True}

        return {}

    def is_select(self) -> bool:
        """
        Check if the statement is a `SELECT` statement.
        """
        return not self._parsed.startswith(".")

    def is_mutating(self) -> bool:
        """
        Check if the statement mutates data (DDL/DML).

        :return: True if the statement mutates data.
        """
        return self._parsed.startswith(".") and not self._parsed.startswith(".show")

    def is_destructive(self) -> bool:
        """
        Check if the statement is destructive DDL.

        Kusto KQL uses dot-commands for management operations. Destructive
        operations start with ``.drop`` or ``.alter``.

        :return: True if the statement is destructive DDL.
        """
        lower = self._parsed.lower()
        return lower.startswith(".drop") or lower.startswith(".alter")

    def optimize(self) -> KustoKQLStatement:
        """
        Return optimized statement.

        Kusto KQL doesn't support optimization, so this method is a no-op.
        """
        return KustoKQLStatement(ast=self._parsed, engine=self.engine)

    def check_functions_present(self, functions: set[str]) -> bool:
        """
        Check if any of the given functions are present in the script.

        :param functions: List of functions to check for
        :return: True if any of the functions are present
        """
        logger.warning("Kusto KQL doesn't support checking for functions present.")
        return False

    def check_tables_present(
        self, tables: set[str], default_schema: str | None = None
    ) -> bool:
        """
        Check if any of the given tables are present in the statement.

        :param tables: Set of table names to check for (case-insensitive)
        :param default_schema: Unused; accepted for interface parity
        :return: True if any of the tables are present
        """
        logger.warning("Kusto KQL doesn't support checking for tables present.")
        return False

    def get_disallowed_tables(
        self,
        tables: set[str],
        default_schema: str | None = None,
        schema_indeterminate: bool = False,
    ) -> set[str]:
        """
        Return the subset of ``tables`` referenced by this statement.

        :param tables: Set of table names to check for (case-insensitive)
        :param default_schema: Unused; accepted for interface parity
        :param schema_indeterminate: Unused; accepted for interface parity
        :return: The matched entries, in their original denylist form
        """
        logger.warning("Kusto KQL doesn't support checking for tables present.")
        return set()

    def get_limit_value(self) -> int | None:
        """
        Get the limit value of the statement.
        """
        tokens = [
            token
            for token in tokenize_kql(self._parsed)
            if token[0] != KQLTokenType.WHITESPACE
        ]
        for idx, (ttype, val) in enumerate(tokens):
            if ttype != KQLTokenType.STRING and val.lower() in {"take", "limit"}:
                if idx + 1 < len(tokens) and tokens[idx + 1][0] == KQLTokenType.NUMBER:
                    return int(tokens[idx + 1][1])
                break

        return None

    def set_limit_value(
        self,
        limit: int,
        method: LimitMethod = LimitMethod.FORCE_LIMIT,
    ) -> None:
        """
        Add a limit to the statement.
        """
        if method != LimitMethod.FORCE_LIMIT:
            raise SupersetParseError(
                self._parsed,
                self.engine,
                message="Kusto KQL only supports the FORCE_LIMIT method.",
            )

        tokens = tokenize_kql(self._parsed)
        found_limit_token = False
        for idx, (ttype, val) in enumerate(tokens):
            if ttype != KQLTokenType.STRING and val.lower() in {"take", "limit"}:
                found_limit_token = True

            if found_limit_token and ttype == KQLTokenType.NUMBER:
                tokens[idx] = (KQLTokenType.NUMBER, str(limit))
                break
        else:
            tokens.extend(
                [
                    (KQLTokenType.WHITESPACE, " "),
                    (KQLTokenType.WORD, "|"),
                    (KQLTokenType.WHITESPACE, " "),
                    (KQLTokenType.WORD, "take"),
                    (KQLTokenType.WHITESPACE, " "),
                    (KQLTokenType.NUMBER, str(limit)),
                ]
            )

        self._parsed = "".join(val for _, val in tokens)

    def parse_predicate(self, predicate: str) -> str:
        """
        Parse a predicate string into an AST.

        :param predicate: The predicate to parse.
        :return: The parsed predicate.
        """
        return predicate


class SQLScript:
    """
    A SQL script, with 0+ statements.
    """

    # Special engines that can't be parsed using sqlglot. Supporting non-SQL engines
    # adds a lot of complexity to Superset, so we should avoid adding new engines to
    # this data structure.
    special_engines = {
        "kustokql": KustoKQLStatement,
    }

    def __init__(
        self,
        script: str,
        engine: str,
    ):
        statement_class = self.special_engines.get(engine, SQLStatement)
        self.engine = engine
        self.statements = statement_class.split_script(script, engine)

    def format(self, comments: bool = True) -> str:
        """
        Pretty-format the SQL script.

        Note that even though KQL is very different from SQL, multiple statements are
        still separated by semi-colons.
        """
        return ";\n".join(statement.format(comments) for statement in self.statements)

    @property
    def has_unparseable_statement(self) -> bool:
        """
        True if any statement in the script cannot be fully modeled as an
        AST whose table references Superset can enumerate. This covers two
        cases that must both fail closed under strict scoping:

        * SQLGlot ``exp.Command`` nodes: statements sqlglot recognises but
          cannot fully parse (e.g. dynamic SQL inside a stored-procedure
          call); ``extract_tables_from_statement`` cannot see the tables.
        * Non-sqlglot engines (e.g. Kusto KQL): the statement class does
          not produce a sqlglot AST at all and its
          ``_extract_tables_from_statement`` returns an empty set, so the
          per-table check would have nothing to enforce against.
        """
        for statement in self.statements:
            if not isinstance(statement, SQLStatement):
                return True
            if isinstance(statement._parsed, exp.Command):  # noqa: SLF001
                return True
        return False

    def get_settings(self) -> dict[str, str | bool]:
        """
        Return the settings for the SQL script.

            >>> statement = SQLScript("SET foo = 'bar'; SET foo = 'baz'")
            >>> statement.get_settings()
            {"foo": "'baz'"}

        """
        settings: dict[str, str | bool] = {}
        for statement in self.statements:
            settings.update(statement.get_settings())

        return settings

    def has_mutation(self) -> bool:
        """
        Check if the script contains mutating statements.

        :return: True if the script contains mutating statements
        """
        return any(statement.is_mutating() for statement in self.statements)

    def has_destructive(self) -> bool:
        """
        Check if the script contains destructive DDL (DROP, TRUNCATE, ALTER).

        :return: True if any statement is destructive DDL.
        """
        return any(statement.is_destructive() for statement in self.statements)

    def optimize(self) -> SQLScript:
        """
        Return optimized script.
        """
        script = copy.deepcopy(self)
        script.statements = [  # type: ignore
            statement.optimize() for statement in self.statements
        ]

        return script

    def check_functions_present(self, functions: set[str]) -> bool:
        """
        Check if any of the given functions are present in the script.

        :param functions: List of functions to check for
        :return: True if any of the functions are present
        """
        return any(
            statement.check_functions_present(functions)
            for statement in self.statements
        )

    def check_tables_present(
        self, tables: set[str], default_schema: str | None = None
    ) -> bool:
        """
        Check if any of the given tables are present in the script.

        :param tables: Set of table names to check for (case-insensitive)
        :param default_schema: Schema unqualified references resolve to at
            runtime (e.g. the session ``search_path`` / selected schema)
        :return: True if any of the tables are present
        """
        return bool(self.get_disallowed_tables(tables, default_schema))

    def get_disallowed_tables(
        self, tables: set[str], default_schema: str | None = None
    ) -> set[str]:
        """
        Return the subset of ``tables`` referenced anywhere in the script.

        :param tables: Set of table names to check for (case-insensitive)
        :param default_schema: Schema unqualified references resolve to at
            runtime (e.g. the session ``search_path`` / selected schema)
        :return: The matched entries, in their original denylist form
        """
        # A `SET search_path` only affects statements that run *after* it, so
        # track the indeterminate state in statement order: an unqualified
        # reference is matched conservatively (against the bare-name portion of
        # qualified denylist entries) only once a preceding statement has
        # rebound the search path. This keeps a qualified denylist entry from
        # being bypassed (e.g. `SET search_path = information_schema; SELECT *
        # FROM tables`) without penalizing statements that ran beforehand.
        found: set[str] = set()
        schema_indeterminate = False
        for statement in self.statements:
            found |= statement.get_disallowed_tables(
                tables, default_schema, schema_indeterminate
            )
            if statement.changes_search_path():
                schema_indeterminate = True
        return found

    def is_valid_ctas(self) -> bool:
        """
        Check if the script contains a valid CTAS statement.

        CTAS (`CREATE TABLE AS SELECT`) can only be run with scripts where the last
        statement is a `SELECT`.
        """
        return self.statements[-1].is_select()

    def is_valid_cvas(self) -> bool:
        """
        Check if the script contains a valid CVAS statement.

        CVAS (`CREATE VIEW AS SELECT`) can only be run with scripts with a single
        `SELECT` statement.
        """
        return len(self.statements) == 1 and self.statements[0].is_select()


def extract_tables_from_statement(
    statement: exp.Expression,
    dialect: Dialects | None,
) -> set[Table]:
    """
    Extract all table references in a single statement.

    Please note that this is not trivial; consider the following queries:

        DESCRIBE some_table;
        SHOW PARTITIONS FROM some_table;
        WITH masked_name AS (SELECT * FROM some_table) SELECT * FROM masked_name;

    See the unit tests for other tricky cases.
    """
    sources: Iterable[exp.Table]

    if isinstance(statement, exp.Describe):
        # A `DESCRIBE` query has no sources in sqlglot, so we need to explicitly
        # query for all tables.
        sources = statement.find_all(exp.Table)
    elif isinstance(statement, exp.Command):
        # Commands, like `SHOW COLUMNS FROM foo`, have to be converted into a
        # `SELECT` statetement in order to extract tables.
        literal = statement.find(exp.Literal)
        if not literal:
            return set()

        pseudo_sql = f"SELECT {literal.this}"
        try:
            _check_script_length(pseudo_sql, None)
            pseudo_query = sqlglot.parse_one(pseudo_sql, dialect=dialect)
        except (ParseError, SupersetParseError):
            return set()
        sources = pseudo_query.find_all(exp.Table)
    else:
        # The same enumeration the AS_SUBQUERY rewrite filters against, so a table
        # authorised here is a table that rewrite hoists into a filtering CTE.
        sources = [source for source, _scope in _rls_real_reads(statement)]

    return {
        Table(
            source.name,
            source.db if source.db != "" else None,
            source.catalog if source.catalog != "" else None,
        )
        for source in sources
    }


def is_cte(source: exp.Table, scope: Scope) -> bool:
    """
    Does this reference resolve to a CTE rather than to a table?

    A CTE reference is an ``exp.Table`` too, so it has to be excluded from the tables a
    statement reads; otherwise a user with access to table `foo` could reach any table
    with a query like this:

        WITH foo AS (SELECT * FROM target_table) SELECT * FROM foo

    The name is resolved through ``Scope.cte_sources`` rather than compared, because it
    can match a CTE while the reference still resolves to a table: a qualified reference
    cannot name a CTE, and nor can a ``WITH`` item's reference to itself or to a later
    item unless ``RECURSIVE`` makes it legal. ``Scope.sources`` will not serve — keyed
    by ``alias_or_name``, it files an aliased CTE reference under the alias, so a real
    table sharing that alias would be taken for the CTE and dropped.

    Where sqlglot registers a name differently than SQL scopes it, this errs toward
    reporting a table: a reference differing from the CTE in letter case, and a
    ``WITH RECURSIVE`` item's reference to itself or to a later item outside a
    set-operation body. That costs an access check the statement does not need, and
    where the engine treats the reference as the CTE it also has the rewrite apply the
    rule to the CTE's projection, which the database rejects if the column is absent.
    Erring the other way: a read in a recursive item's base term is legal and is not
    reported.
    """
    if source.db or source.catalog:
        return False

    resolved = scope.cte_sources.get(source.name)
    return isinstance(resolved, Scope) and resolved.scope_type == ScopeType.CTE


T = TypeVar("T", str, None)


@dataclass
class JinjaSQLResult:
    """
    Result of processing Jinja SQL.

    Contains the processed SQL script and extracted table references.
    """

    script: SQLScript
    tables: set[Table]


def remove_quotes(val: T) -> T:
    """
    Helper that removes surrounding quotes from strings.
    """
    if val is None:
        return None

    if val[0] in {'"', "'", "`"} and val[0] == val[-1]:
        val = val[1:-1]

    return val


def process_jinja_sql(
    sql: str, database: Database, template_params: Optional[dict[str, Any]] = None
) -> JinjaSQLResult:
    """
    Process Jinja-templated SQL and extract table references.

    Due to Jinja templating, a multiphase approach is necessary as the Jinjafied SQL
    statement may represent invalid SQL which is non-parsable by SQLGlot.

    Firstly, we extract any tables referenced within the confines of specific Jinja
    macros. Secondly, we replace these non-SQL Jinja calls with a pseudo-benign SQL
    expression to help ensure that the resulting SQL statements are parsable by
    SQLGlot.

    :param sql: The Jinjafied SQL statement
    :param database: The database associated with the SQL statement
    :param template_params: Optional template parameters for Jinja templating
    :returns: JinjaSQLResult containing the processed script and table references
    :raises SupersetSecurityException: If SQLGlot is unable to parse the SQL statement
    :raises jinja2.exceptions.TemplateError: If the Jinjafied SQL could not be rendered
    """

    from superset.jinja_context import (  # pylint: disable=import-outside-toplevel
        get_template_processor,
    )

    processor = get_template_processor(database)
    ast = processor.env.parse(sql)

    tables = set()

    for node in ast.find_all(nodes.Call):
        if isinstance(node.node, nodes.Getattr) and node.node.attr in (
            "latest_partition",
            "latest_sub_partition",
        ):
            # Try to extract the table referenced in the macro.
            try:
                tables.add(
                    Table(
                        *[
                            remove_quotes(part.strip())
                            for part in node.args[0].as_const().split(".")[::-1]
                            if len(node.args) == 1
                        ]
                    )
                )
            except nodes.Impossible:
                pass

            # Replace the potentially problematic Jinja macro with some benign SQL.
            node.__class__ = nodes.TemplateData
            node.fields = nodes.TemplateData.fields
            node.data = "NULL"

    # re-render template back into a string
    code = processor.env.compile(ast)
    template = Template.from_code(processor.env, code, globals=processor.env.globals)
    rendered_sql = template.render(processor.get_context(), **(template_params or {}))

    parsed_script = SQLScript(
        processor.process_template(rendered_sql),
        engine=database.db_engine_spec.engine,
    )
    for parsed_statement in parsed_script.statements:
        tables |= parsed_statement.tables

    return JinjaSQLResult(script=parsed_script, tables=tables)


def sanitize_clause(clause: str, engine: str) -> str:
    """
    Validate a SQL clause and return it unchanged.

    The clause is parsed to ensure it is a single, well-formed statement. We
    intentionally return the *original* text rather than a re-rendered version:
    round-tripping user SQL through SQLGlot's dialect generator can silently
    alter semantics. For example, the Postgres dialect (borrowed by several
    engines) rewrites ``ROUND(AVG(x), n)`` to ``ROUND(CAST(AVG(x) AS DECIMAL),
    n)``, which rounds the value to an integer before the explicit ``ROUND`` on
    engines whose unqualified ``DECIMAL`` defaults to scale 0 (see #36113).

    Comments are the one exception: a trailing line comment can comment out
    surrounding SQL once the clause is embedded into a larger query (e.g.
    wrapped in parentheses), so any clause that contains comments is re-rendered
    to normalize them into a safe form. That re-rendering uses the *base* dialect
    rather than the engine dialect, so it normalizes comments without re-applying
    the engine-specific rewrites (e.g. the Postgres ``ROUND``/``CAST`` rewrite
    from #36113) that we deliberately avoid above. A trailing statement
    terminator is likewise stripped, since callers embed the clause inside a
    larger fragment (``WHERE (...)``) where a stray ``;`` would produce invalid
    SQL.
    """
    try:
        statement = SQLStatement(clause, engine)
        parsed = statement._parsed  # pylint: disable=protected-access
        if not any(node.comments for node in parsed.walk()):
            return clause.rstrip().rstrip(";").rstrip()

        return _normalized_generator(
            None,
            pretty=False,
            comments=True,
        ).generate(
            parsed,
            copy=True,
        )
    except SupersetParseError as ex:
        raise QueryClauseValidationException(f"Invalid SQL clause: {clause}") from ex


def transpile_to_dialect(
    sql: str,
    target_engine: str,
    source_engine: str | None = None,
    identify: bool = False,
) -> str:
    """
    Transpile SQL from one database dialect to another using SQLGlot.

    Args:
        sql: The SQL query to transpile
        target_engine: The target database engine (e.g., "mysql", "postgresql")
        source_engine: The source database engine. If None, uses generic SQL dialect.
        identify: If True, quote all identifiers per the target dialect.

    Returns:
        The transpiled SQL string

    If the target engine is not in SQLGLOT_DIALECTS, returns the SQL as-is.
    """
    target_dialect = SQLGLOT_DIALECTS.get(target_engine)

    # If no dialect mapping exists, return as-is
    if target_dialect is None:
        return sql

    # Get source dialect (default to generic if not specified)
    source_dialect = SQLGLOT_DIALECTS.get(source_engine) if source_engine else Dialect

    try:
        _check_script_length(sql, source_engine)
        parsed = sqlglot.parse_one(sql, dialect=source_dialect)
        return Dialect.get_or_raise(target_dialect).generate(
            parsed,
            copy=True,
            comments=False,
            pretty=False,
            identify=identify,
        )
    except ParseError as ex:
        raise QueryClauseValidationException(f"Cannot parse SQL clause: {sql}") from ex
    except Exception as ex:
        raise QueryClauseValidationException(
            f"Cannot transpile SQL to {target_engine}: {sql}"
        ) from ex
