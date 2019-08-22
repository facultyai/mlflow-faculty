from functools import partial
from uuid import uuid4

import pytest
from faculty.clients.experiment import (
    ComparisonOperator,
    CompoundFilter,
    ExperimentRunStatus,
    LogicalOperator,
    MetricFilter,
    ParamFilter,
    RunIdFilter,
    RunStatusFilter,
    TagFilter,
)

from mlflow_faculty.filter import parse_filter_string


ATTRIBUTE_IDENTIFIERS = ["attribute", "attributes", "attr", "run"]

DEFINED_CASES = [
    ("IS NULL", False),
    ("IS NOT NULL", True),
    ("is null", False),
    ("is not null", True),
    ("IS  Null", False),
    ("IS   Not   Null", True),
]

DISCRETE_OP_CASES = [
    ("=", ComparisonOperator.EQUAL_TO),
    ("!=", ComparisonOperator.NOT_EQUAL_TO),
]

CONTINUOUS_OP_CASES = DISCRETE_OP_CASES + [
    (">", ComparisonOperator.GREATER_THAN),
    (">=", ComparisonOperator.GREATER_THAN_OR_EQUAL_TO),
    ("<", ComparisonOperator.LESS_THAN),
    ("<=", ComparisonOperator.LESS_THAN_OR_EQUAL_TO),
]

KEY_CASES = [
    ("alpha", "alpha"),
    ('"alpha-1"', "alpha-1"),
    ("`alpha.1`", "alpha.1"),
]

RUN_ID = uuid4()
RUN_ID_CASES = [
    ("'{}'".format(RUN_ID), RUN_ID),
    ('"{}"'.format(RUN_ID), RUN_ID),
]
STATUS_CASES = [
    ("'running'", ExperimentRunStatus.RUNNING),
    ('"finished"', ExperimentRunStatus.FINISHED),
    ("'failed'", ExperimentRunStatus.FAILED),
    ('"scheduled"', ExperimentRunStatus.SCHEDULED),
    ("'killed'", ExperimentRunStatus.KILLED),
    ('"Running"', ExperimentRunStatus.RUNNING),
    ("'FINISHED'", ExperimentRunStatus.FINISHED),
]
STRING_CASES = [("'value'", "value"), ('"value"', "value")]
NUMBER_CASES = [("2", 2), ("2.1", 2.1)]


def _build_defined_test_cases(builder, identifier):
    for sql_condition, defined in DEFINED_CASES:
        filter_string = "{} {}".format(identifier, sql_condition)
        expected_filter = builder(ComparisonOperator.DEFINED, defined)
        yield filter_string, expected_filter


def _build_test_cases(builder, identifier, operator_cases, value_cases):
    for sql_operator, expected_operator in operator_cases:
        for sql_value, expected_value in value_cases:
            filter_string = "{} {} {}".format(
                identifier, sql_operator, sql_value
            )
            expected_filter = builder(expected_operator, expected_value)
            yield filter_string, expected_filter


def _run_id_test_cases():
    for type_identifier in ATTRIBUTE_IDENTIFIERS:
        for sql_key in ["id", "run_id", '"id"', "`run_id`"]:
            identifier = "{}.{}".format(type_identifier, sql_key)
            yield from _build_defined_test_cases(RunIdFilter, identifier)
            yield from _build_test_cases(
                RunIdFilter, identifier, DISCRETE_OP_CASES, RUN_ID_CASES
            )


def _status_test_cases():
    for type_identifier in ATTRIBUTE_IDENTIFIERS:
        for sql_key in ["status", '"status"', "`status`"]:
            identifier = "{}.{}".format(type_identifier, sql_key)
            yield from _build_defined_test_cases(RunStatusFilter, identifier)
            yield from _build_test_cases(
                RunStatusFilter, identifier, DISCRETE_OP_CASES, STATUS_CASES
            )


def _param_test_cases():
    for type_identifier in ["param", "params", "parameter", "parameters"]:
        for sql_key, expected_key in KEY_CASES:
            identifier = "{}.{}".format(type_identifier, sql_key)
            filter_builder = partial(ParamFilter, expected_key)
            yield from _build_defined_test_cases(filter_builder, identifier)
            yield from _build_test_cases(
                filter_builder, identifier, DISCRETE_OP_CASES, STRING_CASES
            )
            yield from _build_test_cases(
                filter_builder, identifier, CONTINUOUS_OP_CASES, NUMBER_CASES
            )


def _metric_test_cases():
    for type_identifier in ["metric", "metrics"]:
        for sql_key, expected_key in KEY_CASES:
            identifier = "{}.{}".format(type_identifier, sql_key)
            filter_builder = partial(MetricFilter, expected_key)
            yield from _build_defined_test_cases(filter_builder, identifier)
            yield from _build_test_cases(
                filter_builder, identifier, CONTINUOUS_OP_CASES, NUMBER_CASES
            )


def _tag_test_cases():
    for type_identifier in ["tag", "tags"]:
        for sql_key, expected_key in KEY_CASES:
            identifier = "{}.{}".format(type_identifier, sql_key)
            filter_builder = partial(TagFilter, expected_key)
            yield from _build_defined_test_cases(filter_builder, identifier)
            yield from _build_test_cases(
                filter_builder, identifier, DISCRETE_OP_CASES, STRING_CASES
            )


def _single_test_cases():
    yield from _run_id_test_cases()
    yield from _status_test_cases()
    yield from _param_test_cases()
    yield from _metric_test_cases()
    yield from _tag_test_cases()


@pytest.mark.parametrize(
    "filter_string, expected_filter", _single_test_cases()
)
def test_parse_filter_string_single(filter_string, expected_filter):
    filter = parse_filter_string(filter_string)
    assert filter == expected_filter
    assert isinstance(filter, type(expected_filter))


RUN_ID_FILTER_STRING = 'attribute.run_id = "{}"'.format(RUN_ID)
RUN_ID_FILTER = RunIdFilter(ComparisonOperator.EQUAL_TO, RUN_ID)

STATUS_FILTER_STRING = 'run.status != "FINISHED"'
STATUS_FILTER = RunStatusFilter(
    ComparisonOperator.NOT_EQUAL_TO, ExperimentRunStatus.FINISHED
)

PARAM_FILTER_STRING = "param.alpha > 20"
PARAM_FILTER = ParamFilter("alpha", ComparisonOperator.GREATER_THAN, 20)

METRIC_FILTER_STRING = "metric.accuracy IS NOT NULL"
METRIC_FILTER = MetricFilter("accuracy", ComparisonOperator.DEFINED, True)

TAG_FILTER_STRING = "tag.`class.name` IS NULL"
TAG_FILTER = TagFilter("class.name", ComparisonOperator.DEFINED, False)

REDUCED_SINGLE_TEST_CASES = [
    (RUN_ID_FILTER_STRING, RUN_ID_FILTER),
    (STATUS_FILTER_STRING, STATUS_FILTER),
    (PARAM_FILTER_STRING, PARAM_FILTER),
    (METRIC_FILTER_STRING, METRIC_FILTER),
    (TAG_FILTER_STRING, TAG_FILTER),
]


@pytest.mark.parametrize(
    "sql_operator, expected_operator",
    [
        ("AND", LogicalOperator.AND),
        ("and", LogicalOperator.AND),
        ("OR", LogicalOperator.OR),
        ("or", LogicalOperator.OR),
    ],
)
@pytest.mark.parametrize("left_string, left_filter", REDUCED_SINGLE_TEST_CASES)
@pytest.mark.parametrize(
    "right_string, right_filter", REDUCED_SINGLE_TEST_CASES
)
def test_parse_filter_string_logical_operator(
    sql_operator,
    expected_operator,
    left_string,
    left_filter,
    right_string,
    right_filter,
):
    filter_string = "{} {} {}".format(left_string, sql_operator, right_string)
    expected_filter = CompoundFilter(
        expected_operator, [left_filter, right_filter]
    )
    filter = parse_filter_string(filter_string)
    assert filter == expected_filter
    assert isinstance(filter, type(expected_filter))


OPERATOR_PRECEDENCE_FILTER_STRING = "{} AND {} OR {}".format(
    RUN_ID_FILTER_STRING, STATUS_FILTER_STRING, METRIC_FILTER_STRING
)
OPERATOR_PRECEDENCE_FILTER = CompoundFilter(
    LogicalOperator.OR,
    [
        CompoundFilter(LogicalOperator.AND, [RUN_ID_FILTER, STATUS_FILTER]),
        METRIC_FILTER,
    ],
)


def test_parse_filter_string_operator_precedence():
    filter = parse_filter_string(OPERATOR_PRECEDENCE_FILTER_STRING)
    assert filter == OPERATOR_PRECEDENCE_FILTER
    assert isinstance(filter, type(OPERATOR_PRECEDENCE_FILTER))


PAREN_FILTER_STRING = "{} AND ({} OR {}) ".format(
    RUN_ID_FILTER_STRING, STATUS_FILTER_STRING, METRIC_FILTER_STRING
)
PAREN_FILTER = CompoundFilter(
    LogicalOperator.AND,
    [
        RUN_ID_FILTER,
        CompoundFilter(LogicalOperator.OR, [STATUS_FILTER, METRIC_FILTER]),
    ],
)

NESTED_PAREN_FILTER_STRING = "({} OR {} OR ({} AND {})) AND {}".format(
    RUN_ID_FILTER_STRING,
    STATUS_FILTER_STRING,
    PARAM_FILTER_STRING,
    METRIC_FILTER_STRING,
    TAG_FILTER_STRING,
)
NESTED_PAREN_FILTER = CompoundFilter(
    LogicalOperator.AND,
    [
        CompoundFilter(
            LogicalOperator.OR,
            [
                RUN_ID_FILTER,
                STATUS_FILTER,
                CompoundFilter(
                    LogicalOperator.AND, [PARAM_FILTER, METRIC_FILTER]
                ),
            ],
        ),
        TAG_FILTER,
    ],
)


@pytest.mark.parametrize(
    "filter_string, expected_filter",
    [
        ("({})".format(RUN_ID_FILTER_STRING), RUN_ID_FILTER),
        (" ( {} ) ".format(STATUS_FILTER_STRING), STATUS_FILTER),
        (PAREN_FILTER_STRING, PAREN_FILTER),
        (NESTED_PAREN_FILTER_STRING, NESTED_PAREN_FILTER),
    ],
)
def test_parse_filter_string_parentheses(filter_string, expected_filter):
    filter = parse_filter_string(filter_string)
    assert filter == expected_filter
    assert isinstance(filter, type(expected_filter))
