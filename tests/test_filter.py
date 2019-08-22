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
        for sql_key in ["id", "run_id", "runId", '"id"', "`run_id`"]:
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
def test_single(filter_string, expected_filter):
    filter = parse_filter_string(filter_string)
    assert filter == expected_filter
    assert isinstance(filter, type(expected_filter))
