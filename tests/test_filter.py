from functools import partial
from uuid import uuid4

import pytest
from faculty.clients.experiment import (
    ComparisonOperator,
    CompoundFilter,
    DeletedAtFilter,
    ExperimentIdFilter,
    ExperimentRunStatus,
    LogicalOperator,
    MetricFilter,
    ParamFilter,
    RunIdFilter,
    RunStatusFilter,
    TagFilter,
)
from mlflow.entities import ViewType

import mlflow_faculty.filter
from mlflow_faculty.filter import (
    MatchesNothing,
    build_search_runs_filter,
    _filter_by_experiment_id,
    _filter_by_mlflow_view_type,
    _parse_filter_string,
)


def test_build_search_runs_filter(mocker):
    experiment_ids = [1, 2, 3]
    view_type = mocker.Mock()
    filter_string = "param.alpha > 0.2"

    mocker.patch("mlflow_faculty.filter._filter_by_experiment_id")
    mocker.patch("mlflow_faculty.filter._filter_by_mlflow_view_type")
    mocker.patch("mlflow_faculty.filter._parse_filter_string")

    expected_filter = CompoundFilter(
        LogicalOperator.AND,
        [
            mlflow_faculty.filter._filter_by_experiment_id.return_value,
            mlflow_faculty.filter._filter_by_mlflow_view_type.return_value,
            mlflow_faculty.filter._parse_filter_string.return_value,
        ],
    )

    filter = build_search_runs_filter(experiment_ids, filter_string, view_type)
    assert filter == expected_filter

    mlflow_faculty.filter._filter_by_experiment_id.assert_called_once_with(
        experiment_ids
    )
    mlflow_faculty.filter._filter_by_mlflow_view_type.assert_called_once_with(
        view_type
    )
    mlflow_faculty.filter._parse_filter_string.assert_called_once_with(
        filter_string
    )


def test_build_search_runs_filter_no_experiment_ids(mocker):
    experiment_ids = None
    view_type = mocker.Mock()
    filter_string = "param.alpha > 0.2"

    mocker.patch("mlflow_faculty.filter._filter_by_experiment_id")
    mocker.patch("mlflow_faculty.filter._filter_by_mlflow_view_type")
    mocker.patch("mlflow_faculty.filter._parse_filter_string")

    expected_filter = CompoundFilter(
        LogicalOperator.AND,
        [
            mlflow_faculty.filter._filter_by_mlflow_view_type.return_value,
            mlflow_faculty.filter._parse_filter_string.return_value,
        ],
    )

    filter = build_search_runs_filter(experiment_ids, filter_string, view_type)
    assert filter == expected_filter

    mlflow_faculty.filter._filter_by_experiment_id.assert_not_called()
    mlflow_faculty.filter._filter_by_mlflow_view_type.assert_called_once_with(
        view_type
    )
    mlflow_faculty.filter._parse_filter_string.assert_called_once_with(
        filter_string
    )


def test_build_search_runs_filter_no_view_type_filer(mocker):
    experiment_ids = [1, 2, 3]
    view_type = mocker.Mock()
    filter_string = "param.alpha > 0.2"

    mocker.patch("mlflow_faculty.filter._filter_by_experiment_id")
    mocker.patch(
        "mlflow_faculty.filter._filter_by_mlflow_view_type", return_value=None
    )
    mocker.patch("mlflow_faculty.filter._parse_filter_string")

    expected_filter = CompoundFilter(
        LogicalOperator.AND,
        [
            mlflow_faculty.filter._filter_by_experiment_id.return_value,
            mlflow_faculty.filter._parse_filter_string.return_value,
        ],
    )

    filter = build_search_runs_filter(experiment_ids, filter_string, view_type)
    assert filter == expected_filter

    mlflow_faculty.filter._filter_by_experiment_id.assert_called_once_with(
        experiment_ids
    )
    mlflow_faculty.filter._filter_by_mlflow_view_type.assert_called_once_with(
        view_type
    )
    mlflow_faculty.filter._parse_filter_string.assert_called_once_with(
        filter_string
    )


@pytest.mark.parametrize("filter_string", [None, "", " "])
def test_build_search_runs_filter_no_filter_string(mocker, filter_string):
    experiment_ids = [1, 2, 3]
    view_type = mocker.Mock()

    mocker.patch("mlflow_faculty.filter._filter_by_experiment_id")
    mocker.patch("mlflow_faculty.filter._filter_by_mlflow_view_type")
    mocker.patch("mlflow_faculty.filter._parse_filter_string")

    expected_filter = CompoundFilter(
        LogicalOperator.AND,
        [
            mlflow_faculty.filter._filter_by_experiment_id.return_value,
            mlflow_faculty.filter._filter_by_mlflow_view_type.return_value,
        ],
    )

    filter = build_search_runs_filter(experiment_ids, filter_string, view_type)
    assert filter == expected_filter

    mlflow_faculty.filter._filter_by_experiment_id.assert_called_once_with(
        experiment_ids
    )
    mlflow_faculty.filter._filter_by_mlflow_view_type.assert_called_once_with(
        view_type
    )
    mlflow_faculty.filter._parse_filter_string.assert_not_called()


def test_build_search_runs_filter_only_experiment_ids(mocker):
    experiment_ids = [1, 2, 3]
    view_type = mocker.Mock()
    filter_string = ""

    mocker.patch("mlflow_faculty.filter._filter_by_experiment_id")
    mocker.patch(
        "mlflow_faculty.filter._filter_by_mlflow_view_type", return_value=None
    )
    mocker.patch("mlflow_faculty.filter._parse_filter_string")

    expected_filter = (
        mlflow_faculty.filter._filter_by_experiment_id.return_value
    )

    filter = build_search_runs_filter(experiment_ids, filter_string, view_type)
    assert filter == expected_filter

    mlflow_faculty.filter._filter_by_experiment_id.assert_called_once_with(
        experiment_ids
    )
    mlflow_faculty.filter._filter_by_mlflow_view_type.assert_called_once_with(
        view_type
    )
    mlflow_faculty.filter._parse_filter_string.assert_not_called()


def test_build_search_runs_filter_only_view_type(mocker):
    experiment_ids = None
    view_type = mocker.Mock()
    filter_string = ""

    mocker.patch("mlflow_faculty.filter._filter_by_experiment_id")
    mocker.patch("mlflow_faculty.filter._filter_by_mlflow_view_type")
    mocker.patch("mlflow_faculty.filter._parse_filter_string")

    expected_filter = (
        mlflow_faculty.filter._filter_by_mlflow_view_type.return_value
    )

    filter = build_search_runs_filter(experiment_ids, filter_string, view_type)
    assert filter == expected_filter

    mlflow_faculty.filter._filter_by_experiment_id.assert_not_called()
    mlflow_faculty.filter._filter_by_mlflow_view_type.assert_called_once_with(
        view_type
    )
    mlflow_faculty.filter._parse_filter_string.assert_not_called()


def test_build_search_runs_filter_only_filter_string(mocker):
    experiment_ids = None
    view_type = mocker.Mock()
    filter_string = "param.alpha > 0.2"

    mocker.patch("mlflow_faculty.filter._filter_by_experiment_id")
    mocker.patch(
        "mlflow_faculty.filter._filter_by_mlflow_view_type", return_value=None
    )
    mocker.patch("mlflow_faculty.filter._parse_filter_string")

    expected_filter = mlflow_faculty.filter._parse_filter_string.return_value

    filter = build_search_runs_filter(experiment_ids, filter_string, view_type)
    assert filter == expected_filter

    mlflow_faculty.filter._filter_by_experiment_id.assert_not_called()
    mlflow_faculty.filter._filter_by_mlflow_view_type.assert_called_once_with(
        view_type
    )
    mlflow_faculty.filter._parse_filter_string.assert_called_once_with(
        filter_string
    )


def test_build_search_runs_filter_no_filters(mocker):
    experiment_ids = None
    view_type = mocker.Mock()
    filter_string = ""

    mocker.patch("mlflow_faculty.filter._filter_by_experiment_id")
    mocker.patch(
        "mlflow_faculty.filter._filter_by_mlflow_view_type", return_value=None
    )
    mocker.patch("mlflow_faculty.filter._parse_filter_string")

    filter = build_search_runs_filter(experiment_ids, filter_string, view_type)
    assert filter is None

    mlflow_faculty.filter._filter_by_experiment_id.assert_not_called()
    mlflow_faculty.filter._filter_by_mlflow_view_type.assert_called_once_with(
        view_type
    )
    mlflow_faculty.filter._parse_filter_string.assert_not_called()


@pytest.mark.parametrize(
    "experiment_ids, expected_filter",
    [
        ([1], ExperimentIdFilter(ComparisonOperator.EQUAL_TO, 1)),
        (
            [1, 2],
            CompoundFilter(
                LogicalOperator.OR,
                [
                    ExperimentIdFilter(ComparisonOperator.EQUAL_TO, 1),
                    ExperimentIdFilter(ComparisonOperator.EQUAL_TO, 2),
                ],
            ),
        ),
        (
            ["3", "4"],
            CompoundFilter(
                LogicalOperator.OR,
                [
                    ExperimentIdFilter(ComparisonOperator.EQUAL_TO, 3),
                    ExperimentIdFilter(ComparisonOperator.EQUAL_TO, 4),
                ],
            ),
        ),
    ],
)
def test_filter_by_experiment_id(experiment_ids, expected_filter):
    assert _filter_by_experiment_id(experiment_ids) == expected_filter


def test_filter_by_experiment_id_invalid_id():
    with pytest.raises(ValueError):
        _filter_by_experiment_id("not a valid integer")


def test_filter_by_experiment_id_empty_list():
    with pytest.raises(MatchesNothing):
        _filter_by_experiment_id([])


@pytest.mark.parametrize(
    "view_type, expected_filter",
    [
        (
            ViewType.DELETED_ONLY,
            DeletedAtFilter(ComparisonOperator.DEFINED, True),
        ),
        (
            ViewType.ACTIVE_ONLY,
            DeletedAtFilter(ComparisonOperator.DEFINED, False),
        ),
        (ViewType.ALL, None),
    ],
)
def test_filter_by_mlflow_view_type(view_type, expected_filter):
    assert _filter_by_mlflow_view_type(view_type) == expected_filter


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
    cases = []
    for sql_condition, defined in DEFINED_CASES:
        filter_string = "{} {}".format(identifier, sql_condition)
        expected_filter = builder(ComparisonOperator.DEFINED, defined)
        cases.append((filter_string, expected_filter))
    return cases


def _build_test_cases(builder, identifier, operator_cases, value_cases):
    cases = []
    for sql_operator, expected_operator in operator_cases:
        for sql_value, expected_value in value_cases:
            filter_string = "{} {} {}".format(
                identifier, sql_operator, sql_value
            )
            expected_filter = builder(expected_operator, expected_value)
            cases.append((filter_string, expected_filter))
    return cases


def _run_id_test_cases():
    cases = []
    for type_identifier in ATTRIBUTE_IDENTIFIERS:
        for sql_key in ["id", "run_id", '"id"', "`run_id`"]:
            identifier = "{}.{}".format(type_identifier, sql_key)
            cases += _build_defined_test_cases(RunIdFilter, identifier)
            cases += _build_test_cases(
                RunIdFilter, identifier, DISCRETE_OP_CASES, RUN_ID_CASES
            )
    return cases


def _status_test_cases():
    cases = []
    for type_identifier in ATTRIBUTE_IDENTIFIERS:
        for sql_key in ["status", '"status"', "`status`"]:
            identifier = "{}.{}".format(type_identifier, sql_key)
            cases += _build_defined_test_cases(RunStatusFilter, identifier)
            cases += _build_test_cases(
                RunStatusFilter, identifier, DISCRETE_OP_CASES, STATUS_CASES
            )
    return cases


def _param_test_cases():
    cases = []
    for type_identifier in ["param", "params", "parameter", "parameters"]:
        for sql_key, expected_key in KEY_CASES:
            identifier = "{}.{}".format(type_identifier, sql_key)
            filter_builder = partial(ParamFilter, expected_key)
            cases += _build_defined_test_cases(filter_builder, identifier)
            cases += _build_test_cases(
                filter_builder, identifier, DISCRETE_OP_CASES, STRING_CASES
            )
            cases += _build_test_cases(
                filter_builder, identifier, CONTINUOUS_OP_CASES, NUMBER_CASES
            )
    return cases


def _metric_test_cases():
    cases = []
    for type_identifier in ["metric", "metrics"]:
        for sql_key, expected_key in KEY_CASES:
            identifier = "{}.{}".format(type_identifier, sql_key)
            filter_builder = partial(MetricFilter, expected_key)
            cases += _build_defined_test_cases(filter_builder, identifier)
            cases += _build_test_cases(
                filter_builder, identifier, CONTINUOUS_OP_CASES, NUMBER_CASES
            )
    return cases


def _tag_test_cases():
    cases = []
    for type_identifier in ["tag", "tags"]:
        for sql_key, expected_key in KEY_CASES:
            identifier = "{}.{}".format(type_identifier, sql_key)
            filter_builder = partial(TagFilter, expected_key)
            cases += _build_defined_test_cases(filter_builder, identifier)
            cases += _build_test_cases(
                filter_builder, identifier, DISCRETE_OP_CASES, STRING_CASES
            )
    return cases


def _single_test_cases():
    cases = []
    cases += _run_id_test_cases()
    cases += _status_test_cases()
    cases += _param_test_cases()
    cases += _metric_test_cases()
    cases += _tag_test_cases()
    return cases


@pytest.mark.parametrize(
    "filter_string, expected_filter", _single_test_cases()
)
def test_parse_filter_string_single(filter_string, expected_filter):
    filter = _parse_filter_string(filter_string)
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
    filter = _parse_filter_string(filter_string)
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
    filter = _parse_filter_string(OPERATOR_PRECEDENCE_FILTER_STRING)
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
    filter = _parse_filter_string(filter_string)
    assert filter == expected_filter
    assert isinstance(filter, type(expected_filter))


@pytest.mark.parametrize(
    "filter_string",
    [
        "param = 'a string'",
        "alpha = 'a string'",
        "p.alpha = 'a string'",
        "attribute.unsupported = 'a string'",
    ],
)
def test_parse_filter_string_invalid_identifier(filter_string):
    with pytest.raises(ValueError, match="Invalid identifier"):
        _parse_filter_string(filter_string)


def test_parse_filter_string_unrecognised_operator():
    with pytest.raises(ValueError, match="is not a valid operator"):
        _parse_filter_string("param.alpha IN 'a string'")


@pytest.mark.parametrize(
    "filter_string",
    [
        "param.alpha IS 'a string'",
        "param.alpha = string",
        "metric.accuracy = 'a string'",
        "tag.`class.name` = 34",
        "run.id = NULL",
        "attr.status = NOT NULL",
        "run.id = 'not-a-uuid'",
        "attr.status = 'not-a-valid-status'",
        "param.alpha = `a string`",
    ],
)
def test_parse_filter_string_invalid_value(filter_string):
    with pytest.raises(ValueError, match="Expected .* but found .*"):
        _parse_filter_string(filter_string)


@pytest.mark.parametrize(
    "filter_string",
    [
        "run.id > '{}'".format(RUN_ID),
        "attr.status >= 'FINISHED'",
        "param.alpha < 'a string'",
        "tag.`class.name` <= 'a string'",
    ],
)
def test_parse_filter_string_invalid_operator(filter_string):
    with pytest.raises(ValueError, match="can only be used with operators"):
        _parse_filter_string(filter_string)
