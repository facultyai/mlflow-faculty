from enum import Enum
from uuid import UUID

import six
import sqlparse
from sqlparse.sql import (
    Comparison as SqlComparison,
    Identifier as SqlIdentifier,
    Statement as SqlStatement,
    Parenthesis as SqlParenthesis,
)
from sqlparse.tokens import Token as SqlTokenType


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

INVALID_IDENTIFIER_TPL = (
    "Invalid identifier {!r}. Expected identifier of format "
    "'attribute.run_id', 'attribute.status', 'metric.<key>', 'tag.<key>', "
    "or 'params.<key>'."
)
INVALID_OPERATOR_TPL = "{!r} is not a valid operator."
INVALID_VALUE_TPL = "Expected {} but found {!r}"


class _KeyType(Enum):
    RUN_ID = "run ID"
    STATUS = "status"
    PARAM = "parameter"
    METRIC = "metric"
    TAG = "tag"


ATTRIBUTE_IDENTIFIERS = {"attribute", "attributes", "attr", "run"}
RUN_ID_IDENTIFIERS = {"id", "run_id"}
PARAM_IDENTIFIERS = {"param", "params", "parameter", "parameters"}
METRIC_IDENTIFIERS = {"metric", "metrics"}
TAG_IDENTIFIERS = {"tag", "tags"}

COMPARISON_OPERATOR_MAPPING = {
    "=": ComparisonOperator.EQUAL_TO,
    "!=": ComparisonOperator.NOT_EQUAL_TO,
    ">": ComparisonOperator.GREATER_THAN,
    ">=": ComparisonOperator.GREATER_THAN_OR_EQUAL_TO,
    "<": ComparisonOperator.LESS_THAN,
    "<=": ComparisonOperator.LESS_THAN_OR_EQUAL_TO,
}
DISCRETE_KEY_TYPES = {_KeyType.RUN_ID, _KeyType.STATUS, _KeyType.TAG}
DISCRETE_OPERATORS = {
    ComparisonOperator.DEFINED,
    ComparisonOperator.EQUAL_TO,
    ComparisonOperator.NOT_EQUAL_TO,
}


class MatchesNothing(Exception):
    """Raised when a filter always matches nothing."""

    pass


def build_search_runs_filter(experiment_ids, filter_string, view_type):
    """Build a filter from the inputs to search_runs in the tracking store."""

    filter_parts = []

    if experiment_ids is not None:
        filter_parts.append(_filter_by_experiment_id(experiment_ids))

    deleted_at_filter = _filter_by_mlflow_view_type(view_type)
    if deleted_at_filter is not None:
        filter_parts.append(deleted_at_filter)

    if filter_string is not None and filter_string.strip() != "":
        filter_parts.append(_parse_filter_string(filter_string))

    if len(filter_parts) == 0:
        return None
    elif len(filter_parts) == 1:
        return filter_parts[0]
    else:
        return CompoundFilter(LogicalOperator.AND, filter_parts)


def _filter_by_experiment_id(experiment_ids):
    """Build a filter that a run is in one of a sequence of experiment IDs."""

    if len(experiment_ids) == 0:
        # Cannot build a filter for this
        raise MatchesNothing()

    parts = [
        ExperimentIdFilter(ComparisonOperator.EQUAL_TO, int(experiment_id))
        for experiment_id in experiment_ids
    ]

    if len(parts) == 1:
        return parts[0]
    else:
        return CompoundFilter(LogicalOperator.OR, parts)


def _filter_by_mlflow_view_type(view_type):
    """Build a filter for an MLflow view type.

    Parameters
    ----------
    view_type : mlflow.entities.ViewType

    Returns
    -------
    A Faculty library filter object, or None if no filter is to be applied.
    """
    if view_type == ViewType.ACTIVE_ONLY:
        return DeletedAtFilter(ComparisonOperator.DEFINED, False)
    elif view_type == ViewType.DELETED_ONLY:
        return DeletedAtFilter(ComparisonOperator.DEFINED, True)
    elif view_type == ViewType.ALL:
        return None
    else:
        raise ValueError("Invalid ViewType: {}".format(view_type))


def _parse_filter_string(mlflow_filter_string):
    """Parse an MLflow filter string into a Faculty filter object."""
    try:
        parsed = sqlparse.parse(mlflow_filter_string)
    except Exception:
        raise ValueError(
            "Error parsing filter '{}'".format(mlflow_filter_string)
        )

    try:
        [statement] = parsed
    except ValueError:
        raise ValueError(
            "Invalid filter '{}'. Must be a single statement.".format(
                mlflow_filter_string
            )
        )

    if not isinstance(statement, SqlStatement):
        raise ValueError(
            "Invalid filter '{}'. Must be a single statement.".format(
                mlflow_filter_string
            )
        )

    return _parse_token_list(statement.tokens)


def _parse_token_list(tokens):
    """Parse a list of sqlparse Tokens and return an equivalent filter."""

    # Ignore whitespace chars
    tokens = [t for t in tokens if not t.is_whitespace]

    if any(_is_or(t) for t in tokens):
        filters = []
        for part in _split_list(tokens, _is_or):
            filters.append(_parse_token_list(part))
        return CompoundFilter(LogicalOperator.OR, filters)

    elif any(_is_and(t) for t in tokens):
        filters = []
        for part in _split_list(tokens, _is_and):
            filters.append(_parse_token_list(part))
        return CompoundFilter(LogicalOperator.AND, filters)

    elif len(tokens) == 1:
        [token] = tokens
        if isinstance(token, SqlParenthesis):
            # Strip opening and closing parentheses
            return _parse_token_list(token.tokens[1:-1])
        elif isinstance(token, SqlComparison):
            return _parse_token_list(token.tokens)
        else:
            raise ValueError(
                "Unsupported filter string component: {!r}".format(
                    token.normalized
                )
            )

    elif len(tokens) == 3:
        return _single_filter_from_tokens(*tokens)

    else:
        raise ValueError(
            "Unsupported filter string component: {!r}".format(
                " ".join(t.normalized for t in tokens)
            )
        )


def _is_and(token):
    return token.match(ttype=SqlTokenType.Keyword, values=["AND"])


def _is_or(token):
    return token.match(ttype=SqlTokenType.Keyword, values=["OR"])


def _split_list(source_list, condition):
    chunk = []
    for value in source_list:
        if condition(value):
            yield chunk
            chunk = []
        else:
            chunk.append(value)
    yield chunk


def _single_filter_from_tokens(identifier_token, operator_token, value_token):
    key_type, key = _parse_identifier(identifier_token)
    operator = _parse_operator(operator_token)
    value = _parse_value(key_type, operator, value_token)

    _validate_operator(key_type, operator, value)

    if key_type == _KeyType.RUN_ID:
        return RunIdFilter(operator, value)
    elif key_type == _KeyType.STATUS:
        return RunStatusFilter(operator, value)
    elif key_type == _KeyType.PARAM:
        return ParamFilter(key, operator, value)
    elif key_type == _KeyType.METRIC:
        return MetricFilter(key, operator, value)
    elif key_type == _KeyType.TAG:
        return TagFilter(key, operator, value)
    else:
        raise Exception("Unexpected key_type")


def _parse_identifier(token):
    if not isinstance(token, SqlIdentifier):
        raise ValueError(INVALID_IDENTIFIER_TPL.format(token.value))

    try:
        key_type_string, key = token.value.split(".", 1)
    except ValueError:
        raise ValueError(INVALID_IDENTIFIER_TPL.format(token.value))

    key = _strip_quotes(key, ['"', "`"])

    if key_type_string in ATTRIBUTE_IDENTIFIERS:
        if key in RUN_ID_IDENTIFIERS:
            return _KeyType.RUN_ID, None
        elif key == "status":
            return _KeyType.STATUS, None
        else:
            raise ValueError(INVALID_IDENTIFIER_TPL.format(token.value))
    elif key_type_string in PARAM_IDENTIFIERS:
        return _KeyType.PARAM, key
    elif key_type_string in METRIC_IDENTIFIERS:
        return _KeyType.METRIC, key
    elif key_type_string in TAG_IDENTIFIERS:
        return _KeyType.TAG, key
    else:
        raise ValueError(INVALID_IDENTIFIER_TPL.format(token.value))


def _parse_operator(token):
    if token.match(ttype=SqlTokenType.Keyword, values=["IS"]):
        return ComparisonOperator.DEFINED
    elif token.ttype == SqlTokenType.Operator.Comparison:
        try:
            return COMPARISON_OPERATOR_MAPPING[token.value]
        except KeyError:
            raise ValueError(INVALID_OPERATOR_TPL.format(token.value))
    else:
        raise ValueError(INVALID_OPERATOR_TPL.format(token.value))


def _parse_value(key_type, operator, value_token):
    if operator == ComparisonOperator.DEFINED:
        return _extract_defined(value_token)
    elif key_type == _KeyType.RUN_ID:
        value_string = _extract_string(value_token)
        try:
            return UUID(value_string)
        except ValueError:
            raise ValueError(INVALID_VALUE_TPL.format("a UUID", value_string))
    elif key_type == _KeyType.STATUS:
        value_string = _extract_string(value_token)
        try:
            return ExperimentRunStatus(value_string.lower())
        except ValueError:
            valid_statuses = {
                status.value.upper() for status in ExperimentRunStatus
            }
            raise ValueError(
                INVALID_VALUE_TPL.format(
                    "a run status (one of {})".format(valid_statuses),
                    value_string,
                )
            )
    elif key_type == _KeyType.PARAM:
        return _extract_number_or_string(value_token)
    elif key_type == _KeyType.METRIC:
        return _extract_number(value_token)
    elif key_type == _KeyType.TAG:
        return _extract_string(value_token)
    else:
        raise Exception("Unexpected key_type")


def _validate_operator(key_type, operator, value):
    if key_type in DISCRETE_KEY_TYPES:
        if operator not in DISCRETE_OPERATORS:
            raise ValueError(
                "{} filters can only be used with operators '=', '!=' and "
                "'IS NULL'".format(key_type.value.capitalize())
            )
    elif key_type == _KeyType.PARAM and isinstance(value, six.string_types):
        if operator not in DISCRETE_OPERATORS:
            raise ValueError(
                "Param filters with string values can only be used with "
                "operators '=', '!=' and 'IS NULL'"
            )


def _extract_defined(token):
    if token.match(ttype=SqlTokenType.Keyword, values=["NULL"]):
        return False
    elif token.match(
        ttype=SqlTokenType.Keyword, values=["NOT +NULL"], regex=True
    ):
        return True
    else:
        raise ValueError(
            INVALID_VALUE_TPL.format("NULL or NOT NULL", token.value)
        )


def _extract_number(token):
    if token.ttype in SqlTokenType.Literal.Number:
        if token.ttype == SqlTokenType.Literal.Number.Integer:
            return int(token.value)
        else:
            return float(token.value)
    else:
        raise ValueError(INVALID_VALUE_TPL.format("a number", token.value))


def _extract_string(token):
    if token.ttype == SqlTokenType.Literal.String.Single or isinstance(
        token, SqlIdentifier
    ):
        return _strip_quotes(token.value, ['"', "'"], require_quotes=True)
    else:
        raise ValueError(
            INVALID_VALUE_TPL.format(
                "a quoted string (e.g. 'my-value')", token.value
            )
        )


def _extract_number_or_string(token):
    # Number takes priority
    try:
        return _extract_number(token)
    except ValueError:
        pass

    try:
        return _extract_string(token)
    except ValueError:
        # Don't raise new exception here to avoid linking it to the past one in
        # Python 3
        pass

    raise ValueError(
        INVALID_VALUE_TPL.format("a number or quoted string", token.value)
    )


def _strip_quotes(value, quotes, require_quotes=False):
    for char in quotes:
        if _is_quoted(value, char):
            return value[1:-1]

    if require_quotes:
        raise ValueError(
            INVALID_VALUE_TPL.format(
                "a string quoted with one of {}".format(set(quotes)), value
            )
        )
    else:
        return value


def _is_quoted(value, quote):
    return len(value) >= 2 and value[0] == quote and value[-1] == quote
