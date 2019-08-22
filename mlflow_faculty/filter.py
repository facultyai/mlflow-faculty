from enum import Enum
from uuid import UUID

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
    ExperimentRunStatus,
    LogicalOperator,
    MetricFilter,
    ParamFilter,
    RunIdFilter,
    RunStatusFilter,
    TagFilter,
)

STRING_VALUE_SQL_TYPES = {SqlTokenType.Literal.String.Single}
NUMERIC_VALUE_SQL_TYPES = {
    SqlTokenType.Literal.Number.Integer,
    SqlTokenType.Literal.Number.Float,
}

INVALID_IDENTIFIER_TPL = (
    "Expected param, metric or tag identifier of format "
    "'metric.<key> <comparator> <value>', 'tag.<key> <comparator> <value>', "
    "or 'params.<key> <comparator> <value>' but found '{token}'."
)
INVALID_OPERATOR_TPL = "'{token}' is not a valid operator."


class _KeyType(Enum):
    RUN_ID = "run ID"
    STATUS = "status"
    PARAM = "parameter"
    METRIC = "metric"
    TAG = "tag"


ATTRIBUTE_IDENTIFIERS = {"attribute", "attributes", "attr", "run"}
RUN_ID_IDENTIFIERS = {"id", "run_id", "runId"}
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


# class ComparisonOperator(Enum):
#     DEFINED = "defined"
#     EQUAL_TO = "eq"
#     NOT_EQUAL_TO = "ne"
#     LESS_THAN = "lt"
#     LESS_THAN_OR_EQUAL_TO = "le"
#     GREATER_THAN = "gt"
#     GREATER_THAN_OR_EQUAL_TO = "ge"


# ProjectIdFilter = namedtuple("ProjectIdFilter", ["operator", "value"])
# ExperimentIdFilter = namedtuple("ExperimentIdFilter", ["operator", "value"])
# RunIdFilter = namedtuple("RunIdFilter", ["operator", "value"])
# DeletedAtFilter = namedtuple("DeletedAtFilter", ["operator", "value"])
# TagFilter = namedtuple("TagFilter", ["key", "operator", "value"])
# ParamFilter = namedtuple("ParamFilter", ["key", "operator", "value"])
# MetricFilter = namedtuple("MetricFilter", ["key", "operator", "value"])


# class LogicalOperator(Enum):
#     AND = "and"
#     OR = "or"


# CompoundFilter = namedtuple("CompoundFilter", ["operator", "conditions"])


def parse_filter_string(mlflow_filter_string):
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


def _is_and(token):
    return token.match(ttype=SqlTokenType.Keyword, values=["AND"])


def _is_or(token):
    return token.match(ttype=SqlTokenType.Keyword, values=["AND"])


def _split_list(source_list, condition):
    chunk = []
    for value in source_list:
        if condition(value):
            yield chunk
            chunk = []
        else:
            chunk.append(value)
    yield chunk


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


def _validate_operator(key_type, operator, value):
    if key_type in DISCRETE_KEY_TYPES:
        if operator not in DISCRETE_OPERATORS:
            raise ValueError(
                "{} filters can only be used with operators '=', '!=' and "
                "'IS NULL'".format(key_type.value.capitalize())
            )
    elif key_type == _KeyType.PARAM and isinstance(value, str):
        if operator not in DISCRETE_OPERATORS:
            raise ValueError(
                "Param filters with string values can only be used with "
                "operators '=', '!=' and 'IS NULL'"
            )


def _single_filter_from_tokens(identifier_token, operator_token, value_token):
    key_type, key = _parse_identifier(identifier_token)
    operator = _parse_operator(operator_token)

    if operator == ComparisonOperator.DEFINED:
        value = _extract_defined(value_token)
    elif key_type == _KeyType.RUN_ID:
        value_string = _extract_string(value_token)
        try:
            value = UUID(value_string)
        except ValueError:
            raise ValueError("{!r} is not a valid UUID".format(value_string))
    elif key_type == _KeyType.STATUS:
        value_string = _extract_string(value_token)
        try:
            value = ExperimentRunStatus(value_string.lower())
        except ValueError:
            raise ValueError(
                "{!r} is not a valid run status".format(value_string)
            )
    elif key_type == _KeyType.PARAM:
        value = _extract_number_or_string(value_token)
    elif key_type == _KeyType.METRIC:
        value = _extract_number(value_token)
    elif key_type == _KeyType.TAG:
        value = _extract_string(value_token)

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
        raise ValueError(INVALID_IDENTIFIER_TPL.format(token=token.value))

    try:
        key_type_string, key = token.value.split(".", 1)
    except ValueError:
        raise ValueError(INVALID_IDENTIFIER_TPL.format(token=token.value))

    key = _trim_backticks(_strip_quotes(key))

    if key_type_string in ATTRIBUTE_IDENTIFIERS:
        if key in RUN_ID_IDENTIFIERS:
            return _KeyType.RUN_ID, None
        elif key == "status":
            return _KeyType.STATUS, None
        else:
            raise ValueError("Unsupported filter attribute '{}'".format(key))
    elif key_type_string in PARAM_IDENTIFIERS:
        return _KeyType.PARAM, key
    elif key_type_string in METRIC_IDENTIFIERS:
        return _KeyType.METRIC, key
    elif key_type_string in TAG_IDENTIFIERS:
        return _KeyType.TAG, key
    else:
        raise ValueError("Unsupported filter '{}'.".format(token.value))


def _parse_operator(token):
    if token.match(ttype=SqlTokenType.Keyword, values=["IS"]):
        return ComparisonOperator.DEFINED
    elif token.ttype == SqlTokenType.Operator.Comparison:
        try:
            return COMPARISON_OPERATOR_MAPPING[token.value]
        except KeyError:
            raise ValueError(INVALID_OPERATOR_TPL.format(token=token.value))
    else:
        raise ValueError(INVALID_OPERATOR_TPL.format(token=token.value))


def _extract_defined(token):
    if token.match(ttype=SqlTokenType.Keyword, values=["NULL"]):
        return False
    elif token.match(
        ttype=SqlTokenType.Keyword, values=["NOT +NULL"], regex=True
    ):
        return True
    else:
        raise ValueError(
            "Expected NULL or NOT NULL but found {!r}".format(token.value)
        )


def _extract_number(token):
    if token.ttype in SqlTokenType.Literal.Number:
        if token.ttype == SqlTokenType.Literal.Number.Integer:
            return int(token.value)
        else:
            return float(token.value)
    else:
        raise ValueError(
            "Expected a number but found {!r}".format(token.value)
        )


def _extract_string(token):
    if token.ttype == SqlTokenType.Literal.String.Single or isinstance(
        token, SqlIdentifier
    ):
        return _strip_quotes(token.value, require_quotes=True)
    else:
        raise ValueError(
            "Expected a quoted string (e.g. 'my-value') "
            "but found {!r}".format(token.value)
        )


def _extract_number_or_string(token):
    #  Number takes priority
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
        "Expected a number or quoted string but found {!r}".format(token.value)
    )


def _strip_quotes(value, require_quotes=False):
    if _is_quoted(value, "'") or _is_quoted(value, '"'):
        return _trim_ends(value)
    elif require_quotes:
        raise ValueError(
            "Parameter value is either not quoted or unidentified quote "
            "types used for string {}. Use either single or double "
            "quotes.".format(value)
        )
    else:
        return value


def _trim_backticks(entity_type):
    """Remove backticks from identifier like `param`, if they exist."""
    if _is_quoted(entity_type, "`"):
        return _trim_ends(entity_type)
    return entity_type


def _is_quoted(value, pattern):
    return (
        len(value) >= 2
        and value.startswith(pattern)
        and value.endswith(pattern)
    )


def _trim_ends(string_value):
    return string_value[1:-1]
