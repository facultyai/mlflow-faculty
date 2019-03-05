import six
from pytz import UTC
from datetime import datetime

EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

def to_timestamp(dt):
    if six.PY2:
        return (dt - EPOCH).total_seconds()
    else:
        return dt.timestamp()

