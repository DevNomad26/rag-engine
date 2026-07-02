import time
from contextlib import contextmanager


@contextmanager
def timer():
    """
    Context manager that measures elapsed time in milliseconds.

    Usage:
        with timer() as t:
            do_work()
        print(t())   # elapsed ms
    """
    start = time.perf_counter()
    elapsed = {}

    def get_elapsed():
        return elapsed.get("ms", 0)

    try:
        yield get_elapsed
    finally:
        elapsed["ms"] = int((time.perf_counter() - start) * 1000)