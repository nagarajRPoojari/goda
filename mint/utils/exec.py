import io
import signal
import sys
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, NamedTuple, Never


class ExecutionResult(NamedTuple):
    success: bool
    output: str = ""
    error: str = ""


class TimeoutHandlerError(Exception):
    pass


def timeout_handler(signum, frame) -> Never:  # noqa: ANN001, ARG001
    raise TimeoutHandlerError("Code execution timed out")


@contextmanager
def time_limit(seconds: int) -> Generator[Any, Any, Any]:
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def execute_code(code: str, timeout: int = 5) -> ExecutionResult:

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    redirected_output = io.StringIO()
    redirected_error = io.StringIO()

    try:
        sys.stdout = redirected_output
        sys.stderr = redirected_error

        with time_limit(timeout):
            exec_globals = {
                "__builtins__": __builtins__,
            }
            exec(code, exec_globals)  # noqa: S102

        output = redirected_output.getvalue()
        error = redirected_error.getvalue()

        success = len(error) == 0

        return ExecutionResult(success=success, output=output, error=error)

    except TimeoutHandlerError as e:
        return ExecutionResult(
            success=False, output=redirected_output.getvalue(), error=f"Timeout: {e!s}"
        )
    except Exception as e:
        return ExecutionResult(
            success=False, output=redirected_output.getvalue(), error=f"{type(e).__name__}: {e!s}"
        )
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
