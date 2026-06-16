import sys
import io
import signal
from contextlib import contextmanager
from typing import NamedTuple


class ExecutionResult(NamedTuple):
    success: bool
    output: str = ""
    error: str = ""


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Code execution timed out")


@contextmanager
def time_limit(seconds: int):
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
                '__builtins__': __builtins__,
            }
            exec(code, exec_globals)
        
        output = redirected_output.getvalue()
        error = redirected_error.getvalue()
        
        success = len(error) == 0
        
        return ExecutionResult(
            success=success,
            output=output,
            error=error
        )
        
    except TimeoutException as e:
        return ExecutionResult(
            success=False,
            output=redirected_output.getvalue(),
            error=f"Timeout: {str(e)}"
        )
    except Exception as e:
        return ExecutionResult(
            success=False,
            output=redirected_output.getvalue(),
            error=f"{type(e).__name__}: {str(e)}"
        )
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
