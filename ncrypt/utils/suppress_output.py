import contextlib
import os


@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        # Save current file descriptors
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)

        # Redirect stdout and stderr to devnull
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)

        try:
            yield

        finally:
            # Restore stdout and stderr
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)
