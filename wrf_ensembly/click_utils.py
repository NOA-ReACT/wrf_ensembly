import datetime
import sys
from functools import wraps

import click


def pass_experiment_path(f):
    """
    Passes the `experiment_path` (pathlib.Path) variable from the context to the wrapped
    function, as the first argument.
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        ctx = click.get_current_context()
        return f(ctx.obj["experiment_path"], *args, **kwargs)

    return decorated_function


def print_start_end(f):
    """
    Prints a start and end message for the wrapped click command.
    The start message includes date, command name, and arguments.
    The end message includes date, duration, and command name.
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        cmd_name = f.__name__.replace("_", "-")
        cmd_args = " ".join(sys.argv[1:])

        start_time = datetime.datetime.now()
        print(f"Starting {cmd_name} at {start_time} with arguments: {cmd_args}")

        result = f(*args, **kwargs)

        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds() // 60
        print(f"Finished {cmd_name} at {end_time}. Duration: {duration} min")

        return result

    return decorated_function


class GroupWithStartEndPrint(click.Group):
    """
    A custom click group that prints start and end messages for each command.
    """

    def command(self, *args, **kwargs):
        # Get the decorator from the original command
        decorator = super(GroupWithStartEndPrint, self).command(*args, **kwargs)

        # Return a wrapper that applies our custom decorator
        def wrapper(f):
            # First apply the standard command decorator
            cmd = decorator(print_start_end(f))

            return cmd

        return wrapper
