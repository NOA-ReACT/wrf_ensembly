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
