import asyncio
import inspect
from functools import partial, wraps
from typing import Any, Callable

from typer import Typer


class AsyncTyper(Typer):
    @staticmethod
    def maybe_run_async(
        decorator: Callable[[Callable[..., Any]], None], f: Callable[..., Any]
    ) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(f):

            @wraps(f)
            def runner(*args: Any, **kwargs: Any) -> Any:
                return asyncio.run(f(*args, **kwargs))

            decorator(runner)
        else:
            decorator(f)
        return f

    def callback(
        self, *args: Any, **kwargs: Any
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        decorator = super().callback(*args, **kwargs)
        return partial(self.maybe_run_async, decorator)  # type: ignore

    def command(
        self, *args: Any, **kwargs: Any
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        decorator = super().command(*args, **kwargs)
        return partial(self.maybe_run_async, decorator)  # type: ignore
