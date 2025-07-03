import asyncio
from functools import wraps, partial
from typing import Any, Callable, TypeVar, Awaitable, cast

F = TypeVar("F", bound=Callable[..., Any])


def async_wrap(func: F) -> Callable[..., Awaitable[Any]]:
    @wraps(func)
    async def run(
        *args: Any,
        loop: asyncio.AbstractEventLoop = None,
        executor: Any = None,
        **kwargs: Any,
    ) -> Any:
        if loop is None:
            loop = asyncio.get_event_loop()
        pfunc = partial(func, *args, **kwargs)
        return await loop.run_in_executor(executor, pfunc)

    return cast(Callable[..., Awaitable[Any]], run)
