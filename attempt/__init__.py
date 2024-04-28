from __future__ import annotations

from functools import wraps
from typing import Callable, Final, Generic, Iterable, ParamSpec, Type, TypeVar

_P = ParamSpec("_P")
_T = TypeVar("_T", covariant=True)
_U = TypeVar("_U", covariant=True)


class Attempted(Generic[_T]):
    """Represents the result of trying to run a function that can either
    return a value (including None) or raise an exception.

    Unlike Java's `Option` or Haskell's `Maybe`, this class keeps track
    of exception information in addition to value information. It is
    more like a `Future` (and might better be called `Past`).
    """

    def __init__(self, value: _T | None, exception: Exception | None) -> None:
        self._value: Final[_T | None] = value
        self._exception: Final[Exception | None] = exception

    @classmethod
    def of(cls, value: _T) -> "Attempted[_T]":  # pyre-ignore[46]
        return cls(value, None)

    @classmethod
    def of_exception(cls, exception: Exception) -> "Attempted[None]":
        return cls(None, exception)  # pyre-ignore[7]

    def __repr__(self):
        if self:
            return f"Attempted.of({self._value!r})"
        else:
            return f"Attempted.of_exception({self.exception!r})"

    def __str__(self):
        if self:
            return f"Attempted.of({self._value!s})"
        else:
            return f"Attempted.of_exception({self.exception!s})"

    @property
    def value(self) -> _T | None:
        if self.exception is not None:
            raise self.exception

        return self._value

    @property
    def exception(self) -> Exception | None:
        return self._exception

    def __bool__(self):
        return self.exception is None

    def map(self, fn: Callable[[_T], _U]) -> "Attempted[_T | _U]":
        """If this `Attempted` contains a value, then try to apply `fn`
        to that value. If an exception is present, then keep the current
        exception.
        """
        if self:
            return Attempt(fn)(self.value)
        else:
            assert self._value is None
            return self

    def flatmap(self, fn: Callable[[_T | None], "Attempted[_U]"]) -> "Attempted[_T | _U]":
        """Similar to `map`, but fn is now an `Attempt`, i.e. it already
        returns an `Attempted`.
        """
        if self:
            return fn(self.value)
        else:
            return self


class IterAttempted(Iterable[Attempted[_T]]):
    """A stream of `Attempted` objects. This class provides methods for
    mapping and filtering.
    """

    def __init__(self, iterable: Iterable[Attempted[_T]]):
        self._iterable = iterable

    @classmethod
    def from_values(cls, xs: Iterable[_T]) -> "IterAttempted[_T]":
        return cls(map(Attempted.of, xs))

    def __iter__(self):
        for x in self._iterable:
            yield x

    def map(self, fn: Callable[[_T], _U]) -> "IterAttempted[_T | _U]":
        """Attempt to apply `fn` to each element in this iterable."""
        return self.__class__(map(lambda x: x.map(fn), self))

    def filter(self) -> "IterAttempted[_T]":
        """Keep only those elements in this iterable that are `True`,
        i.e. do not have an exception value.
        """
        return self.__class__(filter(bool, self))

    def values(self, ignore_failures: bool = False) -> Iterable[_T]:
        """Yields the values contained in the `Attempted` objects in this
        iterable. If `ignore_failures` is True, then any exceptions are
        ignored and only present values are yielded. Otherwise, if any
        one of the `Attempted` objects has an exception, then the exception
        is re-raised.
        """
        for t in self:
            if t:
                yield t.value
            else:
                if not ignore_failures:
                    raise t.exception


class Attempt:  # Callable[_P, Attempted[_T]]
    def __init__(self, fn: Callable[_P, _T], *exc_types: Type[Exception]):
        if not exc_types:
            exc_types = (Exception,)
        self._exc_types = exc_types

        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return Attempted.of(fn(*args, **kwargs))
            except self._exc_types as e:
                return Attempted.of_exception(e)

        self._fn = wrapper

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)
