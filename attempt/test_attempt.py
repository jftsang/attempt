import unittest
from typing import Optional

from . import Attempted, Attempt, IterAttempted


class MyException(Exception):
    pass


class TestAttempted(unittest.TestCase):
    def test_ok(self):
        o = object()
        t = Attempted.of(o)
        self.assertIs(t.value, o)
        self.assertIsNone(t.exception)
        self.assertTrue(t)

    def test_exception(self):
        exc = MyException()
        t = Attempted.of_exception(exc)
        self.assertIs(exc, t.exception)
        with self.assertRaises(MyException):
            _ = t.value

        self.assertFalse(t)

    def test_map(self):
        def f(x):
            if x == 1:
                raise TypeError
            return x + 1

        def g(x):
            if x == 2:
                raise ValueError
            return x + 2

        self.assertIsInstance(Attempted.of(1).map(f).exception, TypeError)
        self.assertIsInstance(Attempted.of(1).map(f).map(g).exception, TypeError)
        self.assertEqual(Attempted.of(3).map(f).map(g).value, 6)


class TestAttempt(unittest.TestCase):
    def test_attempt(self):
        def fn(x: int | str) -> int:
            return x + 1  # pyre-ignore[58] raises if x is str

        t1 = Attempt(fn)(1)
        self.assertTrue(t1)
        self.assertEqual(t1.value, 2)

        t2 = Attempt(fn)("bad")
        self.assertFalse(t2)
        self.assertIsInstance(t2.exception, TypeError)

        with self.assertRaises(TypeError):
            # not catching TypeError
            Attempt(fn, ValueError, RuntimeError)("bad")


class TestIterAttempted(unittest.TestCase):
    def test_ok(self):
        o1 = object()
        o2 = object()
        t1 = Attempted.of(o1)
        t2 = Attempted.of(o2)
        ts = [t1, t2]
        self.assertListEqual(list(IterAttempted.values(ts)), [o1, o2])
        self.assertListEqual(
            list(IterAttempted.values(ts, ignore_failures=True)),
            [o1, o2]
        )

    def test_exception(self):
        o1 = object()
        exc = MyException()
        t1 = Attempted.of(o1)
        t2 = Attempted.of_exception(exc)
        ts = [t1, t2]
        with self.assertRaises(MyException):
            _ = list(IterAttempted.values(ts))

        self.assertListEqual(list(IterAttempted.values(ts, ignore_failures=True)), [o1])

    def test_map(self):
        def f(x):
            if x == 1:
                raise TypeError
            return x + 1

        def g(x):
            if x == 3:
                raise ValueError
            return x + 2

        xs = [10, 20, 30]
        ys = list(IterAttempted.from_values(xs).map(f).values())
        self.assertListEqual(ys, [11, 21, 31])
        zs = list(IterAttempted.from_values(xs).map(f).map(g).values())
        self.assertListEqual(zs, [13, 23, 33])

        xs = [1, 2, 3]
        ys = list(IterAttempted.from_values(xs).map(f).map(g))
        self.assertIsInstance(ys[0].exception, TypeError)
        self.assertIsInstance(ys[1].exception, ValueError)
        self.assertEqual(ys[2].value, 6)
