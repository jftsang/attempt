# attempt

This package provides some utilities for working with functions that
might raise exceptions.

## Example

```python
from attempt import Attempt

def f(x):
    # returns TypeError if x is a str
    return x + 1


xs = [1, 2, 3, "int", None, {}, 4]
ys = map(Attempt(f), xs)
assert list(values(ys, ignore_failures=True)) == [2, 3, 4, 5]
```
