
from sympy.external import import_module


if import_module('llvmlite'):
    import sympy.printing.llvmjitcode as g
    import numpy as np
else:
    disabled = True

import sympy
from sympy.abc import a,b,i,n

def test_simple_expr():
    e = a + 1.0
    f = g.get_jit_callable(e, [a])
    res = float(e.subs({a:4.0}).evalf())
    jit_res = f(4.0)

    assert np.isclose(jit_res, res)

def test_two_arg():
    e = 4.0*a + b + 3.0
    f = g.get_jit_callable(e, [a,b])
    res = float(e.subs({a:4.0, b:3.0}).evalf())
    jit_res = f(4.0, 3.0)

    assert np.isclose(jit_res, res)

def test_func():
    e = 4.0*sympy.exp(-a)
    f = g.get_jit_callable(e, [a])
    res = float(e.subs({a:1.5}).evalf())
    jit_res = f(1.5)

    assert np.isclose(jit_res, res)

def test_sum():
    a = sympy.IndexedBase('a')

    e = 5*sympy.Sum(2*a[i], (i,1,n))
    f = g.get_jit_callable(e, [a])

    actual_array = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
    # Is there a way to evaluated a Sympy sum over an array?
    res = 5*np.sum(2*actual_array)
    jit_res = f(actual_array)

    assert np.isclose(jit_res, res)

