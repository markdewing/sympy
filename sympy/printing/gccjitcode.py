from __future__ import print_function, division

'''
Use GCC JIT to create executable functions from Sympy expressions

  Prerequisites
  -------------
  gccjit - https://gcc.gnu.org/wiki/JIT
  pygccjit - https://github.com/davidmalcolm/pygccjit


  Example
  -------
  import sympy.printing.gccjitcode as g

  a = Symbol('a')
  e = a*a + a + 1
  e1 = g.get_jit_callable(e,args=['a'])
  print(e1(1.1), e.subs('a',1.1))

'''


import sympy
import ctypes

from sympy.external import import_module

gccjit = import_module('gccjit')

class GCCJitPrinter(sympy.printing.printer.Printer):
    def __init__(self, *args, **kwargs):
        p = kwargs.pop('args',list())
        super(GCCJitPrinter, self).__init__(*args, **kwargs)
        self.ctxt = gccjit.Context()
        self.fp_type = self.ctxt.get_type(gccjit.TypeKind.DOUBLE)
        tmp_p = [(a,self.ctxt.new_param(self.fp_type, a)) for a in p]
        self.params = [a[1] for a in tmp_p]
        self.param_dict = dict(tmp_p)
        self.dummy_param = self.ctxt.new_param(self.fp_type,'dummy')

    def _print_Number(self, n, **kwargs):
        const = self.ctxt.new_rvalue_from_double(self.fp_type, n)
        return const

    def _print_Integer(self, expr):
        const = self.ctxt.new_rvalue_from_double(self.fp_type, expr.p)
        return const

    def _print_Symbol(self, s):
        # look up parameter with name s
        return self.param_dict.get(str(s))

    def _print_Pow(self, expr):
        base0 = self._print(expr.base)
        if expr.exp == sympy.S.NegativeOne:
            return self.ctxt.new_binary_op(gccjit.BinaryOp.DIVIDE,
                            self.fp_type,
                            self.ctxt.one(self.fp_type),
                            base0)
        if expr.exp == sympy.S.Half:
            sqrt_func = self.ctxt.new_function(gccjit.FunctionKind.IMPORTED,
                                        self.fp_type,
                                        "sqrt",
                                        [self.dummy_param])
            return self.ctxt.new_call(sqrt_func, [base0])
        if expr.exp == 2:
            return self.ctxt.new_binary_op(gccjit.BinaryOp.MULT,
                            self.fp_type,
                            base0,
                            base0)

        pow_func = self.ctxt.new_function(gccjit.FunctionKind.IMPORTED,
                                        self.fp_type,
                                        "pow",
                                        [self.param_a, self.dummy_param])
        exp0 = self._print(exp)
        return self.ctxt.new_call(pow_func, [base0, exp0])

    def _print_Mul(self, expr):
        nodes = [self._print(a) for a in expr.args]
        e = nodes[0]
        for node in nodes[1:]:
            e = self.ctxt.new_binary_op(gccjit.BinaryOp.MULT,
                                         self.fp_type,
                                         e,
                                         node)
        return e

    def _print_Add(self, expr):
        nodes = [self._print(a) for a in expr.args]
        e = nodes[0]
        for node in nodes[1:]:
            e = self.ctxt.new_binary_op(gccjit.BinaryOp.PLUS,
                                         self.fp_type,
                                         e,
                                         node)
        return e

    # TODO - assumes all called functions take one double precision argument.
    #        Should have a list of math library functions to validate this.
    def _print_Function(self, expr):
        name = expr.func.__name__
        e0 = self._print(expr.args[0])
        generic_func = self.ctxt.new_function(gccjit.FunctionKind.IMPORTED,
                                        self.fp_type,
                                        name,
                                        [self.param_a])
        return self.ctxt.new_call(generic_func, [e0])

def gcc_jit_code(expr, args=None):
    gj = GCCJitPrinter(args=args)

    # uncomment to get a C-like intermediate dump.  Useful for debugging.
    #gj.ctxt.set_bool_option(gccjit.BoolOption.DUMP_INITIAL_GIMPLE, True)

    gj.fn = gj.ctxt.new_function(gccjit.FunctionKind.EXPORTED, gj.fp_type, "func_one", gj.params)
    gj.block = gj.fn.new_block('entry')

    e = gj._print(expr)
    gj.block.end_with_return(e)
    jit_result = gj.ctxt.compile()
    return jit_result

def get_jit_callable(expr,args=None):
    '''Create an executable function from a Sympy expression'''
    jit_result = gcc_jit_code(expr,args)
    void_ptr = jit_result.get_code("func_one")
    doubles = [ctypes.c_double]*len(args)
    func_type = ctypes.CFUNCTYPE(ctypes.c_double, *doubles)
    jit_func = func_type(void_ptr)
    return jit_func
