from __future__ import print_function, division

'''
Use LLVMLite to create executable functions from Sympy expressions

  Prerequisites
  -------------
  LLVM - http://llvm.org/
  llvmlite - https://github.com/numba/llvmlite

Example -------
  import sympy.printing.llvmlitecode as g

  a = Symbol('a')
  e = a*a + a + 1
  e1 = g.get_jit_callable(e,args=['a'])
  print(e1(1.1), e.subs('a',1.1))

'''


import sympy
import ctypes

from sympy.external import import_module

ll = import_module('llvmlite.ir').ir
llvm = import_module('llvmlite.binding').binding

llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

target_machine = llvm.Target.from_default_triple().create_target_machine()
module = ll.Module()

class LLVMJitPrinter(sympy.printing.printer.Printer):
    def __init__(self, *args, **kwargs):
        p = kwargs.pop('args',list())
        super(LLVMJitPrinter, self).__init__(*args, **kwargs)
        self.fp_type = ll.DoubleType()

    def _print_Number(self, n, **kwargs):
        return ll.Constant(self.fp_type, float(n))

    def _print_Integer(self, expr):
        return ll.Constant(self.fp_type, float(expr.p))

    def _print_Symbol(self, s):
        # look up parameter with name s
        return self.param_dict.get(str(s))

    def _print_Pow(self, expr):
        base0 = self._print(expr.base)
        if expr.exp == sympy.S.NegativeOne:
            return self.builder.fdiv(ll.Constant(self.fp_type, 1.0), base0)
        if expr.exp == sympy.S.Half:
            fn_type = ll.FunctionType(self.fp_type, [self.fp_type])
            fn = ll.Function(self.module, fn_type, "sqrt")
            return self.builder.call(fn, [base0], "sqrt")
        if expr.exp == 2:
            return self.builder.fmul(base0, base0)

        exp0 = self._print(expr.exp)
        fn_type = ll.FunctionType(self.fp_type, [self.fp_type, self.fp_type])
        fn = ll.Function(self.module, fn_type, "pow")
        return self.builder.call(fn, [base0, exp0], "pow")

    def _print_Mul(self, expr):
        nodes = [self._print(a) for a in expr.args]
        e = nodes[0]
        for node in nodes[1:]:
            e = self.builder.fmul(e, node)
        return e

    def _print_Add(self, expr):
        nodes = [self._print(a) for a in expr.args]
        e = nodes[0]
        for node in nodes[1:]:
            e = self.builder.fadd(e, node)
        return e

    # TODO - assumes all called functions take one double precision argument.
    #        Should have a list of math library functions to validate this.
    def _print_Function(self, expr):
        name = expr.func.__name__
        e0 = self._print(expr.args[0])
        fn_type = ll.FunctionType(self.fp_type, [self.fp_type])
        fn = ll.Function(self.module, fn_type, name)
        return self.builder.call(fn, [e0], name)

# ensure lifetime of the execution engine persists (else call to compiled function will seg fault)
exe_eng = None

def llvm_jit_code(expr, args=None):
    global exe_eng
    lj = LLVMJitPrinter(args=args)

    fp_type = ll.DoubleType()
    fn_type = ll.FunctionType(fp_type, [fp_type]*len(args))
    fn = ll.Function(module, fn_type, "func_one")
    lj.module = module
    lj.param_dict = {}
    for i,a in enumerate(args):
        fn.args[i].name = a
        lj.param_dict[a] = fn.args[i]
    bb_entry = fn.append_basic_block('entry')
    lj.builder = ll.IRBuilder(bb_entry)

    ret = lj._print(expr)
    lj.builder.ret(ret)

    strmod = str(module)
    #print("LLVM IR")
    #print(strmod)

    llmod = llvm.parse_assembly(strmod)

    pmb = llvm.create_pass_manager_builder()
    pmb.opt_level = 2
    pass_manager = llvm.create_module_pass_manager()
    pmb.populate(pass_manager)

    pass_manager.run(llmod)

    exe_eng = llvm.create_mcjit_compiler(llmod, target_machine)
    exe_eng.finalize_object()

    #print("Assembly")
    #print(target_machine.emit_assembly(llmod))

    fptr = exe_eng.get_pointer_to_function(llmod.get_function("func_one"))

    return fptr


def get_jit_callable(expr, args=None):
    '''Create an executable function from a Sympy expression'''
    fptr = llvm_jit_code(expr, args)
    doubles = [ctypes.c_double]*len(args)
    cfunc = ctypes.CFUNCTYPE(ctypes.c_double, *doubles)(fptr)
    return cfunc
