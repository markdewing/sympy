from __future__ import print_function, division

'''
Use LLVMLite to create executable functions from Sympy expressions

  Prerequisites
  -------------
  LLVM - http://llvm.org/
  llvmlite - https://github.com/numba/llvmlite

Examples:

Scalar expression
-----------------
  import sympy.printing.llvmlitecode as g

  a = Symbol('a')
  e = a*a + a + 1
  e1 = g.get_jit_callable(e, args=[a])
  print(e1(1.1), e.subs('a',1.1))

Sum over a Numpy array
-----------------
  import sympy.printing.llvmlitecode as g
  import numpy as np

  a = IndexedBase('a')
  i = Symbol('i')
  n = Symbol('n')

  e = Sum(a[i], (i,1,n))

  concrete_array = np.ones(10)
  e1 = g.get_jit_callable(e,[a])
  print( e1(concrete_array), sum(concrete_array))

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

# Eventually need to encapsulate the memory storage and array access code
_pyobject_head = [ll.IntType(64), ll.PointerType(ll.IntType(32))]
_head_len = len(_pyobject_head)
_intp = ll.IntType(64)
_intp_star = ll.PointerType(_intp)
_void_star = ll.PointerType(ll.IntType(8))
_numpy_struct = ll.LiteralStructType(_pyobject_head+\
      [_void_star,          # data
       ll.IntType(32),     # nd
       _intp_star,          # dimensions
       _intp_star,          # strides
       _void_star,          # base
       _void_star,          # descr
       ll.IntType(32),     # flags
       _void_star,          # weakreflist
       _void_star,          # maskna_dtype
       _void_star,          # maskna_data
       _intp_star,          # masna_strides
      ])

_array_pointer = ll.PointerType(_numpy_struct)

class loop_creator(object):
    def __init__(self, builder, func):
        self._builder = builder
        self._func = func

    def start(self, loop_variable, start_value, end_value, step_value):
        self.end_value = end_value
        self.step_value = step_value
        self.pre_header_block = self._func.append_basic_block()
        self._builder.branch(self.pre_header_block)
        self.loop_block = self._func.append_basic_block('loop')
        self.exit_block = self._func.append_basic_block('afterloop')
        self._builder.position_at_end(self.pre_header_block)
        self._builder.branch(self.loop_block)

        self._builder.position_at_end(self.loop_block)

        self.index = self._builder.phi(self.int_type)
        self.index.add_incoming(start_value, self.pre_header_block)

    def add_next(self):
        self.next_value = self._builder.add(self.index, self.step_value, 'next')
        self.index.add_incoming(self.next_value, self.loop_block)

    def after(self):
        self.loop_end_block = self._builder.basic_block
        self.after_block = self._func.append_basic_block('afterloop')
        return self.after_block

    def finish(self):
        end_compare = self._builder.icmp_unsigned('<', self.next_value, self.end_value, 'loopcond')
        self._builder.cbranch(end_compare, self.loop_block, self.exit_block)
        self._builder.position_at_end(self.exit_block)



class LLVMJitPrinter(sympy.printing.printer.Printer):
    def __init__(self, *args, **kwargs):
        p = kwargs.pop('args',list())
        super(LLVMJitPrinter, self).__init__(*args, **kwargs)
        self.fp_type = ll.DoubleType()
        self.int_type = ll.IntType(64)

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

    def _print_Indexed(self, expr):
        return self.indexed_vars

    def _print_Sum(self, expr):
        #print('here in sum',expr.function)   # a[i]
        #print('here in sum, limits',expr.limits) # (i,1,n)

        # Need to create a loop over the size of the array,
        # access the elements of the array, and apply the function.
        # Also need to initialized summation variable

        loop = loop_creator(self.builder, self.fn)
        loop.int_type = self.int_type

        init_sum = ll.Constant(self.fp_type, 0.0)
        start_value = ll.Constant(self.int_type, 0)
        step_value = ll.Constant(self.int_type, 1)

        array = self.fn.args[0]
        # need to load from ndarray
        # assume ndim==1 and strides[0]==1 for now
        # dimensions[0] = size

        dim_ptr = self.builder.gep(array, [ll.Constant(ll.IntType(32),0), ll.Constant(ll.IntType(32), 4)])
        dim_value = self.builder.load(dim_ptr)

        end_value = self.builder.load(dim_value)

        loop_variable_name = 'tmp_idx'
        loop.start(loop_variable_name, start_value, end_value, step_value)

        accum = self.builder.phi(self.fp_type)
        accum.add_incoming(init_sum, loop.pre_header_block)

        base_data_ptr = self.builder.gep(array, [ll.Constant(ll.IntType(32),0), ll.Constant(ll.IntType(32), 2)])
        base_data = self.builder.load(base_data_ptr)
        fp_data_ptr = self.builder.bitcast(base_data, ll.PointerType(self.fp_type))
        data_offset_ptr = self.builder.gep(fp_data_ptr, [loop.index])
        data = self.builder.load(data_offset_ptr)

        # evaluate expr.function here
        #   need to store mapping of loaded array element ('data') to indexed expression
        self.indexed_vars = data
        summand = self._print(expr.function)

        added = self.builder.fadd(accum, summand)
        accum.add_incoming(added, loop.loop_block)

        loop.add_next()
        loop.finish()

        return added

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
    arg_types = []
    for arg in args:
        arg_type = fp_type
        if isinstance(arg, sympy.IndexedBase):
            arg_type = _array_pointer
        arg_types.append(arg_type)

    fn_type = ll.FunctionType(fp_type, arg_types)
    fn = ll.Function(module, fn_type, name="func_one")
    lj.module = module
    lj.param_dict = {}
    for i,a in enumerate(args):
        name = str(a)
        if isinstance(a, sympy.Indexed):
            name = str(a.base)
        fn.args[i].name = name
        lj.param_dict[name] = fn.args[i]
    bb_entry = fn.append_basic_block('entry')

    lj.builder = ll.IRBuilder(bb_entry)
    lj.fn = fn

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
    arg_ctypes = []
    for arg in args:
        arg_ctype = ctypes.c_double
        if isinstance(arg, sympy.IndexedBase):
            arg_ctype = ctypes.py_object
        arg_ctypes.append(arg_ctype)

    cfunc = ctypes.CFUNCTYPE(ctypes.c_double, *arg_ctypes)(fptr)
    return cfunc
