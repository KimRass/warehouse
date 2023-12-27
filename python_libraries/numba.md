# `@jit`
- Reference: https://numba.pydata.org/numba-doc/latest/user/5minguide.html
```python
# `nopython`: Set "nopython" mode for best performance, equivalent to `@njit`
# `parallel`: If `True`, enable the automatic parallelization of the function.
@jit(nopython=True)
```
- The Numba `@jit` decorator fundamentally operates in two compilation modes, `nopython` mode and `object` mode.
- he behaviour of the `nopython` compilation mode is to essentially compile the decorated function so that it will run entirely without the involvement of the Python interpreter. This is the recommended and best-practice way to use the Numba `jit` decorator as it leads to the best performance.
```python
# Function is compiled to machine code when called the first time
def function(...):
    ...
```
- Note that Pandas is not understood by Numba and as a result Numba would simply run this code via the interpreter but with the added cost of the Numba internal overheads!

# Types
- Reference: https://numba.pydata.org/numba-doc/latest/reference/types.html
- As an optimizing compiler, Numba needs to decide on the type of each variable to generate efficient machine code. Python’s standard types are not precise enough for that, so we had to develop our own fine-grained type system.
## Signatures
- A signature specifies the type of a function. Exactly which kind of signature is allowed depends on the context (AOT or JIT compilation), but signatures always involve some representation of Numba types to specify the concrete types for the function’s arguments and, if required, the function’s return type.
## Arrays
- The easy way to declare array types is to subscript an elementary type according to the number of dimensions. For example a 1-dimension single-precision array: