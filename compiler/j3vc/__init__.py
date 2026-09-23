"""j3vc: compile-time half of the J3v compiler (teacher labelling, head distillation).

Nothing in this package runs at inference time. The Rust `j3v` binary drives it, then
calibrates and conformance-checks the resulting artifact with its own int8 kernels.
"""
