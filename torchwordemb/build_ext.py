#!/usr/bin/env python3
from torch.utils.ffi import create_extension

ffi = create_extension(
        name='_torchwordemb',
        headers='loadwordemb.h',
        sources=['loadwordemb.c'],
        with_cuda=False,
        extra_compile_args=["--std=c99", "-Wall" ]
)
ffi.build()
