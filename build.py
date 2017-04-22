#!/usr/bin/env python3
from torch.utils.ffi import create_extension

ffi = create_extension(
        'torchwordemb._torchwordemb',
        headers='torchwordemb/src/loadwordemb.h',
        sources=['torchwordemb/src/loadwordemb.c'],
        with_cuda=False,
        package=True,
        relative_to = __file__,
        extra_compile_args=["--std=c99", "-Wall" ]
)
if __name__ == '__main__':
    ffi.build()
