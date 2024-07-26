import logging
import tempfile

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import meta_schedule as ms
from tvm.meta_schedule.testing.custom_builder_runner import run_module_via_rpc
from tvm.meta_schedule.testing.local_rpc import LocalRPC
from tvm.script import tir as T
from tvm.target import Target
from tvm.tir.schedule import BlockRV, Schedule

logging.basicConfig()
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)


@T.prim_func
def flash_atten(q: T.handle, k: T.handle, v: T.handle, o: T.handle) -> None:
    Q = T.match_buffer(q, [64, 4096, 12, 256])
    K = T.match_buffer(k, [64, 4096, 12, 256])
    V = T.match_buffer(v, [64, 4096, 12, 256])
    O = T.match_buffer(o, [64, 4096, 12, 12])

    S = T.alloc_buffer([64, 4096, 12, 12])
    for i, j, k, m, n in T.grid(64, 4096, 12, 12, 256):
        with T.block("flash_atten"):
            vi, vj, vk, vm, vn = T.axis.remap("SSSSR", [i, j, k, m, n])
            with T.init():
                S[vi, vj, vk, vm] = 0.0
            S[vi, vj, vk, vm] = O[vi, vj, vk, vm] + Q[vi, vj, vk, vn] * K[vi, vj, vm, vn]

    for i, j, m, n in T.grid(64, 4096, 12, 12):
        with T.block("norm"):
            vi, vj, vm, vn = T.axis.remap("SSSS", [i, j, m, n])
            O[vi, vj, vm, vn] = S[vi, vj, vm, vn] / 256

def test_tune_flash_atten_cuda():
    with tempfile.TemporaryDirectory() as work_dir:
        target = Target("nvidia/nvidia-a100")
        database = ms.tir_integration.tune_tir(
            mod=flash_atten,
            target=target,
            work_dir=work_dir,
            max_trials_global=32,
            num_trials_per_iter=16,
        )
        sch = ms.tir_integration.compile_tir(database, flash_atten, target)
        if sch is None:
            print("No valid schedule found!")
        else:
            sch.mod.show()
            sch.trace.show()


if __name__ == """__main__""":
    test_tune_flash_atten_cuda()