import numpy as np

from guppylang import guppy, comptime
from guppylang.std.builtins import array, result
from guppylang.std.quantum import cx, h, qubit, measure_array, collect_measurements

from hugr_qir.hugr_to_qir import hugr_to_qir
from hugr_qir.output import OutputFormat

@guppy
def main() -> None:
    qs = array(qubit() for _ in range(2))
    h(qs[0])
    cx(qs[0], qs[1])
    result("c", collect_measurements(measure_array(qs)))

hugr_module = main.compile()

qir_llvm_text = hugr_to_qir(hugr_module, output_format=OutputFormat.LLVM_IR)

print(qir_llvm_text)
