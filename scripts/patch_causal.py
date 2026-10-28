import sys
from pathlib import Path


path = Path(sys.argv[1]) / "setup.py"
source = path.read_text()
marker = "    if FORCE_CXX11_ABI:\n"
replacement = (
    '    if not HIP_BUILD and os.getenv("TORCH_CUDA_ARCH_LIST") == "8.6":\n'
    '        cc_flag = ["-gencode", "arch=compute_86,code=sm_86"]\n\n'
    + marker
)
if replacement not in source:
    if source.count(marker) != 1:
        raise RuntimeError("Unexpected causal-conv1d build configuration")
    path.write_text(source.replace(marker, replacement))
