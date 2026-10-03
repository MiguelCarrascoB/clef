"""GPU + stack smoke test for clef-flash on WSL2 ROCm."""
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
p = torch.cuda.get_device_properties(0)
print("device:", p.name, "| total VRAM: %.1f GB" % (p.total_memory / 1e9))
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
y = (x @ x).mean()
torch.cuda.synchronize()
print("GPU matmul OK:", float(y))
import transformers, safetensors, PIL, fastapi  # noqa: E402

print("transformers:", transformers.__version__)
print("SMOKE TEST PASSED")
