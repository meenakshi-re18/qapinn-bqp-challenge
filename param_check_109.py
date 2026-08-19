from config_b_direct_109 import ConfigDirect109
from model_b import build_model

model = build_model(ConfigDirect109)
n = model.count_parameters()
print(f"Direct 109 param count: {n}")
assert n == 109, "Mismatch — check CLASSICAL_HEAD_LAYERS in config_b_direct_109.py"