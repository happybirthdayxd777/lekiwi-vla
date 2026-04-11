import sys, time, numpy as np
sys.path.insert(0, '/Users/i_am_ai/lerobot/src')

print("Testing imports...")
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy
print("Imports OK")

# Quick mock
class Mock:
    def predict(self, o):
        t = time.time()
        a = np.zeros(9, dtype=np.float32)
        a[0] = 0.5 * np.sin(t * 2 * np.pi)
        return a
    def reset(self): pass

m = Mock()
for i in range(3):
    a = m.predict({})
    print(f"step {i}: arm[0]={a[0]:.3f}")

print("All OK")