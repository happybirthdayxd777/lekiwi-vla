import sys
sys.path.insert(0, '/Users/i_am_ai/lerobot/src')

print("Testing factory imports...")
try:
    from lerobot.policies.factory import get_policy_class
    print("get_policy_class imported OK")
    cls = get_policy_class('act')
    print(f"ACTPolicy: {cls}")
    print("SUCCESS")
except Exception as e:
    print(f"FAILED: {e}")