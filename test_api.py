"""
Test the heroku deployed salary classifier API
"""

import subprocess

print()
exit_code = subprocess.call("./test_api.sh")
print()
# print(f"exit code = {exit_code}")
