"""
Find exact mentions and origin of 66.81, 80.27, 84.47 across research logs.
"""
import os, glob

for p in sorted(glob.glob("research/*.md")):
    with open(p, 'r', encoding='utf-8') as f:
        content = f.read()
        if "66.81" in content or "80.27" in content or "84.47" in content:
            print(f"[*] Found in: {p}")
            for line in content.splitlines():
                if "66.81" in line or "80.27" in line or "84.47" in line:
                    print(f"    {line}")
