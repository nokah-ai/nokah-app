"""
nokah — Code Generator
Usage:
    python generate_codes.py starter 5   → generate 5 Starter codes (3 analyses each)
    python generate_codes.py pro 2        → generate 2 Pro codes (unlimited)
    python generate_codes.py list         → list all codes and their usage
"""

import json, random, string, sys, os

CODES_FILE = "nk_codes.json"

def load_codes():
    if not os.path.exists(CODES_FILE):
        return {}
    with open(CODES_FILE, "r") as f:
        return json.load(f)

def save_codes(codes):
    with open(CODES_FILE, "w") as f:
        json.dump(codes, f, indent=2)

def generate_code(plan, n=1, max_uses=3):
    codes = load_codes()
    new_codes = []
    for _ in range(n):
        suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        prefix = "NK-STARTER" if plan == "starter" else "NK-PRO"
        code = f"{prefix}-{suffix}"
        while code in codes:
            suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
            code = f"{prefix}-{suffix}"
        codes[code] = {
            "plan": plan,
            "uses": 0,
            "max_uses": max_uses if plan == "starter" else 999999,
            "active": True
        }
        new_codes.append(code)
    save_codes(codes)
    return new_codes

def list_codes():
    codes = load_codes()
    if not codes:
        print("No codes found.")
        return
    print(f"\n{'Code':<25} {'Plan':<10} {'Uses':<6} {'Max':<6} {'Status'}")
    print("-" * 60)
    for code, data in sorted(codes.items()):
        status = "✓ Active" if data.get("active") else "✗ Disabled"
        max_u = "∞" if data["plan"] == "pro" else str(data["max_uses"])
        remaining = "∞" if data["plan"] == "pro" else str(data["max_uses"] - data["uses"])
        print(f"{code:<25} {data['plan']:<10} {data['uses']:<6} {max_u:<6} {status} (remaining: {remaining})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "list":
        list_codes()

    elif cmd in ("starter", "pro"):
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        codes = generate_code(cmd, n)
        print(f"\n✓ Generated {len(codes)} {cmd.upper()} code(s):\n")
        for c in codes:
            print(f"  {c}")
        print()

    else:
        print(__doc__)
