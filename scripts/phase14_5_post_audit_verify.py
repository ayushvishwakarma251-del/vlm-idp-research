import json
import hashlib
import os

def main():
    manifest_path = 'experiments/phase14_5/checksums/phase14_baseline_sha256.json'
    with open(manifest_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    matched = 0
    mismatched = []
    missing = []

    for rel_path, entry in data['files'].items():
        norm_path = os.path.normpath(rel_path)
        if not os.path.exists(norm_path):
            missing.append(rel_path)
            continue
        h = hashlib.sha256()
        with open(norm_path, 'rb') as fp:
            while chunk := fp.read(131072):
                h.update(chunk)
        act_sha = h.hexdigest()
        if act_sha != entry['sha256']:
            mismatched.append({'path': rel_path, 'expected': entry['sha256'], 'actual': act_sha})
        else:
            matched += 1

    total = len(data['files'])
    print(f"Post-audit frozen integrity verification:")
    print(f"Total: {total}")
    print(f"Matched: {matched}")
    print(f"Missing: {len(missing)}")
    print(f"Mismatched: {len(mismatched)}")
    if mismatched:
        for m in mismatched:
            print(f"  Mismatched: {m['path']}")

if __name__ == '__main__':
    main()
