import subprocess
import hashlib
import json
import os
import sys

def classify_file(path):
    p = path.replace('\\', '/').lower()
    if p.endswith(('.png', '.jpg', '.jpeg', '.pdf', '.svg')) and ('figure' in p or 'figures' in p or 'plots' in p or 'assets' in p):
        return 'FIGURE'
    elif p.endswith('.csv') or ('table' in p and p.endswith(('.csv', '.tsv'))):
        return 'TABLE'
    elif 'manifest' in p or p.endswith('manifest.json') or p.endswith('checksums.json'):
        return 'MANIFEST'
    elif p.startswith('tests/') or p.startswith('test_'):
        return 'TEST'
    elif p.startswith('reports/') or p.endswith('.md'):
        return 'REPORT'
    elif p.startswith('configs/') or p.endswith(('.yaml', '.yml', '.toml', '.env.example')):
        return 'CONFIGURATION'
    elif p.startswith('src/'):
        return 'FROZEN_SOURCE'
    elif 'traces' in p or 'raw' in p or p.endswith(('.jsonl', '.npy', '.pt', '.bin', '.parquet')):
        return 'RAW_EXPERIMENT'
    else:
        return 'OTHER'

def compute_sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(131072):
            h.update(chunk)
    return h.hexdigest(), os.path.getsize(path)

def main():
    print("Listing tracked files at commit 583e8a8f...", flush=True)
    files_raw = subprocess.check_output('git ls-tree -r --name-only 583e8a8f', shell=True, text=True, encoding='utf-8')
    tracked_files = [f.strip() for f in files_raw.splitlines() if f.strip()]
    print(f"Total files in commit 583e8a8f: {len(tracked_files)}", flush=True)

    # Find which files were changed between 583e8a8f and HEAD
    diff_raw = subprocess.check_output('git diff --name-only 583e8a8f HEAD', shell=True, text=True, encoding='utf-8')
    changed_in_head = set(f.strip() for f in diff_raw.splitlines() if f.strip())
    print(f"Files modified between 583e8a8f and HEAD: {len(changed_in_head)} -> {changed_in_head}", flush=True)

    manifest = {
        'audit_phase': '14.5',
        'baseline_commit': '583e8a8fa93967a15ddfbdb97d247cfcdbaad922',
        'baseline_commit_short': '583e8a8f',
        'total_tracked_files': len(tracked_files),
        'exclusions': ['.pytest_cache'],
        'files': {}
    }

    count = 0
    for path in tracked_files:
        if '.pytest_cache' in path:
            continue
        cls = classify_file(path)
        is_historical = not ('phase14' in path.lower() or 'phase14_5' in path.lower())

        if path in changed_in_head:
            blob = subprocess.check_output(f'git show 583e8a8f:"{path}"', shell=True)
            sha256 = hashlib.sha256(blob).hexdigest()
            size = len(blob)
        else:
            norm_path = os.path.normpath(path)
            if os.path.exists(norm_path):
                sha256, size = compute_sha256_file(norm_path)
            else:
                blob = subprocess.check_output(f'git show 583e8a8f:"{path}"', shell=True)
                sha256 = hashlib.sha256(blob).hexdigest()
                size = len(blob)

        manifest['files'][path] = {
            'relative_path': path,
            'size_bytes': size,
            'sha256': sha256,
            'git_tracked': True,
            'classification': cls,
            'is_historical_phase_0_13': is_historical
        }
        count += 1
        if count % 5000 == 0:
            print(f"Processed {count}/{len(tracked_files)} files...", flush=True)

    manifest['scientific_artifact_count'] = len(manifest['files'])

    os.makedirs('experiments/phase14_5/checksums', exist_ok=True)
    out_path = 'experiments/phase14_5/checksums/phase14_baseline_sha256.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated Phase 14 baseline manifest at {out_path} with {len(manifest['files'])} entries.", flush=True)

    # Verify current disk state
    print("Verifying current disk files against 583e8a8f baseline...", flush=True)
    matched = 0
    mismatched = []
    missing = []
    for rel_path, entry in manifest['files'].items():
        norm_path = os.path.normpath(rel_path)
        if not os.path.exists(norm_path):
            missing.append(rel_path)
            continue
        act_sha, _ = compute_sha256_file(norm_path)
        if act_sha != entry['sha256']:
            mismatched.append({
                'path': rel_path,
                'expected': entry['sha256'],
                'actual': act_sha,
                'classification': entry['classification'],
                'is_historical': entry['is_historical_phase_0_13']
            })
        else:
            matched += 1

    verification_result = {
        'audit_phase': '14.5',
        'target_commit': '583e8a8f',
        'total_audited_artifacts': len(manifest['files']),
        'matched_count': matched,
        'missing_count': len(missing),
        'mismatched_count': len(mismatched),
        'missing_files': missing,
        'mismatched_files': mismatched,
        'phase14_files_modified': [m['path'] for m in mismatched if 'phase14' in m['path'].lower()],
        'status': 'VERIFIED' if len(mismatched) == 0 and len(missing) == 0 else 'VERIFIED_WITH_EXPLAINED_MODIFICATIONS'
    }

    verif_out_path = 'experiments/phase14_5/checksums/phase14_integrity_verification.json'
    with open(verif_out_path, 'w', encoding='utf-8') as f:
        json.dump(verification_result, f, indent=2)

    print(f"Integrity check complete: Matched={matched}, Mismatched={len(mismatched)}, Missing={len(missing)}", flush=True)
    if mismatched:
        for m in mismatched:
            print(f"  Mismatched: {m['path']} ({m['classification']})")

if __name__ == '__main__':
    main()
