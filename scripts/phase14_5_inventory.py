import os
import json
import hashlib
import csv

def get_file_info(path):
    size = os.path.getsize(path)
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    sha256 = h.hexdigest()
    
    record_count = None
    schema_valid = True
    
    if path.endswith('.jsonl'):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                record_count = len(lines)
                for line in lines:
                    json.loads(line)
        except Exception:
            schema_valid = False
    elif path.endswith('.csv'):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
                record_count = max(0, len(rows) - 1)
        except Exception:
            schema_valid = False
    elif path.endswith('.json'):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                d = json.load(f)
                if isinstance(d, list):
                    record_count = len(d)
                elif isinstance(d, dict):
                    record_count = len(d)
        except Exception:
            schema_valid = False
            
    # Classify
    norm_path = path.replace('\\', '/')
    if '/tables/' in norm_path:
        art_type = 'TABLE'
        generated = 'CODE_GENERATED'
    elif '/figures/' in norm_path:
        art_type = 'FIGURE'
        generated = 'CODE_GENERATED'
    elif '/traces/' in norm_path:
        art_type = 'RAW_TRACE'
        generated = 'PHYSICAL_EXECUTION'
    elif '/models/' in norm_path:
        art_type = 'MODEL_WEIGHTS_CONFIG'
        generated = 'DOWNLOADED_PRETRAINED'
    elif norm_path.startswith('src/phase14'):
        art_type = 'SOURCE_CODE'
        generated = 'CODE_IMPLEMENTATION'
    elif norm_path.startswith('reports/phase14'):
        art_type = 'RESEARCH_REPORT'
        generated = 'MANUSCRIPT_REPORT'
    elif norm_path.startswith('tests/'):
        art_type = 'TEST_SUITE'
        generated = 'TEST_CODE'
    elif '/environment/' in norm_path:
        art_type = 'ENVIRONMENT_SPEC'
        generated = 'ENV_PROVENANCE'
    else:
        art_type = 'MANIFEST_OR_SCRIPT'
        generated = 'UTILITY'
        
    return {
        'path': norm_path,
        'artifact_type': art_type,
        'size_bytes': size,
        'sha256': sha256,
        'record_count': record_count,
        'schema_valid': schema_valid,
        'generated_by': generated,
        'reproducible_from_retained_inputs': art_type in ['TABLE', 'FIGURE', 'RESEARCH_REPORT']
    }

def main():
    targets = [
        'experiments/phase14',
        'src/phase14',
        'reports/phase14',
        'tests/test_phase14_systems.py'
    ]
    
    inventory = []
    
    for t in targets:
        if os.path.isfile(t):
            inventory.append(get_file_info(t))
        elif os.path.isdir(t):
            for root, dirs, files in os.walk(t):
                if '__pycache__' in root:
                    continue
                for f in files:
                    p = os.path.join(root, f)
                    inventory.append(get_file_info(p))
                    
    # Save JSON inventory
    os.makedirs('experiments/phase14_5/audit', exist_ok=True)
    json_path = 'experiments/phase14_5/audit/phase14_artifact_inventory.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(inventory, f, indent=2)
    print(f"Inventory saved to {json_path} with {len(inventory)} artifacts.")

    # Generate Markdown report
    md_path = 'reports/phase14_5/ARTIFACT_INVENTORY.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Phase 14 Complete Artifact Inventory\n\n")
        f.write("**Audit Phase**: Phase 14.5\n")
        f.write(f"**Total Tracked Phase 14 Artifacts**: {len(inventory)}\n\n")
        f.write("| Path | Type | Size (Bytes) | Records | Schema Valid | Origin |\n")
        f.write("|:---|:---|:---:|:---:|:---:|:---|\n")
        for item in inventory:
            rec_str = str(item['record_count']) if item['record_count'] is not None else "N/A"
            valid_str = "VALID" if item['schema_valid'] else "INVALID"
            f.write(f"| `{item['path']}` | {item['artifact_type']} | {item['size_bytes']:,} | {rec_str} | {valid_str} | {item['generated_by']} |\n")
    print(f"Markdown inventory saved to {md_path}")

if __name__ == '__main__':
    main()
