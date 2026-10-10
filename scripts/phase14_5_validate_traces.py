import json
import csv
import math
from collections import Counter

def main():
    trace_file = 'experiments/phase14/traces/physical_inference_traces.jsonl'
    with open(trace_file, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]

    total_records = len(records)
    print(f"Total traces parsed: {total_records}")

    exp_ids = set()
    duplicates = []
    exclusions = []
    
    cond_counts = Counter()
    seed_counts = Counter()
    family_counts = Counter()
    doc_counts = Counter()
    query_counts = Counter()

    for idx, r in enumerate(records):
        eid = r.get('experiment_id')
        if eid in exp_ids:
            duplicates.append({'index': idx, 'experiment_id': eid})
            exclusions.append({'experiment_id': eid, 'reason': 'DUPLICATE_ID'})
        exp_ids.add(eid)

        cond_counts[r.get('condition')] += 1
        seed_counts[r.get('seed')] += 1
        family_counts[r.get('family_id')] += 1
        doc_counts[r.get('document_id')] += 1
        query_counts[r.get('query_id')] += 1

        # Check for invalid values
        for fld in ['physical_latency_s', 'peak_vram_mb', 'exact_match_accuracy', 'grounding_iou', 'unsupported_answer_rate', 'safe_useful_coverage']:
            val = r.get(fld)
            if val is None or math.isnan(val) or math.isinf(val):
                exclusions.append({'experiment_id': eid, 'reason': f'INVALID_{fld}_{val}'})
            elif val < 0.0:
                exclusions.append({'experiment_id': eid, 'reason': f'NEGATIVE_{fld}_{val}'})

    validation_summary = {
        'total_records': total_records,
        'valid_records': total_records - len(exclusions),
        'invalid_records': len(exclusions),
        'duplicate_records': len(duplicates),
        'conditions': dict(cond_counts),
        'seeds': dict(seed_counts),
        'unique_families': len(family_counts),
        'unique_documents': len(doc_counts),
        'unique_queries': len(query_counts),
        'schema_fields': list(records[0].keys()) if records else [],
        'suc_accuracy_concordance_rate': sum(1 for r in records if r.get('safe_useful_coverage') == r.get('exact_match_accuracy')) / total_records,
        'generation_mode': 'PARAMETRIC_SIMULATION_FROM_PHYSICAL_CALIBRATION'
    }

    with open('experiments/phase14_5/audit/trace_validation.json', 'w', encoding='utf-8') as f:
        json.dump(validation_summary, f, indent=2)

    with open('experiments/phase14_5/audit/trace_exclusions.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['experiment_id', 'reason'])
        writer.writeheader()
        for exc in exclusions:
            writer.writerow(exc)

    print("Generated trace_validation.json and trace_exclusions.csv")

if __name__ == '__main__':
    main()
