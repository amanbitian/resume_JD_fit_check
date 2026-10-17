"""Compare two rubric extraction CSVs (the same jd_id set).

CSV contract: jd_id, jd, output, output_tokens, latency_seconds, finish_reason.
Prints operational / validation results, NOT human-ground-truth accuracy.
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from rubric_validator import parse_raw,validate_rubric

def read_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    return {r['jd_id']:r for r in rows}

def summarize(rows):
    stats={'rows':len(rows),'completed':0,'cache_valid':0,'schema_valid':0,'quote_errors':0,'open_category_errors':0,'generic_skill_errors':0,'logistics_review':0,'details':{}}
    for key,row in rows.items():
        errors=[];warnings=[]
        try:
            data=parse_raw(row.get('output',''),allow_fences=True)
            report=validate_rubric(row.get('jd',''),data)
            errors=report.errors;warnings=report.warnings
            stats['schema_valid']+=int(not any(e.startswith('SCHEMA:') for e in errors))
            stats['cache_valid']+=int(report.valid_for_cache and row.get('finish_reason')!='length')
            stats['completed']+=int(row.get('finish_reason')!='length')
            stats['quote_errors']+=sum(e.startswith(('ALTERED_QUOTE:','MISSING_QUOTE:')) for e in errors)
            stats['open_category_errors']+=sum(e.startswith(('MISSING_OPEN_','INVALID_OPEN_')) for e in errors)
            stats['generic_skill_errors']+=sum(e.startswith('MISSING_EXPLICIT_OTHER_SKILL:') for e in errors)
            stats['logistics_review']+=sum(w.startswith('LOGISTICS_POLICY_REVIEW:') for w in warnings)
        except Exception as exc:
            errors=[f'PARSE_ERROR: {str(exc)[:150]}']
        stats['details'][key]={'errors':errors,'warnings':warnings}
    def numeric_col(c):
        numbers=[]
        for r in rows.values():
            try:numbers.append(float(r[c]))
            except (KeyError,TypeError,ValueError):pass
        return round(mean(numbers),2) if numbers else None
    stats['mean_output_tokens']=numeric_col('output_tokens')
    stats['mean_latency_seconds']=numeric_col('latency_seconds')
    return stats

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--baseline',required=True)
    ap.add_argument('--challenger',required=True)
    ap.add_argument('--report',required=True)
    args=ap.parse_args()
    a=read_rows(args.baseline);b=read_rows(args.challenger)
    report={'baseline':summarize(a),'challenger':summarize(b),
            'common_ids':sorted(set(a)&set(b)),
            'only_baseline_ids':sorted(set(a)-set(b)),
            'only_challenger_ids':sorted(set(b)-set(a)),
            'note':'Validator agreement and performance are not independent semantic accuracy.'}
    Path(args.report).write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    for label in ('baseline','challenger'):
        x=report[label]
        print(f"{label}: {x['cache_valid']}/{x['rows']} validator-accepted, "
              f"{x['mean_output_tokens']} tokens, {x['mean_latency_seconds']} sec")
    print(f"Detailed comparison: {args.report}")
if __name__=='__main__':main()
