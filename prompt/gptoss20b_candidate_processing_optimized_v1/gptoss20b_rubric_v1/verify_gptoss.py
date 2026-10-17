"""Check GPT-OSS JD or candidate outputs without silently rewriting source evidence.

Install: pip install jsonschema
JD: python verify_gptoss.py jd --jd jd.txt --output rubric.json
Candidate: python verify_gptoss.py candidate --jd jd.txt --resume resume.txt --output assessment.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from jsonschema import Draft202012Validator
from rubric_validator import validate_rubric

ROOT=Path(__file__).resolve().parent

class DuplicateKeyError(ValueError): pass

def strict_json(raw: str):
    def no_duplicates(pairs):
        out={}
        for k,v in pairs:
            if k in out: raise DuplicateKeyError(f'duplicate JSON object key: {k!r}')
            out[k]=v
        return out
    return json.loads(raw,object_pairs_hook=no_duplicates)

def validate_candidate(jd: str,resume: str,data: dict):
    errors=[]
    schema=json.loads((ROOT/'candidate_assessment.schema.json').read_text(encoding='utf-8'))
    for e in Draft202012Validator(schema).iter_errors(data):
        errors.append(f'SCHEMA: {e.json_path}: {e.message}')
    if errors: return errors
    checks=data['checks']
    if data['status']=='unscorable':
        if checks or data['work'] is not None or data['role'] is not None or not data['review']:
            errors.append('UNSCORABLE_CONTRACT: checks must be empty, work/role null, review true')
        if data['years'] != {'required':None,'relevant':None,'certain':False}:
            errors.append('UNSCORABLE_YEARS: years must be null/null/false')
    else:
        if data['work'] is None or data['role'] is None:
            errors.append('SCORED_CONTRACT: work and role must be integers')
    expected_ids=[f'R{i}' for i in range(1,len(checks)+1)]
    if [c['id'] for c in checks]!=expected_ids:errors.append('CHECK_IDS: must be R1..RN without gaps')
    for c in checks:
        if not c['jd'] or c['jd'] not in jd:
            errors.append(f"JD_QUOTE: {c['id']} JD quote absent/not exact")
        if c['status'] in {'met','partial'}:
            if not isinstance(c['quote'],str) or not c['quote'] or c['quote'] not in resume:
                errors.append(f"RESUME_QUOTE: {c['id']} quote absent/not exact")
        elif c['quote'] is not None:
            errors.append(f"NULL_QUOTE: {c['id']} {c['status']} requires null quote")
    years=data['years']
    if years['required'] is None and (years['relevant'] is not None or years['certain']):
        errors.append('YEARS_ABSENT: required null demands relevant null/certain false')
    if years['relevant'] is None and years['certain']:
        errors.append('YEARS_UNCERTAIN: relevant null demands certain false')
    if not years['certain']:
        for c in checks:
            if c['kind']=='experience' and c['status']=='met' and years['required'] is not None:
                errors.append(f"YEARS_CONTRADICTION: {c['id']} experience met but years uncertain")
    if years['required'] is not None and years['relevant'] is not None and years['certain'] and years['relevant'] < years['required']:
        for c in checks:
            if c['kind']=='experience' and c['status']=='met':
                errors.append(f"YEARS_CONTRADICTION: {c['id']} experience met despite shortfall")
    if len(data['reason'].split())>35: errors.append('REASON_LENGTH: exceeds 35 words')
    return errors

def run(kind, jd, output, resume=None):
    raw=Path(output).read_text(encoding='utf-8')
    try: obj=strict_json(raw)
    except (json.JSONDecodeError, DuplicateKeyError) as e: return {'valid':False,'errors':[f'JSON: {e}']}
    if not isinstance(obj,dict):return {'valid':False,'errors':['ROOT: expected object']}
    source=Path(jd).read_text(encoding='utf-8')
    if kind=='jd':
        report=validate_rubric(source,obj)
        return {'valid':report.valid_for_cache,'errors':report.errors,'warnings':report.warnings,
                'coverage':report.coverage,'input_status':report.input_status}
    errors=validate_candidate(source,Path(resume).read_text(encoding='utf-8'),obj)
    return {'valid':not errors,'errors':errors}

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('kind',choices=['jd','candidate'])
    parser.add_argument('--jd',required=True)
    parser.add_argument('--resume')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.kind=='candidate' and not args.resume:parser.error('candidate requires --resume')
    result=run(args.kind,args.jd,args.output,args.resume)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(0 if result['valid'] else 1)
