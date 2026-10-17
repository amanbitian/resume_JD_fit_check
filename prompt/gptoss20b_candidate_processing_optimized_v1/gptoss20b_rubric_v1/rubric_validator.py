"""JD rubric v2.3 validator. Never invent or silently repair semantic requirements.

Dependencies: pip install jsonschema
Usage:
  python rubric_validator.py --jd job.txt --rubric output.json --report report.json
  python rubric_validator.py --eval-csv /mnt/data/jd_extraction_12.csv --report baseline.json

Exit code: 0 = rubric passes all blocking checks; 1 = errors or an audit with failures.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from jsonschema import Draft202012Validator

HERE=Path(__file__).resolve().parent
SCHEMA=json.loads((HERE/'jd_rubric.schema.json').read_text(encoding='utf-8'))
FORMAT_RE=re.compile(r'^```(?:json)?\s*\n?(.*?)\n?```\s*$',re.IGNORECASE|re.DOTALL)
ALIASES={'ci/cd':['ci/cd','ci cd','continuous integration/continuous delivery'],
         'rest apis':['rest apis','rest api','restful apis'],
         'node.js':['node.js','nodejs'],
         'postgresql':['postgresql','postgres'],
         'power bi':['power bi','powerbi'],
         'gcp':['gcp','google cloud platform'],
         'aws':['aws','amazon web services'],
         'javascript':['javascript'],
         'java':['java'],
         'python':['python'],
         'docker':['docker'],
         'jenkins':['jenkins'],
         'kubernetes':['kubernetes','k8s'],
         'terraform':['terraform'],
         'flask':['flask'],
         'react':['react','react.js'],
         'linux':['linux'],
         'azure':['azure'],
         'kafka':['kafka'],
         'pytorch':['pytorch'],
         'tensorflow':['tensorflow']}

@dataclass
class Report:
    errors:list[str]=field(default_factory=list)
    warnings:list[str]=field(default_factory=list)
    coverage:list[str]=field(default_factory=list)
    input_status:str|None=None
    def add(self,code,message,severity='error'):
        (self.errors if severity=='error' else self.warnings).append(f'{code}: {message}')
    @property
    def needs_rubric_review(self):
        return bool(self.errors or any(w.startswith(('OTHER_SKILLS_PRIORITY_REVIEW:', 'HARD_GATE_QUOTE_WEAK:', 'LOGISTICS_POLICY_REVIEW:')) for w in self.warnings))
    @property
    def valid_for_cache(self):
        return not self.needs_rubric_review and self.input_status in ("ok", "partial")
    @property
    def accepted_for_scoring(self): return self.valid_for_cache
    def serialize(self): return {'valid_for_cache':self.valid_for_cache, 'accepted_for_scoring':self.accepted_for_scoring, 'needs_rubric_review':self.needs_rubric_review, **asdict(self)}


def parse_raw(raw: str, *, allow_fences=False):
    raw=raw.strip()
    if allow_fences:
        match=FORMAT_RE.fullmatch(raw)
        if match: raw=match.group(1).strip()
    return json.loads(raw)


def contains_token(text:str, token:str)->bool:
    token=token.strip().lower()
    if not token:return False
    escaped=re.escape(token)
    return bool(re.search(r'(?<![\w])'+escaped+r'(?![\w])',text.lower()))


def canonical_mention(text:str, canonical:str)->bool:
    return any(contains_token(text,x) for x in ALIASES.get(canonical.lower(),[canonical.lower()]))


def extract_structured_field(jd:str, field_name:str)->str|None:
    """Heuristic coverage check for synthetic 'mustskills:'/'goodskills:' JDs.
    This is NOT a general JD parser or source of truth for semantic requirements.
    """
    field_names=['jobtitle','experience','jobdescription','education','mustskills','goodskills','otherskills']
    pattern=r'\b'+re.escape(field_name)+r'\s*["\']?\s*:\s*'
    match=re.search(pattern,jd,re.IGNORECASE)
    if not match:return None
    segment=jd[match.end():]
    stop=re.search(r',\s*["\']?(?:'+ '|'.join(re.escape(n) for n in field_names if n!=field_name)+r')["\']?\s*:',segment,re.IGNORECASE)
    if stop:segment=segment[:stop.start()]
    segment=segment.strip(' \t\n\r"\' ,')
    # Text accidentally concatenated to a skill list is not a skill.
    segment=re.split(r'\bthe ideal candidate\b|\bthe successful candidate\b',segment,flags=re.IGNORECASE)[0]
    return segment.strip(' \t\n\r"\' ,')


def listed_skills(jd:str, key:str)->list[str]:
    value=extract_structured_field(jd,key)
    if not value:return []
    return [x.strip(' \t\n\r"\' .') for x in re.split(r'[;,]',value) if x.strip(' \t\n\r"\' .')]


def group_by_id(rows:list[dict], prefix:str, rep:Report, path:str):
    ids=[row.get('id') for row in rows if isinstance(row,dict)]
    if ids!=[f'{prefix}{i}' for i in range(1,len(ids)+1)]:rep.add('ID_SEQUENCE',f'{path} ids must be {prefix}1..{prefix}{len(ids)} without gaps or duplicates')
    return {row['id']:row for row in rows if isinstance(row,dict) and isinstance(row.get('id'),str)}


ENROLLMENT_RE = re.compile(r"\b(?:currently\s+(?:enrolled|studying|pursuing)|enrolled\s+in|pursuing\s+(?:an?\s+)?(?:bachelor|degree)|studying\s+for\s+a\s+degree)\b", re.I)
COMPLETED_DEGREE_RE = re.compile(r"\b(?:completed|earned|obtained|awarded|graduated\s+with)\s+(?:an?\s+)?(?:bachelor|degree)|\b(?:must\s+hold|must\s+have)\s+(?:an?\s+)?(?:bachelor|degree)", re.I)
INFORMAL_WORK_RE = re.compile(r"\b(?:need(?:s)?\s+(?:someone\s+to\s+)?help\s+with|looking\s+for\s+(?:someone|help)|hire\s+(?:someone|a\s+person))\b",re.I)
WORK_AREA_RE = re.compile(r"\b(?:account|tax|bookkeep|payroll|web|design|software|develop|program|marketing|sales|clean|administrat|support|data|repair|landscap|teach|writing|writer)\w*",re.I)

def likely_informal_hiring(jd:str)->bool:
    return len(jd.strip())<=700 and bool(INFORMAL_WORK_RE.search(jd) and WORK_AREA_RE.search(jd))

def validate_rubric(jd:str, data:dict, *, strict_coverage=True)->Report:
    report=Report()
    structural=list(Draft202012Validator(SCHEMA).iter_errors(data))
    for e in sorted(structural, key=lambda x:str(list(x.absolute_path))):
        path='.'.join(map(str,e.absolute_path)) or '(root)'
        report.add('SCHEMA',f'{path}: {e.message}')
    if structural: return report # Cannot safely assess absent fields.

    report.input_status=data['input_status']
    if data['input_status'] not in ('ok','partial'):
        if data['input_status']=='not_a_jd' and likely_informal_hiring(jd):
            report.add('MISCLASSIFIED_INFORMAL_JD','Short text describes a plausible work/hiring need; review as partial')
        for key in ('skills','key_responsibilities','requirements','any_of_groups'):
            if data[key]:report.add('INVALID_INPUT_CONTENT',f'{key} must be empty when JD status is {data["input_status"]}')
        return report

    skills=group_by_id(data['skills'],'S',report,'skills')
    duties=group_by_id(data['key_responsibilities'],'D',report,'key_responsibilities')
    groups=group_by_id(data['any_of_groups'],'G',report,'any_of_groups')
    other=group_by_id(data['requirements'],'R',report,'requirements')

    for kind, rows in [('responsibility',data['key_responsibilities']),('skill',data['skills']),('requirement',data['requirements'])]:
        for r in rows:
            quote=r['source_quote']; rid=r['id']
            if not quote:
                report.add('MISSING_QUOTE',f'{rid} {kind}: source_quote is empty')
            elif quote not in jd:
                report.add('ALTERED_QUOTE',f'{rid} {kind}: source_quote does not occur verbatim in JD: {quote[:70]!r}')
            if '...' in quote and '...' not in jd:
                report.add('FABRICATED_ELLIPSIS',f'{rid}: ellipsis invented in quote')
            if len(quote)>160: report.add('QUOTE_TOO_LONG',rid)

    seen=set()
    for s in data['skills']:
        k=s['name'].casefold().strip()
        if not k:report.add('EMPTY_SKILL',s['id'])
        if k in seen:report.add('DUPLICATE_SKILL',f'{s["id"]}: {s["name"]!r}')
        seen.add(k)
    for item in [*data['skills'],*data['requirements']]:
        if item['hard_gate_explicit'] and item['priority_origin']!='explicit':
            report.add('INFERRED_HARD_GATE',f'{item["id"]}: hard gate requires explicit individual wording, not {item["priority_origin"]}')
        if item['hard_gate_explicit'] and item['priority']!='mandatory':
            report.add('HARD_GATE_NOT_MANDATORY',f'{item["id"]}: hard gate must have mandatory priority')
        if item['hard_gate_explicit'] and not re.search(r'\b(must|required|mandatory|essential|non.negotiable|minimum|shall|not\s+available|will\s+not\s+be\s+considered|no\s+sponsorship)\b',item['source_quote'],re.I):
            report.add('HARD_GATE_QUOTE_WEAK',f'{item["id"]}: quote lacks hard-minimum wording; inspect full JD context','warning')

    for fld in ('experience_requirement','education_requirement'):
        r=data[fld]
        if r['hard_gate_explicit'] and r['priority']!='mandatory':
            report.add('HARD_GATE_NOT_MANDATORY',f'{fld}: explicit hard gate requires mandatory priority')
    # Enrollment is not a completed-degree minimum in this legacy schema.
    if ENROLLMENT_RE.search(jd):
        enrollment_entries=[r for r in data['requirements'] if
                            r['category']=='other' and
                            re.search(r'\b(enroll|stud|pursu|degree\s+in\s+progress)\w*',r['requirement']+' '+r['source_quote'],re.I)]
        if not enrollment_entries:
            report.add('ENROLLMENT_CONDITION_MISSING','JD contains a current-enrollment condition; encode it in requirements as category=other')
        if data['education_requirement']['min_level'] is not None and not COMPLETED_DEGREE_RE.search(jd):
            report.add('ENROLLMENT_AS_COMPLETED_DEGREE','Current enrollment was incorrectly converted to a minimum completed degree')

    # `otherskills` is a catch-all, not inherently a mandatory heading.
    other_text=extract_structured_field(jd,'otherskills') or ''
    if other_text:
        must_text=extract_structured_field(jd,'mustskills') or ''
        for skill in data['skills']:
            source=skill['source_quote']
            if (skill['priority']=='mandatory' and source and
                source.lower() in other_text.lower() and
                not canonical_mention(must_text,skill['name']) and
                not re.search(r'\b(must|required|mandatory|essential)\b',source,re.I)):
                report.add('OTHER_SKILLS_PRIORITY_REVIEW',
                           f"{skill['id']} {skill['name']!r} marked mandatory from other-skills text without explicit item wording", 'warning')

    # Conservative semantic coverage: only check explicitly listed, recognizable
    # generic competencies, not vague personality adjectives or unstated skills.
    # This is purposefully a closed high-precision list rather than an ontology.
    generic_aliases={
        'problem solving': ('problem solving', 'problem-solving'),
        'agile methodology': ('agile methodology', 'agile methodologies', 'agile'),
        'team leadership': ('team leadership', 'leading teams', 'leadership'),
        'effective communication': ('effective communication', 'communication'),
        'technical writing': ('technical writing',),
        'stakeholder management': ('stakeholder management',),
        'time management': ('time management',),
    }
    for entry in re.split(r'[;,]', other_text):
        key=re.sub(r'\s+', ' ', re.sub(r'[-_]+', ' ', entry.strip(' \t\n\r\"\' .').lower())).strip()
        if key not in generic_aliases:
            continue
        matches=[skill for skill in data['skills']
                 if any(re.sub(r'[-_]+',' ',skill['name'].casefold())==alias or
                        re.sub(r'[-_]+',' ',skill['raw'].casefold())==alias
                        for alias in generic_aliases[key])]
        if not matches:
            report.add('MISSING_EXPLICIT_OTHER_SKILL',f'otherskills lists {entry.strip()!r} but skills does not contain it')
        elif all(skill['hard_gate_explicit'] for skill in matches):
            report.add('GENERIC_SKILL_HARD_GATE',f'{entry.strip()!r} must not be inferred as a hard gate from otherskills alone')

    # Catch a narrow, well-grounded open-example pattern. Do not invent open
    # categories merely because unrelated product names appear in the JD.
    # Example: "enterprise-level SaaS (Salesforce, ServiceNow, etc.)".
    for m in re.finditer(r'([^\n.!?]{0,110}?)\(([^()]{1,100})\)',jd,re.I):
        lead, examples=m.group(1),m.group(2)
        if not re.search(r'\b(?:saas|platforms?|tools?|frameworks?|software|systems?)\b',lead,re.I):
            continue
        if not re.search(r'\betc\.?\s*$|\be\.g\.',examples,re.I):
            continue
        members=[x.strip() for x in re.split(r'[,;]',examples)
                 if x.strip() and x.strip().casefold() not in ('etc.','etc','e.g.','e.g')]
        if len(members)<2 or len(members)>6: continue
        matched=[]
        for example in members:
            found=[skill for skill in data['skills'] if canonical_mention(skill['name'],example)
                   or canonical_mention(skill['raw'],example)]
            if not found: report.add('MISSING_OPEN_EXAMPLE',f'Open-category example {example!r} omitted from skills')
            else: matched.append(found[0])
        if len(matched)<2: continue
        gids={skill['any_of_group'] for skill in matched}
        if len(gids)!=1 or None in gids:
            report.add('MISSING_OPEN_ALTERNATIVE_GROUP',f'{members!r} must share one G# group')
        else:
            group=groups.get(next(iter(gids)))
            if not group or not group['open_category'] or group['min_required']!=1:
                report.add('INVALID_OPEN_ALTERNATIVE_GROUP',f'Group for {members!r} must be open, with min_required=1')

    # These explicitly worded policies are logistics, not technical skills.
    # The contract cannot itself prevent downstream score misuse, so the review
    # warning is blocking for ambiguous/incorrectly encoded policies.
    logistics_patterns=[
        ('work_authorization',r'\b(?:visa\s+)?sponsorship\s+(?:is\s+)?not\s+available\b'),
        ('location',r'\brelocation\s+will\s+not\s+be\s+considered\b'),
    ]
    for category,pattern in logistics_patterns:
        if not re.search(pattern,jd,re.I):continue
        entries=[r for r in data['requirements'] if r['category']==category and
                 re.search(pattern,r['source_quote'],re.I)]
        if not entries:
            report.add('MISSING_EXPLICIT_LOGISTICS',f'JD has explicit {category} policy but matching R# is absent')
        elif any(not r['hard_gate_explicit'] or r['priority_origin']!='explicit' for r in entries):
            report.add('LOGISTICS_POLICY_REVIEW',f'Explicit {category} policy should preserve non-negotiable wording; do not use as a technical-fit gate','warning')

    exp=data['experience_requirement']
    if exp['min_years'] is not None and exp['max_years'] is not None and exp['max_years']<exp['min_years']:
        report.add('INVALID_EXPERIENCE_RANGE','max_years is lower than min_years')

    for s in data['skills']:
        gid=s['any_of_group']
        if gid is not None:
            if gid not in groups:report.add('UNKNOWN_GROUP',f'{s["id"]} references {gid}')
            elif s['id'] not in groups[gid]['member_skill_ids']:
                report.add('MISSING_GROUP_MEMBER',f'{gid} does not list {s["id"]}')
    for g in data['any_of_groups']:
        members=g['member_skill_ids']
        if len(members)!=len(set(members)):report.add('DUPLICATE_GROUP_MEMBER',g['id'])
        if g['min_required']>len(members):report.add('IMPOSSIBLE_GROUP',f'{g["id"]}: min_required exceeds members')
        if g['open_category'] and not g['category_label']:
            report.add('MISSING_CATEGORY',f'{g["id"]}: open alternatives need category_label')
        for sid in members:
            s=skills.get(sid)
            if not s:report.add('UNKNOWN_GROUP_MEMBER',f'{g["id"]}: {sid} absent');continue
            if s['any_of_group']!=g['id']:report.add('GROUP_MISMATCH',f'{sid} any_of_group does not match {g["id"]}')
            for field_name in ('priority','importance'):
                if s[field_name]!=g[field_name]:report.add('GROUP_PRIORITY_MISMATCH',f'{sid}.{field_name} differs from {g["id"]}')
            if s['hard_gate_explicit']!=g['hard_gate_explicit']:
                report.add('GROUP_HARD_GATE_MISMATCH',f'{sid}: hard gate differs from {g["id"]}')

    critical=set(data['top_critical_skill_ids'])
    if len(critical)!=len(data['top_critical_skill_ids']):report.add('DUPLICATE_CRITICAL_ID','Repeated top critical skills')
    for sid in critical:
        if sid not in skills:report.add('UNKNOWN_CRITICAL',sid)
        elif skills[sid]['importance']!='critical':report.add('INVALID_CRITICAL',sid)
    rationale_ids=[r['id'] for r in data['critical_skill_rationale']]
    if set(rationale_ids)!=critical or len(rationale_ids)!=len(set(rationale_ids)):
        report.add('CRITICAL_RATIONALE_MISMATCH','critical_skill_rationale must cover top_critical_skill_ids exactly once')

    for label, priority in [('mustskills','mandatory'),('goodskills','preferred')]:
        expected=listed_skills(jd,label)
        for candidate in expected:
            exact=[s for s in data['skills'] if canonical_mention(s['name'],candidate) or canonical_mention(s['raw'],candidate)]
            if not exact:
                report.add('MISSING_LISTED_SKILL',f'{label}: missing {candidate!r}')
                report.coverage.append(f'MISSING {label}: {candidate}')
            elif not any(s['priority']==priority for s in exact):
                report.add('WRONG_LIST_PRIORITY',f'{label}: {candidate!r} should be {priority} unless individual wording overrides (review)')
        if expected:report.coverage.append(f'{label} entries checked: {len(expected)}')

    # High-precision detection of tools literally stated in an extracted responsibility or
    # the structured 'jobdescription' field. These are coverage signals, not proof of hard gates.
    duty_text=' '.join(d['source_quote'] for d in data['key_responsibilities'])
    structured=extract_structured_field(jd,'jobdescription') or ''
    for tech in ('ci/cd','rest apis','docker','kubernetes','terraform','aws','azure','gcp','python','java','javascript','jenkins','kafka','linux','react','node.js','pytorch','tensorflow'):
        if not (canonical_mention(duty_text,tech) or canonical_mention(structured,tech)):
            continue
        if not any(canonical_mention(s['name'],tech) or canonical_mention(s['raw'],tech) for s in data['skills']):
            report.add('MISSING_DUTY_SKILL',f'{tech} is explicitly named in JD duty but absent from skills')
            report.coverage.append(f'MISSING duty-linked tool: {tech}')

    if len(data['ambiguities'])>5:
        report.add('TOO_MANY_AMBIGUITIES','More than five ambiguity notes','warning')
    if len(data['warnings'])>8:
        report.add('TOO_MANY_WARNINGS','Long warnings list; evaluate usefulness','warning')
    return report


LOGISTICS_CATEGORIES=frozenset({'location','work_authorization','travel_shift'})

def partition_requirements_for_scoring(rubric:dict)->tuple[list[dict], list[dict]]:
    """Separate technical/role requirements from logistics without changing source data.

    IMPORTANT: downstream must use fit_requirements for job-fit scoring and
    route logistical_conditions through its own human-reviewed workflow.
    This helper does NOT determine legal eligibility or make hiring decisions.
    """
    rows=rubric.get('requirements',[])
    return ([r for r in rows if r.get('category') not in LOGISTICS_CATEGORIES],
            [r for r in rows if r.get('category') in LOGISTICS_CATEGORIES])

def versioned_cache_key(jd:str, model:str, extractor_version:str, schema_version:str='2.3')->str:
    """Store alongside raw source + model/settings hash when reproducibility matters."""
    value=json.dumps([jd,model,extractor_version,schema_version],ensure_ascii=False,separators=(',',':'))
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def audit_csv(path:Path):
    results=[]
    with path.open(newline='',encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            result={'jd_id':row.get('jd_id'),'model':row.get('model'),'finish_reason':row.get('finish_reason')}
            try:
                data=parse_raw(row.get('output') or '',allow_fences=True)
                report=validate_rubric(row.get('jd') or '',data)
                result.update(report.serialize())
                result['parsed_after_fence_normalization']=True
            except Exception as exc:
                result.update({'valid_for_cache':False,'parsed_after_fence_normalization':False,'errors':[f'PARSE: {type(exc).__name__}: {str(exc)[:140]}'],'warnings':[],'coverage':[]})
            if row.get('finish_reason')=='length':
                result['errors'].append('TRUNCATED: generation ended at token limit')
                result['valid_for_cache']=False
            results.append(result)
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--jd',type=Path)
    parser.add_argument('--rubric',type=Path)
    parser.add_argument('--eval-csv',type=Path)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    if args.eval_csv:
        results=audit_csv(args.eval_csv)
        payload={'runs':len(results),'cache_valid':sum(r['valid_for_cache'] for r in results),'results':results}
    else:
        if not (args.jd and args.rubric):parser.error('Supply --eval-csv OR both --jd and --rubric')
        jd=args.jd.read_text(encoding='utf-8')
        data=parse_raw(args.rubric.read_text(encoding='utf-8'),allow_fences=True)
        payload=validate_rubric(jd,data).serialize()
    if args.report:
        args.report.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(payload if not args.eval_csv else {k:v for k,v in payload.items() if k!='results'},indent=2,ensure_ascii=False))
    if args.eval_csv: return 0
    return 0 if payload['valid_for_cache'] else 1

if __name__=='__main__':raise SystemExit(main())
