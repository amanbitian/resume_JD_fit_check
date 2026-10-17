"""v3 deterministic JD/resume score calculator.

Use: result = recompute(jd_json, resume_json, model_assessment, reference_date="2026-10-17")
     final_assessment = merge_assessment(model_assessment, result)

Validation intentionally fails closed. Never rank a candidate when required matches
or a required reference date is missing. Output None + unscorable, not 1/10.
This is a reference implementation; calibrate thresholds and do legal/product review
before production use. Python 3.10+, no external dependencies.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from math import ceil
import re

DIM_WEIGHTS = dict(skills=.35, work_experience=.25, role_alignment=.15,
                   domain_alignment=.10, years=.10, education_certifications=.05)
IMPORTANCE = {'critical':3, 'important':2, 'supporting':1}
PRIORITY = {'mandatory':1.0, 'preferred':.5}
ROLE = {'exact':10, 'strong_adjacent':8, 'transferable':5, 'misaligned':1, 'unclear':5}
SENIOR = {'match':0, 'under_1':-2, 'under_2plus':-4, 'over_1':0, 'over_2plus':0, 'unclear':0}
FUNCTION = {'exact':10, 'adjacent':7, 'transferable':4, 'misaligned':0}
INDUSTRY = {'exact':10, 'adjacent':7, 'different':4}
VALUE = {'met':10, 'demonstrated':10, 'partial':5, 'not_evidenced':0, 'contradicted':0}
EXPERIENCE_CATS = {'domain_experience','role_experience','management_experience'}
LOGISTICS = {'language','location','work_authorization','travel_shift'}
INVALID_RESUME = {'not_a_resume','empty_or_unreadable','multiple_candidates'}
INVALID_JD = {'not_a_jd','empty_or_unreadable','multiple_roles'}
MAX_SKILL_SCORE = {'met':(8,10),'partial':(3,6),'not_evidenced':(0,1)}


def _round1(num):
    return float(Decimal(str(num)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))


def _wavg(pairs):
    good = [(w,v) for w,v in pairs if w > 0]
    return _round1(sum(w*v for w,v in good)/sum(w for w,_ in good)) if good else None


def _round_half_down(number):
    return ceil(number - .5 - 1e-10)


def _band(score):
    if score is None: return 'unscorable'
    if score <= 2: return 'no_fit'
    if score <= 4: return 'weak_fit'
    if score <= 6: return 'potential_fit'
    if score <= 8: return 'good_fit'
    return 'strong_fit'


def _index(items, id_field, label):
    out = {}
    for item in items:
        key = item[id_field]
        if key in out: raise ValueError(f'Duplicate {label} identifier {key}')
        out[key] = item
    return out


def _month(value, as_end, reference_date):
    """Return (YYYY*12+MM, high_quality). Year-only dates are UNKNOWN for gates."""
    if value == 'present':
        if not reference_date: return None, False
        value = reference_date[:7]
    if isinstance(value,str) and re.fullmatch(r'\d{4}-\d{2}',value):
        year,month = map(int,value.split('-'))
        if 1<=month<=12 and 1950<=year<=2100:
            return year*12+month, True
    return None, False


def _compute_years(resume, a, jd, reference_date):
    """Return verified experience years only if ALL contributing timelines are known.

    For entry JDs eligible internship months are relevant, NOT professional months.
    Per-month weighted maximum avoids double-counting overlapping employment.
    """
    exp = resume.get('work_experiences') or []
    ref_month = None
    if reference_date:
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', reference_date):
            raise ValueError('reference_date must be YYYY-MM-DD')
        ref_month, _ = _month(reference_date[:7], True, reference_date)
    relevance = _index(a.get('experience_relevance') or [], 'resume_experience_id','role relevance')
    all_refs = {e['id'] for e in exp}
    if set(relevance) != all_refs:
        raise ValueError(f'Missing/extra experience relevance ids: {sorted(all_refs ^ set(relevance))}')
    prof = set()
    weights = defaultdict(float)
    uncertain = []
    allows_intern = jd.get('seniority') in ('intern','entry')
    for e in exp:
        typ = e.get('employment_type')
        start,valid_start = _month(e.get('start'),False,reference_date)
        end,valid_end = _month(e.get('end'),True,reference_date)
        if not(valid_start and valid_end) or start>end:
            uncertain.append(e['id'])
            continue
        if ref_month is not None and start>ref_month:
            uncertain.append(e['id']+':future_start')
            continue
        if ref_month is not None and end>ref_month:
            # Do not award future experience. Future-dated work may be a plan,
            # not a current role; suppress hard year gates and request review.
            end=ref_month
            uncertain.append(e['id']+':future_end')
        rank = relevance[e['id']]['relevance']
        if rank not in ('high','medium','low','none'):
            raise ValueError(f'Bad relevance for {e["id"]}')
        weight = {'high':1.0,'medium':.5,'low':0.0,'none':0.0}[rank]
        is_intern = typ in ('internship','apprenticeship')
        for m in range(start,end+1):
            if not is_intern:
                prof.add(m)
                weights[m] = max(weights[m],weight)
            elif allows_intern:
                weights[m] = max(weights[m],weight * .5)
    # Unknown full-time equivalents, estimated role spans and missing dates mean
    # we only know lower bounds. Do not use them as hard negatives.
    quality = (resume.get('experience_summary') or {}).get('date_quality')
    if quality != 'complete': uncertain.append('date_quality_not_complete')
    if any(e.get('employment_type') in ('part_time','unclear') for e in exp):
        uncertain.append('workload_not_verified')
    required = (jd.get('experience_requirement') or {}).get('min_years')
    basis = (jd.get('experience_requirement') or {}).get('basis')
    total = len(prof)/12
    relevant = sum(weights.values())/12
    used = total if basis=='total' else relevant
    ratio_raw = used/float(required) if required and float(required)>0 else None
    certain = not uncertain
    return {'total_years':_round1(total), 'relevant_years':_round1(relevant),
            'years_used_for_requirement':_round1(used), 'required_min_years':required,
            'ratio':round(ratio_raw,2) if ratio_raw is not None and certain else None,
            'ratio_unrounded':ratio_raw if certain else None,
            'reliable':certain,'uncertainties':sorted(set(uncertain))}


def _score_skills(jd,a,resume):
    jd_skills = _index(jd.get('skills') or [],'id','JD skill')
    sm = _index(a.get('skill_matches') or [],'jd_skill_id','skill match')
    if set(sm)!=set(jd_skills):
        raise ValueError(f'Skill matches missing/extra IDs {sorted(set(sm)^set(jd_skills))}')
    rm_skills = {s.get('name') for s in resume.get('skills') or []}
    ref_ids = {e['id'] for key in ('work_experiences','projects','education','certifications')
               for e in (resume.get(key) or [])}
    ref_ids |= {'skills_section','summary'}
    for sid,m in sm.items():
        st = m['status']; val = m['match_score']
        if st not in MAX_SKILL_SCORE or not isinstance(val,(int,float)):
            raise ValueError(f'Invalid status/score {sid}')
        lo,hi = MAX_SKILL_SCORE[st]
        if not lo<=val<=hi: raise ValueError(f'Status/score mismatch {sid}: {st}/{val}')
        if set(m.get('resume_refs') or []) - ref_ids:
            raise ValueError(f'Unknown resume reference in skill {sid}')
        if st=='met' and (not m.get('resume_refs') or m.get('resume_skill') not in rm_skills):
            raise ValueError(f'Unsupported met skill {sid}')
        if st=='met' and not any(ref.startswith(('E','P')) and ref[1:].isdigit() for ref in m.get('resume_refs') or []):
            raise ValueError(f'Met skill {sid} has no work/project evidence; lists/courses alone cannot prove use')
        if st=='not_evidenced' and m.get('match_type') not in ('adjacent','none'):
            raise ValueError(f'Inconsistent not_evidenced skill {sid}')
    groups = _index(jd.get('any_of_groups') or [],'id','JD group')
    gm = _index(a.get('group_matches') or [],'group_id','group match')
    if set(gm)!=set(groups): raise ValueError('Missing or extra group assessments')
    items = []
    for s in jd_skills.values():
        if s.get('any_of_group'): continue
        m = sm[s['id']]
        status,score = m['status'],m['match_score']
        refs=m.get('resume_refs') or []
        project_only=('project_only' in m.get('modifiers',[]) or
                      (bool(refs) and all(ref.startswith('P') and ref[1:].isdigit() for ref in refs)))
        if (project_only and jd.get('seniority') not in ('intern','entry')
            and s.get('hard_gate_explicit') and s.get('priority')=='mandatory'
            and s.get('importance')=='critical'):
            status,score = 'partial',min(score,6)
        items.append(dict(id=s['id'],priority=s['priority'],importance=s['importance'],
                          hard_gate_explicit=s.get('hard_gate_explicit') is True,
                          status=status,score=score))
    for g in groups.values():
        members = [sm[i] for i in g['member_skill_ids']]
        need = g['min_required']
        if not 1<=need<=len(members): raise ValueError(f'Invalid cardinality group {g["id"]}')
        status='met' if sum(m['status']=='met' for m in members)>=need else ('partial' if any(m['status'] in ('met','partial') for m in members) else 'not_evidenced')
        best=sorted((m['match_score'] for m in members),reverse=True)[:need]
        score=_round1(sum(best)/need)
        if g.get('open_category') and gm[g['id']].get('open_category_resume_skill'):
            alt=gm[g['id']]['open_category_resume_skill']
            refs=gm[g['id']].get('open_category_resume_refs') or []
            if alt not in rm_skills or not refs or set(refs)-ref_ids:
                raise ValueError(f'Open category claim unsupported in {g["id"]}')
            if need!=1: raise ValueError('Open-category n>1 needs a verified multi-skill matcher')
            # Model must provide checked semantic judgement; flag review elsewhere.
            if gm[g['id']]['status']=='met': status,score='met',min(10,max(8,gm[g['id']]['match_score']))
        items.append(dict(id=g['id'],priority=g['priority'],importance=g['importance'],
                          hard_gate_explicit=g.get('hard_gate_explicit') is True,
                          status=status,score=score))
    return items


def recompute(jd:dict,resume:dict,a:dict,reference_date:str|None=None)->dict:
    """Deterministic scoring. Raises ValueError on malformed complete assessments."""
    if resume.get('input_status') in INVALID_RESUME or jd.get('input_status') in INVALID_JD or a.get('input_check',{}).get('status') != 'ok':
        return {'assessment_status':'unscorable','dimension_scores':None,'weighted_score':None,
                'base_score':None,'caps':[],'final_score':None,'fit_band':'unscorable',
                'years':None,'needs_human_review':True,'review_reasons':['Input invalid, mixed, or unscorable']}
    if resume.get('input_status') not in ('ok','partial') or jd.get('input_status') not in ('ok','partial'):
        raise ValueError('Unknown document input status')
    items = _score_skills(jd,a,resume)
    reqs = _index(jd.get('requirements') or [],'id','JD requirement')
    rm = _index(a.get('requirement_matches') or [],'jd_requirement_id','requirement match')
    dm = _index(a.get('responsibility_matches') or [],'jd_responsibility_id','responsibility match')
    expected_d={d['id'] for d in jd.get('key_responsibilities') or []}
    if set(rm)!=set(reqs) or set(dm)!=expected_d:
        raise ValueError('Missing/extra requirement or responsibility match')
    for m in rm.values():
        if m['status'] not in ('met','partial','not_evidenced','contradicted'):raise ValueError('Invalid requirement match')
    for m in dm.values():
        if m['status'] not in ('demonstrated','partial','not_evidenced'):raise ValueError('Invalid responsibility match')
    al=a['alignment']; edu_status=(a.get('education_match') or {}).get('status','not_applicable')
    if edu_status not in ('met','partial','not_evidenced','contradicted','not_applicable'): raise ValueError('Bad education status')
    years=_compute_years(resume,a,jd,reference_date)
    er=jd.get('experience_requirement') or {}
    ed=jd.get('education_requirement') or {}
    if (er.get('min_years') or 0)<0: raise ValueError('Invalid min years')
    if ed.get('min_level') and edu_status=='not_applicable': raise ValueError('Education match missing')

    dims={}
    dims['skills']=_wavg([(IMPORTANCE[i['importance']]*PRIORITY[i['priority']],i['score']) for i in items])
    work=[(2 if d['importance']=='core' else 1,VALUE[dm[d['id']]['status']]) for d in jd.get('key_responsibilities') or []]
    work += [(2 if r['priority']=='mandatory' else 1,VALUE[rm[r['id']]['status']]) for r in reqs.values() if r['category'] in EXPERIENCE_CATS]
    dims['work_experience']=_wavg(work)
    dims['role_alignment']=max(0,min(10,ROLE[al['role']]+SENIOR[al['seniority_fit']]))
    f=FUNCTION[al['functional']]
    dims['domain_alignment']=float(f) if al['industry']=='not_applicable' else _round1(.75*f+.25*INDUSTRY[al['industry']])
    ratio=years['ratio_unrounded']
    dims['years']=None if ratio is None else (10 if ratio>=1 else 7 if ratio>=.8 else 5 if ratio>=.6 else 3 if ratio>=.4 else 1 if ratio>0 else 0)
    edu=[]
    if ed.get('min_level'):
        edu.append((PRIORITY.get(ed.get('priority'),1.0),VALUE[edu_status]))
    edu += [(PRIORITY[r['priority']],VALUE[rm[r['id']]['status']]) for r in reqs.values() if r['category']=='certification']
    dims['education_certifications']=_wavg(edu)
    live={k:v for k,v in dims.items() if v is not None}
    weighted=_round1(sum(DIM_WEIGHTS[k]*v for k,v in live.items()) / sum(DIM_WEIGHTS[k] for k in live))
    base=max(1,min(10,_round_half_down(weighted)))

    # Hard score caps trigger ONLY for explicitly non-negotiable items.
    caps=[]
    def cap(name,val): caps.append((name,val))
    if al['role']=='misaligned' and al['functional']=='misaligned':cap('CAP_ROLE_MISMATCH',3)
    mand=[i for i in items if i['priority']=='mandatory' and i['hard_gate_explicit']]
    crit=[i for i in mand if i['importance']=='critical']
    noncrit=[i for i in mand if i['importance']!='critical']
    n=sum(i['status']=='not_evidenced' for i in crit)
    if n:cap('CAP_CRITICAL_MISSING',5 if n==1 else 3)
    n=sum(i['status']=='partial' for i in crit)
    if n:cap('CAP_CRITICAL_PARTIAL',6 if n<=2 else 5)
    n=sum(i['status']=='not_evidenced' for i in noncrit)
    if n:cap('CAP_MANDATORY_MISSING',6 if n<=2 else 5)
    if sum(i['status']=='partial' for i in noncrit)>=3:cap('CAP_MANDATORY_PARTIAL',7)
    if er.get('hard_gate_explicit') is True and er.get('priority')=='mandatory' and (er.get('min_years') or 0)>0 and ratio is not None:
        if ratio<.5:cap('CAP_YEARS',4)
        elif ratio<.8:cap('CAP_YEARS',6)
    ed_flags=[]
    if ed.get('min_level') and ed.get('priority')=='mandatory' and ed.get('hard_gate_explicit') is True:ed_flags.append(edu_status)
    ed_flags += [rm[r['id']]['status'] for r in reqs.values() if r['category']=='certification' and r['priority']=='mandatory' and r.get('hard_gate_explicit') is True]
    if 'contradicted' in ed_flags:cap('CAP_EDUCATION',5)
    elif any(s in ('not_evidenced','partial') for s in ed_flags):cap('CAP_EDUCATION',6)
    for r in reqs.values():
        if r['priority']!='mandatory' or r.get('hard_gate_explicit') is not True:continue
        status=rm[r['id']]['status'];crit_r=r['importance']=='critical'
        if r['category'] in EXPERIENCE_CATS:
            if status in ('not_evidenced','contradicted'):cap('CAP_EXPERIENCE_REQ',5 if crit_r else 6)
            elif status=='partial':cap('CAP_EXPERIENCE_REQ',6 if crit_r else 7)
        if r['category'] in LOGISTICS and status=='contradicted':cap('CAP_LOGISTICS',3)
    all_mand=all(i['status']=='met' for i in mand) and all(rm[r['id']]['status']=='met' for r in reqs.values() if r['priority']=='mandatory' and r.get('hard_gate_explicit') is True)
    if ed.get('min_level') and ed.get('priority')=='mandatory' and ed.get('hard_gate_explicit') is True:all_mand &= edu_status=='met'
    pref=[i for i in items if i['priority']=='preferred']
    pref_total=sum(IMPORTANCE[i['importance']] for i in pref)
    pref_met=sum(IMPORTANCE[i['importance']] for i in pref if i['status']=='met')
    if not (all_mand and (pref_total==0 or pref_met/pref_total>=.5)):cap('CAP_TOP_BAND',8)
    final=min([base]+[v for _,v in caps]); reasons=[]
    if years['uncertainties'] and (er.get('min_years') or 0)>0: reasons.append('Unverified experience dates or full-time-equivalent hours')
    if resume.get('input_status')=='partial' or jd.get('input_status')=='partial':reasons.append('Partial input document')
    if resume.get('suspicious_instructions_detected') or jd.get('suspicious_instructions_detected'): reasons.append('Possible prompt injection')
    if any(i['priority']=='mandatory' and not i['hard_gate_explicit'] and i['status']!='met' for i in items):reasons.append('Unmet inferred/section-based requirement: validate JD priority')
    if any(r['priority']=='mandatory' and not r.get('hard_gate_explicit') and rm[r['id']]['status']!='met' for r in reqs.values()):reasons.append('Non-skill requirement requires review')
    if any(g.get('open_category') and (a.get('group_matches') or []) for g in jd.get('any_of_groups') or []):reasons.append('Open alternative category requires semantic validation')
    if any(r['category'] in LOGISTICS and r['priority']=='mandatory' and rm[r['id']]['status']!='met' for r in reqs.values()):reasons.append('Mandatory logistics requires recruiter confirmation')
    if a.get('needs_human_review'):reasons.append('Model requested review')
    if a.get('confidence') is not None and a.get('confidence')<.6:reasons.append('Low evidence confidence')
    if final in (6,7) and any(i['status']=='partial' and i['importance']=='critical' for i in items):reasons.append('Borderline score and partial critical skill')
    return {'assessment_status':'needs_review' if reasons else 'scored','dimension_scores':dims,
            'weighted_score':weighted,'base_score':base,'caps':caps,'final_score':final,
            'fit_band':_band(final),'years':years,'needs_human_review':bool(reasons),
            'review_reasons':sorted(set(reasons))}


def merge_assessment(model,calc):
    """Replace all model-calculated fields; preserve evidence only."""
    output=dict(model)
    for k in ('dimension_scores','weighted_score','base_score','final_score','fit_band'):
        output[k]=calc[k]
    output['caps_applied']=[{'cap':name,'value':v,'reason':'Deterministic scoring policy'} for name,v in calc['caps']]
    if calc.get('years') is not None:
        output['years']={k:v for k,v in calc['years'].items() if k in ('total_years','relevant_years','years_used_for_requirement','required_min_years','ratio')}
    output['assessment_status']=calc['assessment_status']
    output['needs_human_review']=calc['needs_human_review']
    output['review_reasons']=calc['review_reasons']
    return output


def compare(model,calc):
    """Report numerical drift, not ground-truth accuracy."""
    diff={}
    for k in ('weighted_score','base_score','final_score','fit_band'):
        if model.get(k)!=calc.get(k):diff[k]={'model':model.get(k),'calculated':calc.get(k)}
    for k,v in (calc.get('dimension_scores') or {}).items():
        guess=(model.get('dimension_scores') or {}).get(k)
        if guess is None and v is None:continue
        if guess is None or v is None or abs(guess-v)>.15:
            diff['dimension.'+k]={'model':guess,'calculated':v}
    return diff
