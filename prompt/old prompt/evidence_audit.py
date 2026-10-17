"""Attach verified character spans to exact extractor quotes; do not trust LLM offsets.

Usage: verified, issues = annotate_quotes(extracted_resume, raw_resume, kind='resume')
Only raw original-document text should be supplied to this function.
"""
from copy import deepcopy
import re


def _span(text, quote):
    if not quote or not isinstance(quote,str):
        return {'status':'missing','start_char':None,'end_char':None,'occurrences':0}
    # Exact first; fallback only normalizes sequences of whitespace in PDF extraction.
    positions=[(m.start(),m.end()) for m in re.finditer(re.escape(quote),text)]
    if not positions:
        pattern=r'\s+'.join(re.escape(s) for s in quote.split())
        positions=[(m.start(),m.end()) for m in re.finditer(pattern,text)] if pattern else []
    if not positions:
        return {'status':'unverified','start_char':None,'end_char':None,'occurrences':0}
    start,end=positions[0]
    return {'status':'verified' if len(positions)==1 else 'ambiguous',
            'start_char':start,'end_char':end,'occurrences':len(positions)}


def annotate_quotes(extraction,raw_text,kind):
    """Return enriched copy, audit findings list. Never silently accept unverified quotes."""
    if kind not in ('resume','jd'):raise ValueError('kind must be resume or jd')
    out=deepcopy(extraction)
    containers=(('work_experiences','skills') if kind=='resume' else ('key_responsibilities','skills','requirements'))
    issues=[]
    for container in containers:
        for idx,item in enumerate(out.get(container,[]) or []):
            key='evidence_quote' if container=='skills' and kind=='resume' else 'source_quote'
            span=_span(raw_text,item.get(key))
            item['source_span']=span
            if span['status']!='verified':
                issues.append({'collection':container,'id':item.get('id',idx),'status':span['status']})
    return out,issues
