"""Single-request local LM Studio / OpenAI-compatible runner.

Examples:
 python run_local_once.py jd --jd jd.txt --model YOUR_LOADED_MODEL_ID --output rubric.json
 python run_local_once.py candidate --jd jd.txt --rubric rubric.json --resume resume.txt --model YOUR_LOADED_MODEL_ID --output assessment.json

Calls the model only when YOU run this script; nothing is sent from this package automatically.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError,URLError
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser()
    p.add_argument('task',choices=['jd','candidate'])
    p.add_argument('--jd',required=True)
    p.add_argument('--rubric')
    p.add_argument('--resume')
    p.add_argument('--model',required=True,help='Exact loaded model ID in your runtime')
    p.add_argument('--endpoint',default='http://127.0.0.1:1234/v1/chat/completions')
    p.add_argument('--output',required=True)
    p.add_argument('--reasoning-effort',choices=['low','medium','high'],default='low')
    p.add_argument('--omit-reasoning-effort',action='store_true',help='Use when runtime ignores or rejects reasoning_effort')
    p.add_argument('--max-tokens',type=int,default=8192)
    p.add_argument('--temperature',type=float,default=0)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--mode',choices=['schema','standalone'],default='schema')
    args=p.parse_args()
    if args.task=='candidate' and not (args.rubric and args.resume):p.error('candidate requires --rubric and --resume')
    jd=Path(args.jd).read_text(encoding='utf-8')
    if args.task=='jd':
        name='GPTOSS20B_JD_EXTRACTOR_SCHEMA_MODE.txt' if args.mode=='schema' else 'GPTOSS20B_JD_EXTRACTOR_STANDALONE.txt'
        schema='jd_rubric.schema.json'
        user='REFERENCE_DATE: 2026-10-17\n<JOB_DESCRIPTION>\n'+jd+'\n</JOB_DESCRIPTION>'
    else:
        name='GPTOSS20B_CANDIDATE_MATCHER_EXPERIMENTAL.txt'
        schema='candidate_assessment.schema.json'
        rub=Path(args.rubric).read_text(encoding='utf-8')
        resume=Path(args.resume).read_text(encoding='utf-8')
        user='REFERENCE_DATE: 2026-10-17\n<ORIGINAL_JD>\n'+jd+'\n</ORIGINAL_JD>\n<VALIDATED_JD_RUBRIC>\n'+rub+'\n</VALIDATED_JD_RUBRIC>\n<RESUME>\n'+resume+'\n</RESUME>'
    sys=(ROOT/name).read_text(encoding='utf-8')
    payload={'model':args.model,'messages':[{'role':'system','content':sys},{'role':'user','content':user}],
             'temperature':args.temperature,'seed':args.seed,'max_tokens':args.max_tokens,'stream':False}
    if not args.omit_reasoning_effort:payload['reasoning_effort']=args.reasoning_effort
    if args.mode=='schema':
        s=json.loads((ROOT/schema).read_text(encoding='utf-8'))
        # Some backends accept subsets of JSON Schema, not necessarily every Draft2020 keyword.
        payload['response_format']={'type':'json_schema','json_schema':{'name':('jd_rubric' if args.task=='jd' else 'candidate_assessment'),'strict':True,'schema':s}}
    req=Request(args.endpoint,json.dumps(payload).encode('utf-8'),headers={'Content-Type':'application/json'},method='POST')
    started=time.perf_counter()
    try:
        with urlopen(req,timeout=900) as r:res=json.load(r)
    except HTTPError as e:
        raise SystemExit(f'HTTP {e.code}: {e.read().decode("utf-8",errors="replace")}\nCheck backend schema and reasoning-effort support.')
    except URLError as e:raise SystemExit(f'Could not contact local inference server: {e}')
    elapsed=time.perf_counter()-started
    choice=res['choices'][0]
    content=choice['message'].get('content','')
    if not isinstance(content,str):raise SystemExit('Output message.content is not a string; inspect runtime adapter.')
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(content,encoding='utf-8')
    metrics={'task':args.task,'model':args.model,'mode':args.mode,'reasoning_effort_requested':None if args.omit_reasoning_effort else args.reasoning_effort,
             'finish_reason':choice.get('finish_reason'),'usage':res.get('usage',{}),'elapsed_seconds':round(elapsed,3)}
    Path(str(out)+'.metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(out),'metrics':metrics},indent=2))
    print('Run verify_gptoss.py to verify source quote fidelity and downstream schema before caching.')
if __name__=='__main__':main()
