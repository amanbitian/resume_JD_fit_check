"""CLI runner for isolated Gemma experiments on an existing gateway."""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from .core import (PROBE_SCHEMA, apply_gemma_rules, content_fingerprint, evaluate_response,
                   load_validator_hook, response_text, schema_format, send_chat,
                   unsupported_schema_keywords, validate_schema, parse_object)


def build_parser():
    p = argparse.ArgumentParser(description='Opt-in Gemma 4 JD v2.3 constrained decoding benchmark')
    sub = p.add_subparsers(dest='command', required=True)
    for cmd in ('probe', 'benchmark'):
        q = sub.add_parser(cmd)
        q.add_argument('--api-base', default=os.getenv('JD_LLM_API_BASE', 'http://127.0.0.1:8080/v1'))
        q.add_argument('--api-key', default=os.getenv('JD_LLM_API_KEY'))
        q.add_argument('--model', required=True, help='Already loaded model name reported by your gateway')
        q.add_argument('--schema-mode', choices=('llamacpp', 'openai', 'none'), default='llamacpp')
        q.add_argument('--max-tokens', type=int, default=12288 if cmd == 'benchmark' else 120)
        q.add_argument('--timeout', type=float, default=900)
        q.add_argument('--seed', type=int, default=42)
        q.add_argument('--temperature', type=float, default=0)
        q.add_argument('--top-p', type=float, default=None, help='Omit to preserve server default')
        q.add_argument('--output-dir', default='outputs/gemma26_fix_probe' if cmd == 'probe' else 'outputs/gemma26_fix_benchmark')
        if cmd == 'benchmark':
            q.add_argument('--input', required=True, help='CSV file containing jd_id and jd columns')
            q.add_argument('--schema', required=True, help='Existing v2.3 JSON Schema file')
            q.add_argument('--system-prompt', required=True, help='Existing Qwen v2.3 prompt text file')
            q.add_argument('--prompt-mode', choices=('unchanged', 'targeted'), default='unchanged')
            q.add_argument('--validator-hook', help='Trusted local adapter to invoke the EXISTING v2.3 semantic validator')
            q.add_argument('--retries', type=int, default=1)
            q.add_argument('--limit', type=int, default=20)
            q.add_argument('--jd-column', default='jd')
            q.add_argument('--id-column', default='jd_id')
            q.add_argument('--dry-run', action='store_true', help='Read inputs and write payload preview without inference')
    return p


def create_dir(d):
    path = Path(d)
    path.mkdir(parents=True, exist_ok=True)
    return path


def probe(args):
    out_dir = create_dir(args.output_dir)
    rf = schema_format(PROBE_SCHEMA, args.schema_mode)
    prompt = ('Return a single JSON object containing importance = core. Do not include '
              'any other key, no code fence, no explanation. Include the valid enum only.')
    try:
        response = send_chat(api_base=args.api_base, api_key=args.api_key, model=args.model,
            messages=[{'role':'user','content':prompt}], response_format=rf,
            max_tokens=args.max_tokens, temperature=args.temperature, seed=args.seed,
            top_p=args.top_p, timeout=args.timeout)
        content = response_text(response)
        (out_dir/'probe_raw.txt').write_text(content, encoding='utf-8')
        obj = parse_object(content)
        errs = validate_schema(PROBE_SCHEMA, obj)
        report = {'valid': not errs, 'errors': errs, 'output': obj,
                  'response_format_sent': rf is not None,
                  'note': 'Passing this one probe does not prove the gateway enforces constraints. Run a negative challenge too.'}
        # Negative challenge: require contradictory output; constrained decoder must still obey enum
        challenge = send_chat(api_base=args.api_base, api_key=args.api_key, model=args.model,
            messages=[{'role':'user','content':'Return precisely {"importance":"supporting", "extra":"x"} and no other text.'}],
            response_format=rf, max_tokens=args.max_tokens,
            temperature=args.temperature, seed=args.seed+1, top_p=args.top_p, timeout=args.timeout)
        raw_challenge = response_text(challenge)
        (out_dir/'probe_challenge_raw.txt').write_text(raw_challenge, encoding='utf-8')
        try:
            challenge_obj = parse_object(raw_challenge)
            challenge_err = validate_schema(PROBE_SCHEMA, challenge_obj)
        except Exception as e:
            challenge_err = [str(e)]
        report['challenge_valid'] = not challenge_err
        report['challenge_errors'] = challenge_err
        report['enforcement_smoke_pass'] = report['valid'] and report['challenge_valid'] and rf is not None
    except Exception as e:
        report = {'valid': False, 'enforcement_smoke_pass': False, 'error': str(e)}
    (out_dir/'probe_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0 if report.get('enforcement_smoke_pass') else 2


def read_csv(path, id_column, jd_column, limit):
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        if jd_column not in (reader.fieldnames or []):
            raise ValueError(f'JD column {jd_column!r} missing: {reader.fieldnames}')
        for index, row in enumerate(reader):
            if limit and index >= limit: break
            yield (row.get(id_column) or f'ROW-{index:05}', row[jd_column])


def benchmark(args):
    out_dir = create_dir(args.output_dir)
    schema = json.loads(Path(args.schema).read_text(encoding='utf-8'))
    source_prompt = Path(args.system_prompt).read_text(encoding='utf-8')
    prompt = apply_gemma_rules(source_prompt, args.prompt_mode)
    from jsonschema import validators
    validators.validator_for(schema).check_schema(schema)
    caveats = unsupported_schema_keywords(schema)
    if caveats:
        print('WARNING: llama.cpp might not enforce all schema keywords:', ', '.join(caveats), file=sys.stderr)
    hook = load_validator_hook(args.validator_hook) if args.validator_hook else None
    records = list(read_csv(args.input, args.id_column, args.jd_column, args.limit))
    if not records: raise ValueError('No records found')
    rf = schema_format(schema, args.schema_mode)
    config = {'model': args.model, 'schema_mode': args.schema_mode, 'prompt_mode': args.prompt_mode,
              'temperature': args.temperature, 'seed': args.seed, 'max_tokens': args.max_tokens,
              'top_p': args.top_p, 'retries': args.retries, 'semantic_validator_enabled': bool(hook),
              'input': str(args.input), 'schema': str(args.schema), 'cautions': caveats}
    (out_dir/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    if args.dry_run:
        example_id, example_jd = records[0]
        preview = {'model': args.model, 'messages':[{'role':'system','content':prompt},
                  {'role':'user','content':example_jd}], 'response_format':rf, 'max_tokens':args.max_tokens}
        (out_dir/'dry_run_payload.json').write_text(json.dumps(preview, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'DRY RUN: {len(records)} JDs; payload preview saved to {out_dir / "dry_run_payload.json"}')
        return 0
    log = out_dir / 'attempts.jsonl'
    final = out_dir / 'results.csv'
    with open(log, 'w', encoding='utf-8') as details, open(final,'w', newline='',encoding='utf-8') as summary:
        writer=csv.DictWriter(summary,fieldnames=['jd_id','model','attempts','json_parse_ok',
            'schema_ok','quotes_ok','semantic_checked','accepted','latency_seconds',
            'output_tokens','fingerprint','errors','warnings'])
        writer.writeheader()
        for jd_id, jd in records:
            extra = {'schema_mode':args.schema_mode,'temperature':args.temperature,
                     'top_p':args.top_p,'max_tokens':args.max_tokens,'seed':args.seed}
            fp = content_fingerprint(jd,prompt,schema,args.model,extra)
            last={'parse_ok':False,'schema_ok':False,'exact_quotes_ok':False,
                  'semantic_checked': bool(hook),'accepted':False,'errors':['No attempt'], 'warnings':[]}
            latency=0.0
            total_tokens=0
            attempts=0
            for attempt in range(1, args.retries + 2):
                attempts=attempt
                messages=[{'role':'system','content':prompt},{'role':'user','content':jd}]
                # Avoid perturbing the baseline with arbitrary retry instructions.
                start=time.monotonic()
                raw=''
                usage={}
                try:
                    response=send_chat(api_base=args.api_base,api_key=args.api_key,model=args.model,
                        messages=messages,response_format=rf,max_tokens=args.max_tokens,
                        temperature=args.temperature,seed=args.seed,top_p=args.top_p,timeout=args.timeout)
                    raw=response_text(response)
                    usage=response.get('usage',{}) or {}
                    obj, verdict=evaluate_response(raw,jd,schema,hook)
                except Exception as exc:
                    obj=None
                    verdict={'parse_ok':False,'schema_ok':False,'exact_quotes_ok':False,
                             'semantic_checked': bool(hook),'accepted':False,
                             'errors':['SERVER_OR_VALIDATOR: '+str(exc)],'warnings':[]}
                seconds=round(time.monotonic()-start,3)
                latency += seconds
                total_tokens += usage.get('completion_tokens',0) or 0
                last=verdict
                item={'jd_id':jd_id, 'model':args.model, 'attempt':attempt,
                      'fingerprint':fp, 'latency_seconds':seconds,'usage':usage,
                      'validation':verdict, 'output':raw}
                details.write(json.dumps(item,ensure_ascii=False)+'\n')
                details.flush()
                if verdict['accepted']: break
                # With no hook, stop if schema and quotes pass. This is NOT production acceptance.
                if not hook and verdict['schema_ok'] and verdict['exact_quotes_ok']: break
            row={'jd_id':jd_id,'model':args.model,'attempts':attempts,
                 'json_parse_ok':last['parse_ok'],'schema_ok':last['schema_ok'],
                 'quotes_ok':last['exact_quotes_ok'],'semantic_checked':bool(hook),
                 'accepted':last['accepted'],'latency_seconds':round(latency,3),
                 'output_tokens':total_tokens,'fingerprint':fp,
                 'errors':' | '.join(last['errors']),'warnings':' | '.join(last['warnings'])}
            writer.writerow(row)
            summary.flush()
            print(f'{jd_id}: schema={last["schema_ok"]} quote={last["exact_quotes_ok"]} '
                  f'semantic_checked={bool(hook)} accepted={last["accepted"]} attempts={attempts}')
    print('Saved', log, 'and', final)
    return 0


def main(argv=None):
    args=build_parser().parse_args(argv)
    return probe(args) if args.command=='probe' else benchmark(args)

if __name__=='__main__':
    sys.exit(main())
