"""Strict-schema Gemini/Gemma v2.3 evaluation components.

This module does not alter JD semantics, silently repair model output, or
replace the project's existing semantic validator.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from jsonschema import validators


GEMMA_RULES = """
GEMMA V2.3 FIELD CONTRACT — follow the JSON schema above ALL examples:
- skills[].priority must use its schema enum (not 'responsibility' or 'inferred').
- skills[].importance: critical / important / supporting, where allowed by the schema.
- key_responsibilities[].importance: core / secondary ONLY; never 'important' or 'supporting'.
- experience_requirement and education_requirement: use jd_evidence and only the
  other fields actually permitted in those schema objects. Do NOT add source_quote.
- Supply ALL required keys, including jd_evidence where specified.
- Use ONLY explicitly supported hard gates; do not promote preferences into requirements.
- Copy each source_quote character for character from the supplied JD.
- Include explicitly named technologies in both requirements and responsibilities;
  do not invent a technology that is not in the JD.
- If mustskills and goodskills overlap, preserve explicitly mandatory minima and
  distinct preferred advanced proficiency; do not silently downgrade mandatory items.
- Before answering, check arrays, enums, required properties and ID consistency.
- Return exactly one complete JSON object, with no Markdown code fence.
""".strip()


def apply_gemma_rules(system_prompt: str, mode: str = 'unchanged') -> str:
    if mode == 'unchanged':
        return system_prompt
    if mode == 'targeted':
        return system_prompt.rstrip() + "\n\n" + GEMMA_RULES
    raise ValueError(f"unknown prompt mode: {mode}")


def parse_object(text: str) -> dict[str, Any]:
    """Strict JSON parsing; optionally remove a whole Markdown code fence.

    No character-level repair is performed, ever.
    """
    s = text.strip()
    m = re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```', s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        s = m.group(1)
    obj = json.loads(s)
    if not isinstance(obj, dict):
        raise ValueError('top-level JSON must be an object')
    return obj


def validate_schema(schema: dict[str, Any], obj: dict[str, Any]) -> list[str]:
    Validator = validators.validator_for(schema)
    Validator.check_schema(schema)
    validator = Validator(schema)
    errors = sorted(validator.iter_errors(obj), key=lambda e: (tuple(str(p) for p in e.path), e.message))
    return [f"{'.'.join(map(str, error.absolute_path)) or '$'}: {error.message}" for error in errors]


def quoted_evidence_issues(obj: dict[str, Any], jd: str) -> list[str]:
    """Conservative supplemental check only; not semantic completeness verification.

    Verifies nonempty source_quote fields are byte-exact substrings of JD (as Unicode
    Python strings). Does not substitute for the repository's v2.3 validator.
    """
    issues = []
    def visit(o: Any, path: str) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                where = (path + '.' + k).lstrip('.')
                if k == 'source_quote' and isinstance(v, str) and v and v not in jd:
                    issues.append(f'{where}: quote not found verbatim in JD')
                visit(v, where)
        elif isinstance(o, list):
            for i, v in enumerate(o):
                visit(v, f'{path}[{i}]')
    visit(obj, '')
    return issues


def content_fingerprint(jd: str, prompt: str, schema: dict[str, Any], model: str, extra: dict | None = None) -> str:
    """Content-addressed key: source mutations cannot be hidden behind stable JD ids."""
    data = json.dumps({'jd': jd, 'prompt': prompt, 'schema': schema, 'model': model,
                       'extra': extra or {}}, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def schema_format(schema: dict[str, Any], mode: str) -> dict[str, Any] | None:
    if mode == 'none':
        return None
    if mode == 'llamacpp':
        # llama.cpp server variant documented in its server README.
        return {'type': 'json_schema', 'schema': schema}
    if mode == 'openai':
        return {'type': 'json_schema', 'json_schema':
                {'name': 'jd_rubric_v23', 'strict': True, 'schema': schema}}
    raise ValueError(f'unknown schema mode: {mode}')


def unsupported_schema_keywords(schema: Any) -> list[str]:
    """Warnings about keywords that may not be enforced by llama.cpp grammar."""
    caution = {'uniqueItems', 'contains', 'minContains', 'maxContains', 'not',
               'if', 'then', 'else', 'dependentSchemas', 'patternProperties',
               '$dynamicRef', '$anchor'}
    issues = []
    def walk(v: Any, path: str = '$') -> None:
        if isinstance(v, dict):
            for k, x in v.items():
                if k in caution:
                    issues.append(f'{path}.{k}')
                walk(x, f'{path}.{k}')
        elif isinstance(v, list):
            for i, x in enumerate(v):
                walk(x, f'{path}[{i}]')
    walk(schema)
    return issues


def send_chat(*, api_base: str, api_key: str | None, model: str, messages: list[dict],
              response_format: dict | None, max_tokens: int, temperature: float, seed: int,
              top_p: float | None, timeout: float) -> dict:
    """Call an *already running* OpenAI-compatible service. Never load/reload models."""
    url = api_base.rstrip('/')
    if not url.endswith('/chat/completions'):
        url = url.rstrip('/') + '/chat/completions' if url.endswith('/v1') else url + '/v1/chat/completions'
    payload: dict[str, Any] = {
        'model': model, 'messages': messages, 'temperature': temperature,
        'max_tokens': max_tokens, 'seed': seed, 'stream': False
    }
    if top_p is not None:
        payload['top_p'] = top_p
    if response_format is not None:
        payload['response_format'] = response_format
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = 'Bearer ' + api_key
    req = urllib.request.Request(url, json.dumps(payload).encode('utf-8'), headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            out = json.load(res)
    except urllib.error.HTTPError as err:
        details = err.read(2500).decode('utf-8', errors='replace')
        raise RuntimeError(f'gateway HTTP {err.code}: {details}') from err
    if not isinstance(out, dict) or not out.get('choices'):
        raise RuntimeError(f'missing choices in gateway result: {str(out)[:200]}')
    return out


def response_text(response: dict) -> str:
    ch = response['choices'][0]
    msg = ch.get('message', {})
    content = msg.get('content')
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ''.join(x.get('text', '') for x in content if isinstance(x, dict))
    raise ValueError('No final content returned; inspect chat template/reasoning parser')


def load_validator_hook(file_path: str):
    """Opt-in integration hook: def validate(rubric: dict, jd: str) -> dict.

    Unlike schema-only checks this must invoke the project's real v2.3 semantic
    validator. No hook => no 'production accepted' label.
    """
    path = Path(file_path).resolve()
    spec = importlib.util.spec_from_file_location('gemma_user_validator', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import validator hook at {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, 'validate', None)
    if not callable(fn):
        raise RuntimeError('validator hook must define validate(rubric, jd)')
    return fn


def evaluate_response(output: str, jd: str, schema: dict[str, Any], hook=None) -> tuple[dict | None, dict]:
    result = {'parse_ok': False, 'schema_ok': False, 'exact_quotes_ok': False,
              'semantic_checked': hook is not None, 'accepted': False,
              'errors': [], 'warnings': []}
    try:
        rubric = parse_object(output)
    except (ValueError, json.JSONDecodeError) as err:
        result['errors'].append(f'JSON_PARSE: {err}')
        return None, result
    result['parse_ok'] = True
    result['errors'].extend(f'SCHEMA: {e}' for e in validate_schema(schema, rubric))
    result['schema_ok'] = not result['errors']
    quote_errors = quoted_evidence_issues(rubric, jd)
    result['errors'].extend('QUOTE: ' + e for e in quote_errors)
    result['exact_quotes_ok'] = not quote_errors
    if not result['errors'] and hook is not None:
        verdict = hook(rubric, jd)
        if not isinstance(verdict, dict) or type(verdict.get('accepted')) is not bool:
            raise TypeError('validator hook must return {"accepted": bool, "errors": [...], "warnings": [...]}')
        result['errors'].extend('SEMANTIC: ' + str(e) for e in verdict.get('errors', []))
        result['warnings'].extend(str(e) for e in verdict.get('warnings', []))
        if verdict['accepted'] and result['errors']:
            raise ValueError('semantic hook returned accepted=True and non-empty errors')
        result['accepted'] = verdict['accepted'] and not result['errors']
    return rubric, result

PROBE_SCHEMA = {
  'type': 'object',
  'properties': {'importance': {'type': 'string', 'enum': ['core', 'secondary']}},
  'required': ['importance'], 'additionalProperties': False
}
