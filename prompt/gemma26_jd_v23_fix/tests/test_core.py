import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gemma_jd_fix.core import (
    GEMMA_RULES, PROBE_SCHEMA, apply_gemma_rules, content_fingerprint,
    evaluate_response, parse_object, quoted_evidence_issues, schema_format,
    unsupported_schema_keywords, validate_schema, load_validator_hook,
)


class TestCore(unittest.TestCase):
    def test_parse_plain_json(self):
        self.assertEqual(parse_object('{"importance":"core"}'), {'importance': 'core'})

    def test_parse_only_full_fence(self):
        self.assertEqual(parse_object('```json\n{"importance":"core"}\n```'), {'importance':'core'})
        with self.assertRaises(ValueError):
            parse_object('extra commentary\n{"importance":"core"}')

    def test_never_repair_null_extra_quote(self):
        with self.assertRaises(ValueError):
            parse_object('{"min_years":null"}')

    def test_reject_enum_and_extra_key(self):
        self.assertTrue(validate_schema(PROBE_SCHEMA, {'importance':'supporting'}))
        self.assertTrue(validate_schema(PROBE_SCHEMA, {'importance':'core','source_quote':'x'}))
        self.assertEqual(validate_schema(PROBE_SCHEMA, {'importance':'core'}), [])

    def test_reject_missing_required(self):
        self.assertTrue(validate_schema(PROBE_SCHEMA, {}))

    def test_quote_is_verbatim(self):
        jd='Must know test-driven development and Python.'
        obj={'skills':[{'source_quote':'test‑driven development'}]}
        self.assertEqual(len(quoted_evidence_issues(obj,jd)),1)
        self.assertEqual(quoted_evidence_issues({'skills':[{'source_quote':'test-driven development'}]},jd),[])

    def test_evaluate_does_not_claim_accepted_without_hook(self):
        obj, v = evaluate_response('{"importance":"core"}', 'job', PROBE_SCHEMA)
        self.assertTrue(v['schema_ok'])
        self.assertFalse(v['semantic_checked'])
        self.assertFalse(v['accepted'])

    def test_evaluate_accepted_with_hook(self):
        hook=lambda rubric,jd: {'accepted':True,'errors':[],'warnings':[]}
        _,v=evaluate_response('{"importance":"core"}', 'job', PROBE_SCHEMA, hook=hook)
        self.assertTrue(v['accepted'])

    def test_evaluate_rejects_semantic_errors(self):
        hook=lambda rubric,jd: {'accepted':False,'errors':['missing CI/CD'],'warnings':[]}
        _,v=evaluate_response('{"importance":"core"}', 'job', PROBE_SCHEMA, hook=hook)
        self.assertFalse(v['accepted'])
        self.assertTrue(any('missing CI/CD' in e for e in v['errors']))

    def test_false_semantic_hook_cannot_overrule_schema(self):
        hook=lambda rubric,jd: {'accepted':True,'errors':[],'warnings':[]}
        _,v=evaluate_response('{"importance":"supporting"}', 'job', PROBE_SCHEMA, hook=hook)
        self.assertFalse(v['accepted'])
        self.assertFalse(v['schema_ok'])

    def test_cache_key_tracks_jd_change(self):
        a=content_fingerprint('Python','system',PROBE_SCHEMA,'gemma')
        b=content_fingerprint('Python and AWS','system',PROBE_SCHEMA,'gemma')
        c=content_fingerprint('Python','system',PROBE_SCHEMA,'gemma', {'max_tokens':8000})
        self.assertNotEqual(a,b)
        self.assertNotEqual(a,c)

    def test_prompt_baseline_is_byte_identical(self):
        prompt='CURRENT PRODUCTION QWEN PROMPT\n'
        self.assertEqual(apply_gemma_rules(prompt, 'unchanged'),prompt)
        self.assertIn('key_responsibilities', apply_gemma_rules(prompt, 'targeted'))

    def test_response_modes(self):
        self.assertEqual(schema_format(PROBE_SCHEMA,'llamacpp')['schema'],PROBE_SCHEMA)
        self.assertEqual(schema_format(PROBE_SCHEMA,'openai')['json_schema']['schema'], PROBE_SCHEMA)
        self.assertIsNone(schema_format(PROBE_SCHEMA,'none'))

    def test_keywords_warning(self):
        self.assertIn('$.uniqueItems', unsupported_schema_keywords({'uniqueItems': True}))

    def test_unwired_hook_fails_closed(self):
        from pathlib import Path
        template=Path(__file__).resolve().parents[1]/'examples'/'validator_hook_TEMPLATE.py'
        hook=load_validator_hook(str(template))
        with self.assertRaises(NotImplementedError):
            hook({},'jd')

if __name__ == '__main__':
    unittest.main()
