import copy
import json
import unittest
from pathlib import Path
from jsonschema import Draft202012Validator
from verify_gptoss import strict_json, DuplicateKeyError, validate_candidate
ROOT=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
    def setUp(self):
        self.jd_schema=json.loads((ROOT/'jd_rubric.schema.json').read_text())
        self.candidate_schema=json.loads((ROOT/'candidate_assessment.schema.json').read_text())
        self.jd_shape=json.loads((ROOT/'valid_json_shape_example.json').read_text())
        self.assessment=json.loads((ROOT/'candidate_scored_shape_example.json').read_text())
    def test_jd_shape(self): Draft202012Validator(self.jd_schema).validate(self.jd_shape)
    def test_candidate_shape(self): Draft202012Validator(self.candidate_schema).validate(self.assessment)
    def test_stray_quote_rejected(self):
        with self.assertRaises(json.JSONDecodeError):strict_json('{"min_years":null"}')
    def test_duplicate_key_rejected(self):
        with self.assertRaises(DuplicateKeyError):strict_json('{"min_years":null,"min_years":0}')
    def test_candidate_valid(self):
        self.assertEqual([],validate_candidate('Python','My Python experience',self.assessment))
    def test_quote_character_change_rejected(self):
        a=copy.deepcopy(self.assessment)
        a['checks'][0]['jd']='test‑driven'
        self.assertTrue(any('JD_QUOTE' in e for e in validate_candidate('test-driven','Python',a)))
    def test_unrelated_resume_quote_rejected(self):
        a=copy.deepcopy(self.assessment);a['checks'][0]['quote']='Scala'
        self.assertTrue(any('RESUME_QUOTE' in e for e in validate_candidate('Python','Python',a)))
    def test_years_contradiction(self):
        a=copy.deepcopy(self.assessment)
        a['checks'][0]['kind']='experience';a['years']={'required':5,'relevant':1,'certain':True}
        self.assertTrue(any('YEARS_CONTRADICTION' in e for e in validate_candidate('Python','Python',a)))
    def test_missing_keys_rejected(self):
        a=copy.deepcopy(self.assessment);a.pop('review')
        self.assertTrue(any('SCHEMA' in e for e in validate_candidate('Python','Python',a)))
if __name__=='__main__':unittest.main()
