import json,sys,unittest
from copy import deepcopy
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rubric_validator import validate_rubric,versioned_cache_key
ROWS={r['item_id']:r for r in pd.read_csv(ROOT/'tests'/'fixtures'/'results_full_text.csv').to_dict('records')}

def source(jd_id):
    r=ROWS[jd_id];return r['jd'],json.loads(r['output'])

class SemanticRefinementTests(unittest.TestCase):
    def test_historical_schema_is_unchanged(self):
        original=json.loads((ROOT/'tests'/'fixtures'/'v2_1_schema.json').read_text())
        revised=json.loads((ROOT/'jd_rubric.schema.json').read_text())
        self.assertEqual(original,revised)

    def test_intern_incorrectly_completing_degree_is_rejected(self):
        jd,rubric=source('JD-000016')
        report=validate_rubric(jd,rubric)
        self.assertIn('ENROLLMENT_AS_COMPLETED_DEGREE',str(report.errors))
        self.assertIn('ENROLLMENT_CONDITION_MISSING',str(report.errors))

    def test_corrected_enrollment_rubric_is_accepted(self):
        jd,rubric=source('JD-000016')
        rubric=deepcopy(rubric)
        rubric['education_requirement']={'min_level':None,'accepted_fields':[],'priority':None,
           'equivalent_experience_accepted':False,'hard_gate_explicit':False,'jd_evidence':None}
        phrase='Currently studying for a degree in Computer Science, Software Engineering, or Computer Engineering'
        self.assertIn(phrase,jd)
        rubric['requirements'].append({'id':'R3','category':'other',
           'requirement':'Currently studying toward a degree in Computer Science, Software Engineering or Computer Engineering',
           'priority':'mandatory','importance':'important','priority_origin':'heading',
           'hard_gate_explicit':False,'min_years':None,'jd_evidence':'Current enrollment in specified degree fields',
           'source_quote':phrase})
        report=validate_rubric(jd,rubric)
        self.assertTrue(report.valid_for_cache, report.errors)

    def test_short_work_request_is_not_not_a_jd(self):
        jd,rubric=source('JD-000314')
        report=validate_rubric(jd,rubric)
        self.assertIn('MISCLASSIFIED_INFORMAL_JD',str(report.errors))
        self.assertFalse(report.valid_for_cache)

    def test_short_work_request_as_partial_with_duty_validates(self):
        jd,rubric=source('JD-000314')
        rubric=deepcopy(rubric)
        rubric.update(input_status='partial',warnings=['Informal work request; qualifications unstated'],
                      role_summary='Accounting and tax support for a small business',
                      primary_functional_domain='finance_accounting')
        rubric['key_responsibilities']=[{'id':'D1','responsibility':'Help with accounting and taxes','importance':'core','source_quote':'needs help with accountingtaxes'}]
        report=validate_rubric(jd,rubric)
        self.assertTrue(report.valid_for_cache,report.errors)

    def test_generic_otherskills_mandatory_flags_review(self):
        jd,rubric=source('JD-001951')
        report=validate_rubric(jd,rubric)
        self.assertTrue(any('OTHER_SKILLS_PRIORITY_REVIEW' in w for w in report.warnings))
        self.assertTrue(report.needs_rubric_review)
        self.assertFalse(report.valid_for_cache)

    def test_generic_otherskills_fixed_priority_passes(self):
        jd,rubric=source('JD-001951')
        rubric=deepcopy(rubric)
        for s in rubric['skills']:
            if s['name'].lower() in ('problem solving','effective communication','technical writing'):
                s['priority']='preferred';s['priority_origin']='inferred';s['importance']='supporting'
        # Three generic skills may merit human review even when preferred; validator only guards hard overclaim.
        report=validate_rubric(jd,rubric)
        self.assertTrue(report.valid_for_cache,(report.errors,report.warnings))

    def test_valid_existing_JDs_remain_valid(self):
        for key in ('JD-000160','JD-000753','JD-001896','JD-003899','JD-011997','JD-013090','JD-016936'):
            jd,rubric=source(key)
            report=validate_rubric(jd,rubric)
            self.assertTrue(report.valid_for_cache,(key,report.errors,report.warnings))

    def test_cache_key_changes_with_version(self):
        self.assertNotEqual(versioned_cache_key('JD','qwen','v2.1',schema_version='2.1'),versioned_cache_key('JD','qwen','v2.2'))

if __name__=='__main__':unittest.main()
