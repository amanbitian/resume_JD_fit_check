import csv
import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rubric_validator import validate_rubric, versioned_cache_key, partition_requirements_for_scoring

with (ROOT/'tests/fixtures/results_v22.csv').open(encoding='utf-8-sig',newline='') as f:
    ROWS={r['jd_id']:r for r in csv.DictReader(f)}

def source(jd_id):
    r=ROWS[jd_id]
    return r['jd'],json.loads(r['output'])

class TargetedRegressionTests(unittest.TestCase):
    def test_schema_identical_to_legacy(self):
        original=json.loads((ROOT/'tests/fixtures/v2_2_schema.json').read_text())
        current=json.loads((ROOT/'jd_rubric.schema.json').read_text())
        self.assertEqual(original,current)

    def test_previous_12_outputs_9_cache_valid_with_new_checks(self):
        valid=[];bad={}
        for key in ROWS:
            jd,rubric=source(key)
            report=validate_rubric(jd,rubric)
            if report.valid_for_cache:valid.append(key)
            else:bad[key]=report.errors+report.warnings
        self.assertEqual(len(valid),9)
        self.assertEqual(set(bad),{'JD-000002','JD-008861','JD-016936'})

    def test_missing_open_group_is_caught(self):
        jd,rubric=source('JD-000002')
        r=validate_rubric(jd,rubric)
        self.assertIn('MISSING_OPEN_ALTERNATIVE_GROUP',str(r.errors))

    def test_correct_open_group_and_logistics_pass(self):
        jd,rubric=source('JD-000002');rubric=deepcopy(rubric)
        # Replace the over-generalized SaaS skill with the two literal examples.
        template=rubric['skills'][-1]
        self.assertEqual(template['name'],'SaaS Development')
        quote='Familiarity with developing on an enterprise-level SaaS (Salesforce, ServiceNow, etc.)'
        self.assertIn(quote,jd)
        template.update(name='Salesforce',raw='Salesforce',any_of_group='G1',source_quote=quote,
                        jd_evidence='Example enterprise SaaS platform')
        second=deepcopy(template);second.update(id='S18',name='ServiceNow',raw='ServiceNow')
        rubric['skills'].append(second)
        rubric['any_of_groups']=[{
            'id':'G1','description':'Enterprise SaaS platforms','member_skill_ids':['S17','S18'],
            'min_required':1,'open_category':True,'category_label':'enterprise_saas',
            'priority':template['priority'],'importance':template['importance'],
            'priority_origin':template['priority_origin'],'hard_gate_explicit':False
        }]
        for r in rubric['requirements']:
            if r['category'] in ('location','work_authorization'):
                r['hard_gate_explicit']=True
        report=validate_rubric(jd,rubric)
        self.assertTrue(report.valid_for_cache,(report.errors,report.warnings))
        fit, logistics=partition_requirements_for_scoring(rubric)
        self.assertEqual(len(logistics),2)
        self.assertFalse(any(r['category'] in ('location','work_authorization') for r in fit))

    def test_generic_problem_solving_omission_caught(self):
        for key in ('JD-008861','JD-016936'):
            jd,rubric=source(key)
            r=validate_rubric(jd,rubric)
            self.assertIn('MISSING_EXPLICIT_OTHER_SKILL',str(r.errors))

    def test_generic_problem_solving_recovered(self):
        jd,rubric=source('JD-008861');rubric=deepcopy(rubric)
        rubric['skills'].append({
            'id':f"S{len(rubric['skills'])+1}", 'name':'Problem Solving','raw':'problem solving',
            'skill_type':'interpersonal','priority':'preferred','importance':'supporting',
            'priority_origin':'inferred','hard_gate_explicit':False,'any_of_group':None,
            'min_years':None,'expected_level':'unspecified','jd_evidence':'Other listed competency',
            'source_quote':'problem solving'})
        r=validate_rubric(jd,rubric)
        self.assertTrue(r.valid_for_cache,(r.errors,r.warnings))

    def test_logistics_not_technical_fit(self):
        jd,rubric=source('JD-000002')
        fit,conditions=partition_requirements_for_scoring(rubric)
        self.assertEqual(len(conditions),2)
        self.assertTrue(all(r['category'] in ('work_authorization','location') for r in conditions))
        self.assertTrue(all(r['category'] not in ('work_authorization','location','travel_shift') for r in fit))

    def test_valid_unchanged_cases_do_not_regress(self):
        for key in ('JD-000016','JD-000314','JD-001951','JD-003899','JD-011997'):
            jd,rubric=source(key)
            report=validate_rubric(jd,rubric)
            self.assertTrue(report.valid_for_cache,(key,report.errors,report.warnings))

    def test_cache_key_changes_on_promotion(self):
        self.assertNotEqual(versioned_cache_key('abc','qwen','v2.2',schema_version='2.2'),
                            versioned_cache_key('abc','qwen','v2.3'))

if __name__=='__main__':unittest.main()
