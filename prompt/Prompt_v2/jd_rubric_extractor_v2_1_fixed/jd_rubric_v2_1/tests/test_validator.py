import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rubric_validator import validate_rubric,listed_skills,parse_raw,versioned_cache_key

JD='jobtitle: Platform Engineer, experience: "3 years", jobdescription: implement ci/cd; manage docker deployments, education: "bachelor", mustskills: "AWS; Azure", goodskills: "Python, JavaScript", otherskills: "collaboration"'

def good():
  d={
    'input_status':'ok','warnings':[],'suspicious_instructions_detected':False,
    'job_title':'Platform Engineer','normalized_title':'Platform Engineer','role_family':'Platform Engineering','seniority':'mid','seniority_basis':'years','management_scope':'individual_contributor','employment_type':'unclear','work_arrangement':'unclear','locations':[],
    'role_summary':'Manage deployment pipelines.','primary_functional_domain':'devops_cloud_infrastructure','secondary_functional_domains':[],'functional_subdomains':[],
    'primary_industry_domain':None,'secondary_industry_domains':[],'industry_subdomains':[],'domain_rationale':'',
    'key_responsibilities':[{'id':'D1','responsibility':'Implement CI/CD','importance':'core','source_quote':'implement ci/cd'},{'id':'D2','responsibility':'Manage Docker deployments','importance':'core','source_quote':'manage docker deployments'}],
    'skills':[], 'any_of_groups':[], 'top_critical_skill_ids':[],'critical_skill_rationale':[],
    'experience_requirement':{'min_years':3,'max_years':None,'basis':'unclear','priority':'mandatory','hard_gate_explicit':False,'jd_evidence':'3 years'},
    'education_requirement':{'min_level':'bachelor','accepted_fields':[],'priority':'mandatory','equivalent_experience_accepted':False,'hard_gate_explicit':False,'jd_evidence':'bachelor'},
    'requirements':[],'ambiguities':[]}
  candidates=[('AWS','AWS','mandatory','heading','"AWS; Azure"'),('Azure','Azure','mandatory','heading','"AWS; Azure"'),('Python','Python','preferred','heading','"Python, JavaScript"'),('JavaScript','JavaScript','preferred','heading','"Python, JavaScript"'),('CI/CD','ci/cd','mandatory','responsibility','implement ci/cd'),('Docker','docker','mandatory','responsibility','manage docker deployments')]
  for i,(n,r,pr,origin,q) in enumerate(candidates,1):
    d['skills'].append({'id':f'S{i}','name':n,'raw':r,'skill_type':'tool','priority':pr,'importance':'important','priority_origin':origin,'hard_gate_explicit':False,'any_of_group':None,'min_years':None,'expected_level':'unspecified','jd_evidence':n,'source_quote':q})
  return d

class ValidationTests(unittest.TestCase):
 def codes(self,r):return [x.split(':')[0] for x in r.errors]
 def test_reference_passes(self):
  result=validate_rubric(JD,good())
  self.assertEqual(result.errors,[],result.errors)
 def test_schema_drift_blocks(self):
  x=good();x['input_validity']=x.pop('input_status')
  self.assertIn('SCHEMA',self.codes(validate_rubric(JD,x)))
 def test_malformed_enum_blocks(self):
  x=good();x['skills'][0]['priority']='must'
  self.assertIn('SCHEMA',self.codes(validate_rubric(JD,x)))
 def test_quote_must_be_exact(self):
  x=good();x['skills'][0]['source_quote']='AWS ... Azure'
  self.assertIn('ALTERED_QUOTE',self.codes(validate_rubric(JD,x)))
 def test_missing_duty_tool_blocks(self):
  x=good();x['skills']=[s for s in x['skills'] if s['name']!='CI/CD'];
  x['skills'][-1]['id']='S5'
  self.assertIn('MISSING_DUTY_SKILL',self.codes(validate_rubric(JD,x)))
 def test_missing_mustskill_blocks(self):
  x=good();x['skills']=[s for s in x['skills'] if s['name']!='Azure'];
  for i,s in enumerate(x['skills'],1):s['id']=f'S{i}'
  self.assertIn('MISSING_LISTED_SKILL',self.codes(validate_rubric(JD,x)))
 def test_inferred_gate_blocks(self):
  x=good();x['skills'][4]['hard_gate_explicit']=True
  self.assertIn('INFERRED_HARD_GATE',self.codes(validate_rubric(JD,x)))
 def test_group_reference_blocks(self):
  x=good();x['skills'][0]['any_of_group']='G1'
  self.assertIn('UNKNOWN_GROUP',self.codes(validate_rubric(JD,x)))
 def test_string_null_is_wrong(self):
  x=good();x['skills'][0]['min_years']='null'
  self.assertIn('SCHEMA',self.codes(validate_rubric(JD,x)))
 def test_fences_normalize(self):
  self.assertEqual(parse_raw('```json\n{"ok":1}\n```',allow_fences=True),{'ok':1})
 def test_version_cache_changes(self):
  self.assertNotEqual(versioned_cache_key(JD,'qwen','v2'),versioned_cache_key(JD,'qwen','v2.1'))
 def test_structured_list_parser(self):
  self.assertEqual(listed_skills(JD,'mustskills'),['AWS','Azure'])
  self.assertEqual(listed_skills(JD,'goodskills'),['Python','JavaScript'])

if __name__=='__main__':unittest.main()
