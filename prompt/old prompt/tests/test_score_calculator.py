import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from score_calculator import recompute, merge_assessment


def fixture():
    jd={
      'input_status':'ok','seniority':'mid',
      'skills':[{'id':'S1','name':'Python','priority':'mandatory','importance':'critical','hard_gate_explicit':True,'any_of_group':None}],
      'any_of_groups':[], 'key_responsibilities':[{'id':'D1','importance':'core'}],
      'requirements':[],
      'experience_requirement':{'min_years':5,'basis':'relevant','priority':'mandatory','hard_gate_explicit':True},
      'education_requirement':{'min_level':None,'priority':None,'hard_gate_explicit':False},
    }
    resume={
      'input_status':'ok','suspicious_instructions_detected':False,
      'experience_summary':{'date_quality':'complete'},
      'skills':[{'name':'Python'}],
      'work_experiences':[{'id':'E1','employment_type':'full_time','start':'2020-01','end':'2024-12'}],
      'projects':[],'education':[],'certifications':[],
    }
    a={
      'input_check':{'status':'ok'},
      'skill_matches':[{'jd_skill_id':'S1','status':'met','match_type':'exact','resume_skill':'Python',
                       'resume_refs':['E1'],'modifiers':[],'match_score':10}],
      'group_matches':[], 'responsibility_matches':[{'jd_responsibility_id':'D1','status':'demonstrated'}],
      'requirement_matches':[], 'education_match':{'status':'not_applicable'},
      'experience_relevance':[{'resume_experience_id':'E1','relevance':'high'}],
      'alignment':{'role':'exact','functional':'exact','seniority_fit':'match','industry':'not_applicable'},
      'needs_human_review':False,'confidence':.9,
    }
    return jd,resume,a


class ScoringTests(unittest.TestCase):
    def test_full_five_years(self):
        jd,r,a=fixture()
        out=recompute(jd,r,a,'2026-10-17')
        self.assertEqual(out['final_score'],10)
        self.assertEqual(out['years']['total_years'],5)
        self.assertEqual(out['assessment_status'],'scored')

    def test_unscorable_not_no_fit(self):
        jd,r,a=fixture();r['input_status']='empty_or_unreadable'
        out=recompute(jd,r,a,'2026-10-17')
        self.assertIsNone(out['final_score'])
        self.assertEqual(out['fit_band'],'unscorable')

    def test_raw_ratio_threshold_not_display_rounding(self):
        jd,r,a=fixture()
        r['work_experiences'][0]['end']='2023-11' # 47 months
        jd['experience_requirement']['min_years']=4.9
        out=recompute(jd,r,a,'2026-10-17')
        self.assertEqual(out['years']['ratio'],.80) # rendered 0.79932 rounds to .80
        self.assertIn(('CAP_YEARS',6),out['caps']) # raw threshold still < .8

    def test_undated_exp_does_not_trigger_missing_years_cap(self):
        jd,r,a=fixture()
        r['work_experiences'][0]['start']='2020'
        r['experience_summary']['date_quality']='partial'
        out=recompute(jd,r,a,'2026-10-17')
        self.assertIsNone(out['dimension_scores']['years'])
        self.assertFalse(any(name=='CAP_YEARS' for name,_ in out['caps']))
        self.assertTrue(out['needs_human_review'])

    def test_inferred_requirement_no_hard_cap(self):
        jd,r,a=fixture()
        jd['skills'][0]['hard_gate_explicit']=False
        a['skill_matches'][0].update(status='not_evidenced',match_type='none',resume_skill=None,resume_refs=[],match_score=0)
        out=recompute(jd,r,a,'2026-10-17')
        self.assertFalse(any(name=='CAP_CRITICAL_MISSING' for name,_ in out['caps']))
        self.assertTrue(out['needs_human_review'])

    def test_critical_explicit_missing_gets_cap(self):
        jd,r,a=fixture()
        a['skill_matches'][0].update(status='not_evidenced',match_type='none',resume_skill=None,resume_refs=[],match_score=0)
        out=recompute(jd,r,a,'2026-10-17')
        self.assertIn(('CAP_CRITICAL_MISSING',5),out['caps'])

    def test_project_only_mandatory_critical(self):
        jd,r,a=fixture()
        r['projects']=[{'id':'P1'}]
        a['skill_matches'][0]['resume_refs']=['P1']
        a['skill_matches'][0]['modifiers']=['project_only']
        a['skill_matches'][0]['match_score']=8
        out=recompute(jd,r,a,'2026-10-17')
        self.assertIn(('CAP_CRITICAL_PARTIAL',6),out['caps'])

    def test_entry_internships_count_only_relevant(self):
        jd,r,a=fixture()
        jd['seniority']='entry'; jd['experience_requirement']['min_years']=.25
        r['work_experiences']=[{'id':'E1','employment_type':'internship','start':'2023-01','end':'2023-12'}]
        out=recompute(jd,r,a,'2026-10-17')
        self.assertEqual(out['years']['total_years'],0)
        self.assertEqual(out['years']['relevant_years'],.5)
        self.assertEqual(out['dimension_scores']['years'],10)

    def test_overlapping_not_double_counted(self):
        jd,r,a=fixture()
        r['work_experiences'] += [{'id':'E2','employment_type':'full_time','start':'2023-01','end':'2024-12'}]
        a['experience_relevance'] += [{'resume_experience_id':'E2','relevance':'high'}]
        out=recompute(jd,r,a,'2026-10-17')
        self.assertEqual(out['years']['total_years'],5)
        self.assertEqual(out['years']['relevant_years'],5)

    def test_future_end_never_awards_future_months(self):
        jd,r,a=fixture()
        r['work_experiences'][0]['end']='2030-01'
        out=recompute(jd,r,a,'2026-10-17')
        self.assertLessEqual(out['years']['total_years'],6.9)
        self.assertTrue(out['needs_human_review'])
        self.assertFalse(any(name=='CAP_YEARS' for name,_ in out['caps']))

    def test_missing_match_rejected(self):
        jd,r,a=fixture()
        a['skill_matches']=[]
        with self.assertRaises(ValueError):recompute(jd,r,a,'2026-10-17')

    def test_result_merge_replaces_model_arithmetic(self):
        jd,r,a=fixture();a['final_score']=1
        c=recompute(jd,r,a,'2026-10-17')
        merged=merge_assessment(a,c)
        self.assertEqual(merged['final_score'],10)
        self.assertEqual(merged['fit_band'],'strong_fit')

    def test_missing_reference_date_for_present(self):
        jd,r,a=fixture();r['work_experiences'][0]['end']='present'
        out=recompute(jd,r,a)
        self.assertIsNone(out['dimension_scores']['years'])
        self.assertTrue(out['needs_human_review'])


if __name__=='__main__':unittest.main()
