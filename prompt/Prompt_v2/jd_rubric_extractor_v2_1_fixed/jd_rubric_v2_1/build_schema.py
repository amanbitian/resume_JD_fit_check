import json
from pathlib import Path
ROOT=Path(__file__).parent

def enum(*vals): return {'type':'string','enum':list(vals)}
def text(nullable=False, maxLength=None):
    ans={'type':['string','null'] if nullable else 'string'}
    if maxLength: ans['maxLength']=maxLength
    return ans
def integer(nullable=False, minimum=None):
    ans={'type':['integer','null'] if nullable else 'integer'}
    if minimum is not None: ans['minimum']=minimum
    return ans
def number(nullable=False): return {'type':['number','null'] if nullable else 'number','minimum':0}
def strings(): return {'type':'array','items':{'type':'string'}}
def array(item): return {'type':'array','items':item}
def obj(props, required=True):
    return {'type':'object','properties':props,'required':list(props) if required else [],'additionalProperties':False}
priority=enum('mandatory','preferred')
importance=enum('critical','important','supporting')
origin=enum('explicit','heading','responsibility','inferred')
q=text(maxLength=160)
duty=obj({'id':{'type':'string','pattern':'^D[1-9][0-9]*$'},'responsibility':text(),'importance':enum('core','secondary'),'source_quote':q})
skill=obj({'id':{'type':'string','pattern':'^S[1-9][0-9]*$'},'name':text(),'raw':text(),'skill_type':enum('technical','tool','domain_knowledge','process_method','interpersonal'),'priority':priority,'importance':importance,'priority_origin':origin,'hard_gate_explicit':{'type':'boolean'},'any_of_group':{'type':['string','null'],'pattern':'^G[1-9][0-9]*$'},'min_years':number(True),'expected_level':enum('basic','working','advanced','expert','unspecified'),'jd_evidence':text(),'source_quote':q})
group=obj({'id':{'type':'string','pattern':'^G[1-9][0-9]*$'},'description':text(),'member_skill_ids':strings(),'min_required':integer(minimum=1),'open_category':{'type':'boolean'},'category_label':text(True),'priority':priority,'importance':importance,'priority_origin':origin,'hard_gate_explicit':{'type':'boolean'}})
non_skill=obj({'id':{'type':'string','pattern':'^R[1-9][0-9]*$'},'category':enum('certification','domain_experience','role_experience','management_experience','language','location','work_authorization','travel_shift','other'),'requirement':text(),'priority':priority,'importance':importance,'priority_origin':origin,'hard_gate_explicit':{'type':'boolean'},'min_years':number(True),'jd_evidence':text(),'source_quote':q})
functional='software_engineering devops_cloud_infrastructure quality_assurance_testing data_engineering data_analytics data_science_ai_ml security_risk it_support_administration product_management project_program_management design_creative content_writing_communications marketing sales_business_development customer_success_support finance_accounting human_resources_recruiting operations_supply_chain consulting_strategy legal_compliance research_science teaching_training healthcare_clinical electrical_electronics_engineering mechanical_engineering civil_engineering administrative other'.split()
industry='technology_software it_services_outsourcing edtech_education financial_services_banking fintech_payments insurance healthcare_life_sciences ecommerce_retail_consumer manufacturing_industrial automotive_aerospace energy_utilities telecommunications media_entertainment_gaming advertising_marketing_services professional_services_consulting government_public_sector transportation_logistics real_estate_construction hospitality_travel agriculture nonprofit other'.split()
schema=obj({
'input_status':enum('ok','partial','not_a_jd','empty_or_unreadable','multiple_roles'),
'warnings':strings(),
'suspicious_instructions_detected':{'type':'boolean'},
'job_title':text(True), 'normalized_title':text(True),'role_family':text(True),
'seniority':enum('intern','entry','mid','senior','lead_principal','manager','senior_manager_director','executive','unclear'),
'seniority_basis':enum('title','years','scope','unclear'),
'management_scope':enum('individual_contributor','people_manager','either','unclear'),
'employment_type':enum('full_time','part_time','contract','internship','freelance','unclear'),
'work_arrangement':enum('onsite','hybrid','remote','unclear'),
'locations':strings(),'role_summary':text(),
'primary_functional_domain':{'type':['string','null'],'enum':functional+[None]},
'secondary_functional_domains':array(enum(*functional)),
'functional_subdomains':strings(),
'primary_industry_domain':{'type':['string','null'],'enum':industry+[None]},
'secondary_industry_domains':array(enum(*industry)),
'industry_subdomains':strings(),
'domain_rationale':text(),
'key_responsibilities':array(duty),
'skills':array(skill),
'any_of_groups':array(group),
'top_critical_skill_ids':strings(),
'critical_skill_rationale':array(obj({'id':text(),'why':text()})),
'experience_requirement':obj({'min_years':number(True),'max_years':number(True),'basis':enum('total','relevant','unclear'),'priority':{'type':['string','null'],'enum':['mandatory','preferred',None]},'hard_gate_explicit':{'type':'boolean'},'jd_evidence':text(True)}),
'education_requirement':obj({'min_level':{'type':['string','null'],'enum':['secondary','diploma','bachelor','postgraduate_diploma','master','mba_pgdm','doctorate','other',None]},'accepted_fields':strings(),'priority':{'type':['string','null'],'enum':['mandatory','preferred',None]},'equivalent_experience_accepted':{'type':'boolean'},'hard_gate_explicit':{'type':'boolean'},'jd_evidence':text(True)}),
'requirements':array(non_skill),
'ambiguities':strings(),
})
schema['$schema']='https://json-schema.org/draft/2020-12/schema'
schema['title']='Canonical JD Rubric v2.1 — strict compatibility schema'
schema['description']='All fields required; no extra wrappers or renamed keys. Source quote and cross-field integrity require separate application validation.'
(ROOT/'jd_rubric.schema.json').write_text(json.dumps(schema,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print('Wrote schema; top-level fields:', len(schema['properties']))
