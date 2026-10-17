import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from gemma_jd_fix.cli import main
from gemma_jd_fix.core import PROBE_SCHEMA

class TestCLI(unittest.TestCase):
    def test_dry_run_without_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            schema_file=base/'schema.json';schema_file.write_text(json.dumps(PROBE_SCHEMA),encoding='utf-8')
            prompt_file=base/'prompt.txt';prompt_file.write_text('Extract one object',encoding='utf-8')
            inputs=base/'jd.csv'
            with open(inputs, 'w',newline='',encoding='utf-8') as h:
                w=csv.DictWriter(h,fieldnames=['jd_id','jd']);w.writeheader();w.writerow({'jd_id':'X1','jd':'Python and AWS'})
            out=base/'results'
            rc=main(['benchmark','--model','test-model','--schema',str(schema_file),
                 '--system-prompt',str(prompt_file),'--input',str(inputs),
                 '--output-dir',str(out),'--dry-run'])
            self.assertEqual(rc,0)
            payload=json.loads((out/'dry_run_payload.json').read_text(encoding='utf-8'))
            self.assertEqual(payload['messages'][0]['content'],'Extract one object')
            self.assertEqual(payload['response_format']['schema'],PROBE_SCHEMA)

    @patch('gemma_jd_fix.cli.send_chat')
    def test_accepted_flag_requires_semantic_hook(self,mock_send):
        mock_send.return_value={'choices':[{'message':{'content':'{"importance":"core"}'}}],
                                'usage':{'completion_tokens':8}}
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            (base/'schema.json').write_text(json.dumps(PROBE_SCHEMA))
            (base/'prompt.txt').write_text('Extract')
            with open(base/'in.csv','w',newline='') as h:
                w=csv.DictWriter(h,fieldnames=['jd_id','jd']);w.writeheader();w.writerow({'jd_id':'X','jd':'Test JD'})
            main(['benchmark','--model','m','--schema',str(base/'schema.json'),
                  '--system-prompt',str(base/'prompt.txt'),'--input',str(base/'in.csv'),
                  '--output-dir',str(base/'out')])
            with open(base/'out'/'results.csv') as h:
                row=next(csv.DictReader(h))
            self.assertEqual(row['schema_ok'],'True')
            self.assertEqual(row['accepted'],'False')
            self.assertEqual(mock_send.call_count,1)

if __name__=='__main__':
    unittest.main()
