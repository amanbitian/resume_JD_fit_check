import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evidence_audit import annotate_quotes

class AuditTests(unittest.TestCase):
    def test_verbatim_offset(self):
        doc={'skills':[{'name':'Python','evidence_quote':'Built APIs with Python'}]}
        result,issues=annotate_quotes(doc,'I Built APIs with Python in 2024','resume')
        self.assertEqual(result['skills'][0]['source_span']['status'],'verified')
        self.assertEqual(issues,[])
    def test_nonexistent_quote_flagged(self):
        doc={'skills':[{'name':'Python','evidence_quote':'Built APIs with Python'}]}
        result,issues=annotate_quotes(doc,'Built reports in Excel','resume')
        self.assertEqual(issues[0]['status'],'unverified')
if __name__=='__main__':unittest.main()
