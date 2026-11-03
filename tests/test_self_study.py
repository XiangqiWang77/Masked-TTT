import json
import unittest

from masked_ttt.request import Request
from masked_ttt.self_study import (answer_is_in_quote,parse_exercises,practice_request,
                                   short_answer_equivalent,teacher_practice_request,
                                   verified_answer_hint)


class SelfStudyTests(unittest.TestCase):
    def setUp(self):
        self.parts=['X项目原定预算300万元。','X项目最终批准预算320万元。','Y项目最终批准预算250万元。']
        self.document=''.join(self.parts);self.spans=[];offset=0
        for part in self.parts:
            self.spans.append((offset,offset+len(part)));offset+=len(part)

    def test_selected_quote_and_answer_are_required(self):
        items=[
            dict(question='X项目最终批准预算是多少？',answer='320',evidence=self.parts[1]),
            dict(question='X原定预算是多少？',answer='300',evidence=self.parts[0]),
            dict(question='Y项目最终预算是多少？',answer='250',evidence=self.parts[2]),
        ]
        text='```json\n'+json.dumps({'exercises':items},ensure_ascii=False)+'\n```'
        got,rejected=parse_exercises(text,self.document,self.spans,[1,2],'两者相差多少？',limit=3)
        self.assertEqual([x.answer for x in got],['320','250'])
        self.assertEqual(rejected['quote_not_in_selected_chunk'],1)

    def test_numbers_must_not_match_inside_other_numbers(self):
        self.assertTrue(answer_is_in_quote('320','X预算为320万元。'))
        self.assertFalse(answer_is_in_quote('320','X预算为1320万元。'))
        self.assertFalse(answer_is_in_quote('320','X预算为3200万元。'))
        self.assertFalse(answer_is_in_quote('320','X预算为320.5万元。'))

    def test_independent_short_answer_comparison_is_not_substring(self):
        self.assertTrue(short_answer_equivalent('kitchen','The answer is the kitchen.'))
        self.assertTrue(short_answer_equivalent('one','1'))
        self.assertFalse(short_answer_equivalent('320','not 320'))
        self.assertFalse(short_answer_equivalent('320','320 or 250'))
        self.assertTrue(short_answer_equivalent('7-year-old','7 years old.'))

    def test_sentence_answers_require_a_corroborated_unambiguous_extract(self):
        self.assertEqual(verified_answer_hint('Sandra moved to the garden.',
            'The garden.','Sandra moved to the garden.'),'garden')
        self.assertEqual(verified_answer_hint('John went to the bedroom.',
            'John is in the bedroom.','John went to the bedroom.'),'bedroom')
        self.assertEqual(verified_answer_hint('320','The answer is 320.',
            'The final budget was 320.'),'320')
        for proposed,verified,evidence in [
            ('not 320','320','The budget was not 320.'),
            ('320 or 250','320','The budget was 320 or 250.'),
            ('320 and 250','320','X was 320 and Y was 250.'),
            ('went to garden','kitchen','went to garden'),
            ('went to garden','unknown','went to garden'),
            ('went to garden','garden','went to office'),
        ]:
            self.assertIsNone(verified_answer_hint(proposed,verified,evidence))

    def test_practice_question_keeps_document_and_does_not_include_main_answer(self):
        original=Request(document=self.document,question='两者相差多少？',
            prompt='Context: '+self.document+'\nQuestion: 两者相差多少？',max_new_tokens=20)
        practice=practice_request(original,'X最终预算是多少？')
        self.assertEqual(practice.document,original.document)
        self.assertIn('X最终预算是多少？',practice.prompt)
        self.assertNotIn('两者相差多少？',practice.prompt)
        self.assertNotIn('70',practice.prompt)
        teacher=teacher_practice_request(practice,'320')
        self.assertIn('source-supported short answer is: 320',teacher.prompt)
        self.assertNotIn('source-supported short answer',practice.prompt)

    def test_invalid_json_does_not_invent_exercises(self):
        exercises,rejected=parse_exercises('not json',self.document,self.spans,[1],
                                           '两者相差多少？')
        self.assertFalse(exercises);self.assertEqual(rejected,{'invalid_json':1})

    def test_truncated_outer_json_recovers_only_complete_grounded_exercises(self):
        first=dict(question='X最终预算是多少？',answer='320',evidence=self.parts[1])
        text='{"exercises":['+json.dumps(first,ensure_ascii=False)+', {"question":"unfinished'
        exercises,rejected=parse_exercises(text,self.document,self.spans,[1],
                                           '两者相差多少？',limit=2)
        self.assertEqual(len(exercises),1)
        self.assertEqual(exercises[0].answer,'320')
        self.assertEqual(rejected,{'partial_json_recovered':1})
        no_array=json.dumps(first,ensure_ascii=False)+' trailing non-json'
        exercises,rejected=parse_exercises(no_array,self.document,self.spans,[1],'两者相差多少？')
        self.assertFalse(exercises);self.assertEqual(rejected,{'invalid_json':1})


if __name__=='__main__':unittest.main()
