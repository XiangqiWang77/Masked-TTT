"""Deterministic, text-only contracts; no Torch, model loading, or training."""
import asyncio
import json
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from contextualized.answer_guided_questions import (
    build_answer_guided_dataset, direct_answer_leakage, focused_evidence_view, frozen_text_call,
    literal_quotes, parse_object_with_status, propose_evidence_clusters, unsupported_question_claim,
)


def cluster(index='e1', text='Sandra moved to the kitchen.'):
    return {'id': index, 'text': text,
            'source_chunks': [{'source_index': 4, 'span': [12, 12+len(text)], 'text': text}]}


def answer(value='Sandra is in the kitchen.', quote='Sandra moved to the kitchen.'):
    return {'skip': False, 'answer': value, 'quotes': [quote], 'why_useful': 'Establish the relevant location.'}


class FakeCalls:
    def __init__(self, *outputs): self.outputs = iter(outputs); self.calls = []
    async def __call__(self, engine, instruction, payload, **kwargs):
        self.calls.append((instruction, payload, kwargs))
        value = next(self.outputs)
        if isinstance(value, tuple): return value[0]
        return {'response_count': 1, 'raw_output': json.dumps(value), 'prompt_tokens': 10, 'completion_tokens': 5}


def build(clusters, fake, count=3, **kwargs):
    return asyncio.run(build_answer_guided_dataset(NS(), 'Where is Sandra?', clusters,
        max_examples=count, call_model=fake, **kwargs))


class AnswerGuidedTests(unittest.TestCase):
    def test_cluster_proposal_builds_literal_single_and_cross_source_groups(self):
        units = [
            {'id': '2', 'source_index': 2, 'span': [10, 10+len('Sandra entered the office.')],
             'text': 'Sandra entered the office.'},
            {'id': '11', 'source_index': 11, 'span': [80, 80+len('Sandra moved to the garden.')],
             'text': 'Sandra moved to the garden.'},
        ]
        fake = FakeCalls({'clusters': [{'source_ids': [2]}, {'source_ids': ['11', '2']}]})
        result = asyncio.run(propose_evidence_clusters(NS(), 'Where is Sandra?', units, call_model=fake))
        self.assertEqual([c['id'] for c in result['clusters']],
                         ['cluster_2_11', 'cluster_2', 'cluster_11'])
        self.assertEqual(result['clusters'][0]['source_chunks'],
                         [{**unit, 'query_anchor': True} for unit in units])
        self.assertIn('[Source 11]', result['clusters'][0]['text'])
        self.assertEqual(result['audit']['accepted_count'], 3)
        self.assertEqual(result['audit']['retrieval_seeded_count'], 3)
        self.assertEqual(result['audit']['model_accepted_count'], 0)

    def test_cluster_proposal_rejects_unknown_duplicate_and_oversized_groups(self):
        units = [cluster('a')['source_chunks'][0], cluster('b')['source_chunks'][0]]
        for i, unit in enumerate(units): unit.update(id=str(i), source_index=i)
        fake = FakeCalls({'clusters': [{'source_ids': ['0']}, {'source_ids': ['0']},
                                        {'source_ids': ['missing']}, {'source_ids': ['0', '1']}]})
        result = asyncio.run(propose_evidence_clusters(NS(), 'Where?', units,
            max_chunks_per_cluster=1, call_model=fake))
        self.assertEqual(len(result['clusters']), 2)
        self.assertEqual([r['reason'] for r in result['audit']['rejections']],
                         ['duplicate_cluster', 'duplicate_cluster', 'unknown_source_id',
                          'cluster_chunk_budget_exceeded'])

    def test_query_anchors_are_retained_paired_and_unanchored_model_cluster_is_rejected(self):
        units = [
            {'id': 'a', 'source_index': 0, 'span': [0, 20], 'text': 'Sandra entered town.',
             'lexical_score': 1.0},
            {'id': 'b', 'source_index': 1, 'span': [30, 51], 'text': 'Sandra left the hall.',
             'lexical_score': 2.0},
            {'id': 'noise', 'source_index': 2, 'span': [60, 79], 'text': 'Henry forged steel.',
             'lexical_score': 0.0},
        ]
        fake = FakeCalls({'clusters': [{'source_ids': ['noise']},
                                        {'source_ids': ['a', 'noise']}]})
        result = asyncio.run(propose_evidence_clusters(NS(), 'Where is Sandra?', units,
            max_clusters=8, call_model=fake))
        self.assertEqual([c['id'] for c in result['clusters']],
            ['cluster_a_b', 'cluster_b', 'cluster_a', 'cluster_a_noise'])
        self.assertEqual(result['audit']['query_anchor_source_ids'], ['a', 'b'])
        self.assertIn('missing_query_anchor', [r['reason'] for r in result['audit']['rejections']])

    def test_entity_linked_anchor_bridge_pairs_are_deterministic(self):
        units = [
            {'id': 'old', 'source_index': 0, 'span': [0, 32],
             'text': 'John discarded the football.', 'lexical_score': 2.0, 'bridge_score': 1.0},
            {'id': 'held', 'source_index': 1, 'span': [40, 73],
             'text': 'John picked up the football.', 'lexical_score': 1.0, 'bridge_score': 1.0},
            {'id': 'move', 'source_index': 2, 'span': [80, 105],
             'text': 'John went to the bedroom.', 'lexical_score': 0.0, 'bridge_score': 1.5},
        ]
        # Correct the deliberately explicit spans to the exact source lengths.
        for unit in units:
            unit['span'][1] = unit['span'][0] + len(unit['text'])
        result = asyncio.run(propose_evidence_clusters(NS(), 'Where is the football?', units,
            max_clusters=12, call_model=FakeCalls({'clusters': []})))
        ids = [row['id'] for row in result['clusters']]
        self.assertIn('cluster_old_held_move', ids)
        self.assertIn('cluster_held_move', ids)
        self.assertLess(ids.index('cluster_held_move'), len(ids))

    def test_answer_first_then_closed_book_question_preserves_text_and_provenance(self):
        source = cluster(); fake = FakeCalls(answer(), {'question': 'Where is Sandra?'})
        result = build([source], fake)
        self.assertEqual(len(fake.calls), 2)
        self.assertNotIn('extracted_answer', fake.calls[0][1])
        self.assertEqual(fake.calls[1][1]['extracted_answer'], 'Sandra is in the kitchen.')
        example = result['examples'][0]
        self.assertEqual(example['evidence'], source)
        self.assertEqual(example['question'], 'Where is Sandra?')
        quote = example['answer_source_quotes'][0]
        self.assertEqual(quote['matches'][0]['source_index'], 4)
        self.assertEqual(quote['matches'][0]['chunk_start'], 0)
        self.assertEqual(quote['matches'][0]['document_start'], 12)
        self.assertFalse(example['source_facts_verified'])
        self.assertEqual(example['semantic_status'], 'model_estimate')
        self.assertEqual(result['audit']['shortfall'], 2)
        self.assertEqual(result['audit']['cost']['prompt_tokens']['total'], 20)
        question_input = result['audit']['calls'][1]['input']
        self.assertEqual(question_input['evidence'], fake.calls[0][1]['evidence'])
        self.assertEqual(question_input['evidence_id'], source['id'])
        self.assertEqual(question_input['supporting_quotes'], ['Sandra moved to the kitchen.'])
        json.dumps(result, allow_nan=False)

    def test_simple_two_stage_requires_entailment_and_preserves_disjunction(self):
        source = cluster(text='Julie is either in the kitchen or the bedroom.')
        fake = FakeCalls(
            {'skip': False, 'answer': 'It is uncertain whether Julie is in the kitchen.',
             'quotes': [source['text']], 'why_useful': 'The alternative is unresolved.'},
            {'questions': ['Is Julie definitely in the kitchen?']})
        result = build([source], fake, count=1, simple_two_stage=True)
        instruction = fake.calls[0][0]
        self.assertIn('logically entailed', instruction)
        self.assertIn('either A or B', instruction)
        self.assertIn('uncertain or only possible', instruction)
        self.assertIn('a later explicit event supersedes an earlier state', instruction)
        self.assertEqual(result['examples'][0]['answer'],
                         'It is uncertain whether Julie is in the kitchen.')

    def test_simple_two_stage_skips_no_information_meta_answer(self):
        source = cluster()
        fake = FakeCalls({
            'skip': False,
            'answer': 'The evidence does not provide any information about Sandra.',
            'quotes': [source['text']],
            'why_useful': 'claims absence',
        })
        result = build([source], fake, count=1, simple_two_stage=True)
        self.assertFalse(result['examples'])
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(result['audit']['rejections'][-1]['reason'],
                         'no_query_relevant_fact_in_answer')

    def test_simple_two_stage_rejects_answer_type_and_source_deictic_questions(self):
        source = cluster()
        fake = FakeCalls(
            {'skip': False, 'answer': 'Yes, Sandra is in the kitchen.',
             'quotes': [source['text']], 'why_useful': 'location'},
            {'questions': ['Where is Sandra?',
                           'Does the evidence confirm Sandra is in the kitchen?',
                           'Is Sandra in the kitchen according to the information provided?']})
        result = build([source], fake, count=2, questions_per_answer=2,
                       simple_two_stage=True)
        self.assertFalse(result['examples'])
        reasons = [row['reason'] for row in result['audit']['rejections']]
        self.assertIn('question_answer_type_mismatch', reasons)
        self.assertIn('evidence_deictic_question', reasons)
        self.assertEqual(fake.calls[1][1]['requested_question_candidates'], 8)

    def test_simple_two_stage_rejects_question_that_reveals_supporting_fact(self):
        text = 'Julie is either in the school or the cinema.'
        source = cluster(text=text)
        fake = FakeCalls(
            {'skip': False, 'answer': 'No, Julie is not in the bedroom.',
             'quotes': [text], 'why_useful': 'rules out a third location'},
            {'questions': [
                'If Julie is either in the school or the cinema, is she in the bedroom?'
            ]})
        result = build([source], fake, count=1, simple_two_stage=True)
        self.assertFalse(result['examples'])
        self.assertEqual(result['audit']['rejections'][-1]['reason'],
                         'supporting_fact_revealed_in_question')

    def test_query_complete_core_metadata_is_preserved_in_each_example(self):
        source = cluster()
        source['origin'] = 'test+query_complete_core'
        source['query_complete_core_source_indices'] = [4]
        fake = FakeCalls(answer(), {'questions': ['Where is Sandra?']})
        result = build([source], fake, count=1, simple_two_stage=True)
        evidence = result['examples'][0]['evidence']
        self.assertEqual(evidence['query_complete_core_source_indices'], [4])
        self.assertEqual(evidence['origin'], 'test+query_complete_core')

    def test_answer_extraction_sees_query_sentences_not_anchor_chunk_distractors(self):
        text = 'Henry forged a sword. Sandra moved to the kitchen. A battle followed.'
        source = cluster(text=text)
        fake = FakeCalls(answer(quote='"Sandra moved to the kitchen."'),
                         {'questions': ['Where did Sandra move?']})
        result = build([source], fake)
        view = fake.calls[0][1]['evidence']['text']
        self.assertIn('Sandra moved to the kitchen.', view)
        self.assertNotIn('Henry forged', view)
        self.assertNotIn('battle followed', view)
        self.assertEqual(result['examples'][0]['answer_source_quotes'][0]['quote'],
                         'Sandra moved to the kitchen.')
        self.assertEqual(result['examples'][0]['answer_source_quotes'][0]['reported_quote'],
                         '"Sandra moved to the kitchen."')

    def test_question_crafting_receives_evidence_and_restores_quote_chronology(self):
        pickup, movement = 'John picked up the football there.', 'John went to the bedroom.'
        source = {'id': 'joint', 'text': f'[Source 9]\n{pickup}\n\n[Source 35]\n{movement}',
            'source_chunks': [
                {'id': '9', 'source_index': 9, 'span': [10, 10+len(pickup)], 'text': pickup},
                {'id': '35', 'source_index': 35, 'span': [100, 100+len(movement)], 'text': movement}]}
        fake = FakeCalls({'skip': False, 'answer': 'the bedroom', 'quotes': [movement, pickup],
                          'why_useful': 'John carries the football on the later movement.'},
                         {'questions': ['Where did John take the football?']})
        result = asyncio.run(build_answer_guided_dataset(NS(), 'Where is the football?', [source],
                            max_examples=1, call_model=fake))
        payload = fake.calls[1][1]
        self.assertEqual(payload['evidence'], fake.calls[0][1]['evidence'])
        self.assertEqual(payload['supporting_quotes'], [pickup, movement])
        self.assertEqual([c['source_index'] for c in payload['evidence']['source_excerpts']], [9, 35])
        self.assertIn('carries', payload['answer_reasoning'])
        self.assertEqual(result['audit']['calls'][0]['parsed']['quotes'], [movement, pickup])
        self.assertEqual(result['examples'][0]['evidence'], source)

    def test_sentence_selection_is_not_a_closed_book_knowledge_question(self):
        fake = FakeCalls(answer('the kitchen'), {'questions': ['Which sentence gives Sandra\'s location?']})
        result = build([cluster()], fake)
        self.assertFalse(result['examples'])
        self.assertEqual(result['audit']['rejections'][-1]['reason'], 'evidence_deictic_question')

    def test_bridge_excerpt_does_not_follow_unresolved_there(self):
        anchor = {'id': 'a', 'source_index': 0, 'span': [0, 31],
                  'text': 'John picked up the football there.'}
        bridge_text = 'A tower was there. John went to the bedroom. Trains continued.'
        bridge = {'id': 'b', 'source_index': 1, 'span': [40, 40+len(bridge_text)],
                  'text': bridge_text}
        view = focused_evidence_view({'id': 'joint', 'source_chunks': [anchor, bridge]},
                                     'Where is the football?')
        self.assertIn('John went to the bedroom.', view['text'])
        self.assertNotIn('A tower was there.', view['text'])
        self.assertNotIn('Trains continued.', view['text'])

    def test_multiple_question_variants_share_one_answer_evidence_and_two_calls(self):
        questions = ['Where is Sandra?', 'Which place contains Sandra?',
                     "Name Sandra's location.", 'At what place can Sandra be found?']
        source = cluster(); fake = FakeCalls(answer(), {'questions': questions})
        result = build([source], fake, count=4, questions_per_answer=4)
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual([row['question'] for row in result['examples']], questions)
        self.assertEqual({row['answer'] for row in result['examples']}, {'Sandra is in the kitchen.'})
        self.assertTrue(all(row['evidence'] == source for row in result['examples']))
        self.assertTrue(all(row['call_indices'] == [0, 1] for row in result['examples']))
        self.assertEqual(result['audit']['decisions'][0]['example_indices'], [0, 1, 2, 3])
        self.assertEqual(result['audit']['shortfall'], 0)

    def test_multi_source_target_requires_literal_support_from_two_sources(self):
        first, second = 'Sandra travelled to the office.', 'Sandra later moved to the garden.'
        evidence = {'id': 'joint', 'text': f'[Source 1]\n{first}\n\n[Source 2]\n{second}',
            'source_chunks': [
                {'source_index': 1, 'span': [10, 10+len(first)], 'text': first},
                {'source_index': 2, 'span': [80, 80+len(second)], 'text': second},
            ]}
        weak = FakeCalls(answer('Sandra is now in the garden.', second))
        rejected = build([evidence], weak)
        self.assertFalse(rejected['examples'])
        self.assertEqual(rejected['audit']['rejections'][0]['reason'], 'multi_source_answer_not_joint')
        self.assertEqual(len(weak.calls), 1)
        joint_answer = {'skip': False, 'answer': 'Sandra is now in the garden after previously visiting the office.',
                        'quotes': [first, second], 'why_useful': 'Combines the successive locations.'}
        strong = FakeCalls(joint_answer, {'questions': ['Where is Sandra after her successive moves?']})
        accepted = build([evidence], strong)
        self.assertEqual(len(accepted['examples']), 1)
        self.assertEqual(accepted['audit']['calls'][1]['input']['supporting_quotes'], [first, second])

    def test_multi_source_quotes_must_form_a_connected_entity_chain(self):
        first, second = 'Mary journeyed to the garden.', 'Sandra got the apple there.'
        evidence = {'id': 'disconnected', 'text': first+' '+second, 'source_chunks': [
            {'source_index': 1, 'span': [0, len(first)], 'text': first},
            {'source_index': 2, 'span': [len(first)+1, len(first)+1+len(second)], 'text': second},
        ]}
        fake = FakeCalls({'skip': False, 'answer': 'the garden', 'quotes': [first, second],
                          'why_useful': 'claims a location'})
        result = build([evidence], fake)
        self.assertFalse(result['examples'])
        self.assertEqual(result['audit']['rejections'][0]['reason'],
                         'multi_source_quotes_disconnected')

    def test_question_variant_dedup_and_leakage_are_filtered_without_repair(self):
        fake = FakeCalls(answer('kitchen'), {'questions': [
            'Where is Sandra?', '  WHERE is Sandra?  ', 'Is Sandra in the kitchen?',
            "Name Sandra's location."]})
        result = build([cluster()], fake, count=4, questions_per_answer=4)
        self.assertEqual([row['question'] for row in result['examples']],
                         ['Where is Sandra?', "Name Sandra's location."])
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual([row['reason'] for row in result['audit']['rejections']],
                         ['duplicate_question', 'literal_answer_phrase_in_question'])

    def test_question_variants_cannot_invent_time_deixis_or_answer_type(self):
        fake = FakeCalls(answer(), {'questions': [
            'Where is Sandra currently located?',
            'Where is Sandra according to the statement?',
            'What action did Sandra take?',
            'Where is Sandra physically located?',
        ]})
        result = build([cluster()], fake, count=4, questions_per_answer=4)
        self.assertEqual([row['question'] for row in result['examples']],
                         ['Where is Sandra physically located?'])
        self.assertEqual([row['reason'] for row in result['audit']['rejections']],
                         ['unsupported_temporal_qualifier', 'evidence_deictic_question',
                          'question_answer_type_mismatch'])
        self.assertIsNone(unsupported_question_claim(
            'Where was the apple before the bathroom?', 'the bedroom',
            'Where was the apple before the bathroom?'))

    def test_explicit_skip_and_wrong_cluster_quote_do_not_get_question_calls(self):
        fake = FakeCalls({'skip': True, 'reason': 'No useful fact.'}, answer(quote='A fact from another cluster.'))
        result = build([cluster('skip'), cluster('bad')], fake)
        self.assertEqual(result['examples'], [])
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual([r['reason'] for r in result['audit']['rejections']],
                         ['model_explicit_skip', 'quote_not_in_this_evidence'])
        self.assertEqual(result['audit']['shortfall'], 3)

    def test_invalid_format_answer_leakage_and_exact_duplicates_are_skipped_without_repair(self):
        fake = FakeCalls(answer('kitchen'), {'question': 'Where is Sandra, in the kitchen?'},
                         answer(), {'question': 'The location is the kitchen.'},
                         answer(), {'question': 'Where is Sandra?'},
                         answer(), {'question': '  WHERE is Sandra?  '})
        result = build([cluster(str(i)) for i in range(4)], fake)
        self.assertEqual(len(result['examples']), 1)
        self.assertEqual(len(fake.calls), 8)
        self.assertEqual([r['reason'] for r in result['audit']['rejections']],
            ['literal_answer_phrase_in_question', 'declarative_answer_in_request_role', 'duplicate_question'])
        self.assertEqual(result['audit']['rejections'][-1]['previous_evidence_id'], '2')

    def test_budget_stops_in_order_and_records_unvisited_clusters(self):
        fake = FakeCalls(answer(), {'question': 'Where is Sandra?'})
        result = build([cluster('first'), cluster('unvisited')], fake, count=1)
        self.assertEqual([row['evidence']['id'] for row in result['examples']], ['first'])
        self.assertEqual(result['audit']['unprocessed_evidence_ids'], ['unvisited'])
        self.assertEqual(result['audit']['shortfall'], 0)
        self.assertEqual(len(fake.calls), 2)

    def test_missing_and_malformed_response_costs_remain_visible(self):
        fake = FakeCalls(({'response_count': 0, 'raw_output': None, 'prompt_tokens': 11},),
                         ({'response_count': 1, 'raw_output': '{"skip":false,', 'prompt_tokens': 9, 'completion_tokens': 7},))
        result = build([cluster('missing'), cluster('malformed')], fake)
        self.assertEqual(len(result['audit']['calls']), 2)
        self.assertEqual([c['parse_status'] for c in result['audit']['calls']], ['unknown', 'unknown'])
        self.assertEqual(result['audit']['calls'][1]['raw_output'], '{"skip":false,')
        self.assertIsNone(result['audit']['cost']['completion_tokens']['total'])
        self.assertEqual(result['audit']['cost']['completion_tokens']['unknown_calls'], 1)
        self.assertEqual(result['audit']['cost']['prompt_tokens']['total'], 20)

    def test_doubled_quote_json_transport_is_normalized_without_model_retry(self):
        raw = ('{"skip":false,"answer":"Sandra is in the kitchen.",'
               '"quotes":[""Sandra moved to the kitchen.""],"why_useful":"location"}')
        fake = FakeCalls(({'response_count': 1, 'raw_output': raw, 'prompt_tokens': 9,
                           'completion_tokens': 7},), {'questions': ['Where did Sandra move?']})
        result = build([cluster()], fake)
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(len(result['examples']), 1)
        self.assertEqual(result['audit']['calls'][0]['parse_status'],
                         'parsed_after_doubled_quote_normalization')
        self.assertEqual(result['examples'][0]['answer_source_quotes'][0]['quote'],
                         'Sandra moved to the kitchen.')

    def test_quote_wrapper_with_one_trailing_comma_keeps_exact_provenance(self):
        text = 'John picked up the football there.'
        checked = literal_quotes(['"John picked up the football there.",'], [
            {'source_index': 9, 'span': [100, 100+len(text)], 'text': text}])
        self.assertEqual(checked[0]['quote'], text)
        self.assertEqual(checked[0]['reported_quote'], '"John picked up the football there.",')

    def test_one_unmatched_trailing_brace_is_normalized_without_retry(self):
        value, status = parse_object_with_status('{"clusters":[{"source_ids":["1"]}]}}')
        self.assertEqual(value, {'clusters': [{'source_ids': ['1']}]})
        self.assertEqual(status, 'parsed_after_trailing_brace_normalization')

    def test_json_control_character_is_parsed_without_retry_and_raw_rewrite(self):
        value, status = parse_object_with_status('{"answer":"park/\tcinema"}')
        self.assertEqual(value, {'answer': 'park/\tcinema'})
        self.assertEqual(status, 'parsed_with_json_control_characters')

    def test_ambiguous_extra_responses_never_silently_pick_first(self):
        fake = FakeCalls(({'response_count': 2, 'raw_output': json.dumps(answer()),
                          'unexpected_outputs': ['one', 'two']},))
        result = build([cluster()], fake)
        self.assertFalse(result['examples'])
        self.assertEqual(result['audit']['calls'][0]['unexpected_outputs'], ['one', 'two'])

    def test_empty_source_and_invalid_ids_do_not_trigger_inference(self):
        fake = FakeCalls()
        result = build([cluster(text=' ')], fake)
        self.assertFalse(fake.calls)
        self.assertEqual(result['audit']['rejections'][0]['reason'], 'empty_evidence')
        with self.assertRaisesRegex(ValueError, 'unique'): build([cluster(), cluster()], fake)
        self.assertFalse(fake.calls)

    def test_lexical_leak_guard_is_word_based_and_does_not_reject_boolean_format(self):
        self.assertIsNone(direct_answer_leakage('Where is Annabelle?', 'Ann'))
        self.assertIsNone(direct_answer_leakage('Is Sandra home? Answer yes or no.', 'yes'))
        self.assertEqual(direct_answer_leakage('Is the deadline 7 days?', '7 days'), 'literal_answer_phrase_in_question')

    def test_default_transport_requests_no_soft_targets_and_preserves_missing_response(self):
        runtimes, requests = {}, []
        class Client:
            Config = staticmethod(lambda **kwargs: NS(**kwargs))
            def __init__(self, config): self.engine = runtimes[config.runtime_key]
            async def chat(self, chats, **kwargs):
                requests.append((chats, kwargs))
                return NS(samples=[], usage=NS(prompt_tokens=12, completion_tokens=0))
        fake_upstream = NS(LocalStudyClient=Client, _RUNTIMES=runtimes)
        engine = NS(tokenizer=NS(name_or_path='fake'))
        with patch.dict(sys.modules, {'contextualized.upstream_study': fake_upstream}):
            result = asyncio.run(frozen_text_call(engine, 'Extract', {'query': 'Where?'}, max_tokens=32))
        self.assertIsNone(requests[0][1]['top_logprobs'])
        self.assertEqual(requests[0][1]['temperature'], 0.)
        self.assertEqual(len(requests[0][0]), 1)
        self.assertEqual(result['response_count'], 0)
        self.assertEqual(result['prompt_tokens'], 12)
        self.assertFalse(runtimes)


if __name__ == '__main__': unittest.main()
