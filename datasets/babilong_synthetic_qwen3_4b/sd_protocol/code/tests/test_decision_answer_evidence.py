from experiments.decision_answer_evidence import prioritize_decision_clusters


def chunk(index, *, keep=False, lexical=0.0, text=None):
    return {
        'id': str(index), 'source_index': index,
        'text': text or f'chunk {index}', 'span': [10 * index, 10 * index + 5],
        'decision_api_accepted': keep,
        'decision_api_probability_yes': 0.9 if keep else 0.1,
        'lexical_score': lexical,
    }


def test_unanchored_decision_drops_do_not_become_exercises():
    relevant = chunk(0, keep=True, text='Mary gave the apple to Fred.')
    bridge = chunk(1, text='Mary entered the room.')
    unrelated = chunk(2, text='Peter whispered to Sam.')
    clusters = [
        {'origin': 'bridge', 'source_chunks': [relevant, bridge]},
        {'origin': 'generic', 'source_chunks': [unrelated]},
    ]

    selected, _, rejected = prioritize_decision_clusters(
        'Who gave the apple to Fred?', [relevant, bridge, unrelated], clusters,
        max_clusters=8, max_chunks=4)

    identities = {tuple(row['source_index'] for row in group['source_chunks'])
                  for group in selected}
    assert (0, 1) in identities
    assert (2,) not in identities
    assert rejected == 1


def test_positive_lexical_anchor_survives_without_api_keep():
    anchor = chunk(0, lexical=1.0, text='The milk was in the garden.')
    complement = chunk(1, text='Sandra later returned to the kitchen.')
    selected, _, rejected = prioritize_decision_clusters(
        'Where was the milk before the kitchen?', [anchor, complement],
        [{'origin': 'lexical_bridge', 'source_chunks': [anchor, complement]}],
        max_clusters=8, max_chunks=4)
    assert any([c['source_index'] for c in row['source_chunks']] == [0, 1]
               for row in selected)
    assert rejected == 0


def test_every_local_focus_retains_query_complete_cross_chunk_core():
    early = chunk(1, keep=True, lexical=.4,
                  text='Mary is either in the school or the bedroom.')
    late = chunk(28, keep=True, lexical=.8, text='Mary is in the park.')
    bridge = chunk(10, lexical=0, text='Bill travelled to the bedroom.')
    selected, _, _ = prioritize_decision_clusters(
        'Is Mary in the park?', [early, bridge, late],
        [
            {'origin': 'joint', 'source_chunks': [early, late]},
            {'origin': 'old_singleton', 'source_chunks': [early]},
            {'origin': 'bridge', 'source_chunks': [early, bridge]},
        ], max_clusters=8, max_chunks=4)
    identities = [tuple(row['source_index'] for row in group['source_chunks'])
                  for group in selected]
    assert identities
    assert all(1 in identity and 28 in identity for identity in identities)
    assert (1,) not in identities
    assert len(identities) == len(set(identities))
    assert all(group['query_complete_core_source_indices'] == [1, 28]
               for group in selected)
