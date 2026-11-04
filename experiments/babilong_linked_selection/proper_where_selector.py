"""Consume cached Decision scores and use the fixed question-overlap policy."""
from masked_ttt.frozen_scores import FrozenJSONSelector, load_scores
from masked_ttt.selection import PreparedRequest
from experiments.babilong_linked_selection.question_overlap_selector import question_overlap


class ProperWhereSelector(FrozenJSONSelector):
    def prepare(self, backbone, original):
        prepared = super().prepare(backbone, original)
        data = load_scores(self.directory, original.document, original.question, self.case_key, self.by_case)
        mapping = prepared.metadata["source_mapping"]
        records = (data["records"] if mapping is None else
                   [dict(data["records"][item["original_chunk"]], chunk=i)
                    for i, item in enumerate(mapping)])
        selected = question_overlap(prepared.request.document, prepared.request.question,
                                    prepared.layout.spans, records, backbone.config.keep_fraction,
                                    mode="proper_where")
        metadata = dict(prepared.metadata)
        metadata.update(selected_chunks=selected,
            selected_original_chunks=[records[i].get("original_chunks",
                [mapping[i]["original_chunk"]] if mapping is not None else [i]) for i in selected],
            selected_tokens=sum(len(prepared.layout.chunks[i]) for i in selected),
            selection_policy="decision_proper_where_v1", entity_reserve_fraction=.25,
            question_overlap_policy="proper_where", reference_answers_used=False, online_api_calls=0)
        return PreparedRequest(prepared.request, prepared.layout, selected, metadata)
