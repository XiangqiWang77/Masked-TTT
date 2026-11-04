"""Apply the fixed entity-link policy to an existing frozen Decisions cache."""
from masked_ttt.frozen_scores import FrozenJSONSelector,load_scores
from masked_ttt.selection import PreparedRequest

from .selector import choose


class LinkedFrozenSelector(FrozenJSONSelector):
    def prepare(self,backbone,original):
        prepared=super().prepare(backbone,original)
        data=load_scores(self.directory,original.document,original.question,
                         self.case_key,self.by_case)
        mapping=prepared.metadata['source_mapping']
        if mapping is None:
            records=data['records']
        else:
            records=[dict(data['records'][item['original_chunk']],chunk=i)
                     for i,item in enumerate(mapping)]
        selected=choose(prepared.request.document,prepared.request.question,
            prepared.layout.spans,records,backbone.config.keep_fraction,'entity',.25)
        metadata=dict(prepared.metadata)
        metadata.update(selected_chunks=selected,
            selected_original_chunks=[records[i].get('original_chunks',
                [mapping[i]['original_chunk']] if mapping is not None else [i]) for i in selected],
            selected_tokens=sum(len(prepared.layout.chunks[i]) for i in selected),
            selection_policy='decision_entity_link_v1',entity_reserve_fraction=.25,
            reference_answers_used=False,online_api_calls=0)
        return PreparedRequest(prepared.request,prepared.layout,selected,metadata)
