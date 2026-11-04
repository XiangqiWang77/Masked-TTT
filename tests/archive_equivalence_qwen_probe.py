"""Emit a deterministic Qwen3 full-student/selected-cartridge trace."""
import json

from masked_ttt import MaskedTTT,Request
from masked_ttt.request import prepare_layout
from masked_ttt.selection import PreparedRequest
from test_standalone import make_engine


def main():
    engine=make_engine()
    request=Request(document='Old note says red. Current note says blue. '*6,
                    question='What is in the current note?',max_new_tokens=3)
    class Selector:
        def prepare(self,backbone,item):
            layout=prepare_layout(backbone,item)
            return PreparedRequest(item,layout,[1,len(layout.chunks)-2],dict(backend='unit_test'))
    result=MaskedTTT(engine,Selector()).answer(request)
    full=result.get('full_context',{}).get('prediction',result['prediction'])
    print(json.dumps(dict(prediction=full,standalone=result['standalone_cartridge']['prediction'],
        teacher=result['teacher']['prediction'],steps=result['training']['steps'],
        frozen=result['backbone_unchanged'],cache_unchanged=result['original_document_KV_unchanged']),
        sort_keys=True))


if __name__=='__main__':main()
