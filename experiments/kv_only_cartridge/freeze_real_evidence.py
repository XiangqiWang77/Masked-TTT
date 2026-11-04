#!/usr/bin/env python3
"""Summarize saved real-model cartridge preflights without new model calls."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


RELEASE_CHECKS=('same_teacher_tokens','same_standalone_tokens','same_standalone_logits',
                'long_attention_gradients_match','source_and_training_released')
REQUEST_CHECKS=('same_teacher_tokens','same_student_rollouts','same_training_gradients',
    'same_standalone_tokens','same_standalone_logits','same_request_chat_date',
    'no_full_kv_or_extra_adapter_at_decode','original_benchmark_runtime_unchanged',
    'same_generic_tokens_before_after_release')


def release(data,expected_model):
    gpu=data.get('gpu')
    if (data.get('model')!=expected_model or data.get('passed') is not True or
        not isinstance(gpu,str) or not gpu or 'cpu' in gpu.lower() or
        any(data.get(key) is not True for key in RELEASE_CHECKS) or
        data.get('long_attention_backend')!='flash' or data.get('long_attention_positions')!=32 or
        data.get('evaluated_positions',0)<1 or data.get('adapter_bytes_after')!=0 or
        data.get('online_api_calls')!=0 or data.get('reference_answers_used') is not False):
        raise ValueError('The real-model KV-only release evidence is incomplete')
    return dict(model=expected_model,gpu=gpu,passed=True,generated_positions=data['evaluated_positions'],
        attention_positions=32,flash_attention_gradients_match=True,standalone_tokens_and_logits_match=True,
        source_and_training_released=True,adapter_bytes_after=0)


def request_local(data):
    gpu=data.get('identity',{}).get('gpu')
    if (data.get('model')!='qwen' or data.get('model_type')!='qwen3' or data.get('passed') is not True or
        not isinstance(gpu,str) or not gpu or 'cpu' in gpu.lower() or
        any(data.get(key) is not True for key in REQUEST_CHECKS) or
        data.get('generic_teacher_answer_in_prompt') is not False or
        data.get('online_api_calls')!=0 or data.get('reference_answers_used') is not False):
        raise ValueError('The Qwen request-local evidence is incomplete')
    return dict(model='Qwen/Qwen3-4B-Instruct-2507',gpu=gpu,passed=True,
        teacher_tokens_and_student_rollouts_match=True,training_gradients_match=True,
        standalone_tokens_and_logits_match=True,generic_format_tokens_match_after_release=True,
        teacher_answer_in_prompt=False,source_and_training_released=True)


def freeze(qwen_release,gpt_release,qwen_request):
    return dict(created_utc=datetime.now(timezone.utc).isoformat(),
        scope='synthetic real-model deployment checks; not BABILONG accuracy',
        online_api_calls=0,reference_answers_used=False,new_model_calls=0,
        kv_only_release={
            'qwen':release(qwen_release,'Qwen/Qwen3-4B-Instruct-2507'),
            'gpt':release(gpt_release,'openai/gpt-oss-20b')},
        request_local={'qwen':request_local(qwen_request),
                       'gpt':dict(state='pending_real_gpu_check')})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('qwen-release','gpt-release','qwen-request','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    data=freeze(*(json.loads(path.read_text()) for path in
                  (args.qwen_release,args.gpt_release,args.qwen_request)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(data['kv_only_release'],ensure_ascii=False))


if __name__=='__main__':main()
