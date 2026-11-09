"""Freeze source-Teacher/grounded-veto SD with source-disjoint development and holdout."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
import os
from pathlib import Path
import shutil

from experiments.run_syn_sd_factorial import read,write


def prepare(previous_config,output):
    previous=read(previous_config);config=deepcopy(previous);output=Path(output)
    plan=read(previous['source_plan']);metadata={item['key']:item for item in plan['cases']}
    development=[];holdout=[]
    for old in previous['cases']:
        item=deepcopy(old)
        item['analysis_split']='holdout' if metadata[item['key']]['split']=='source_disjoint_holdout' else 'development'
        (holdout if item['analysis_split']=='holdout' else development).append(item)
    if len(development)!=160 or len(holdout)!=840:
        raise ValueError('Retain the existing source-family-disjoint 160/840 split')
    config.update(protocol='native-Cartridges-source-teacher-question-guided-grounded-veto-v3',
        output=str(output),cases=development+holdout,
        paired_input_dir=previous['paired_input_dir'],
        teacher_context_policy='question_source_evidence',
        teacher_context='original_2048_token_source_for_vanilla; bound_evidence_for_question_guided',
        teacher_sampling_temperature=0.,
        selective_gate='verified_quote_grounded_veto',
        selective_critic_context='full_document_with_focused_source_quotes',
        empty_selection_policy='original_SD',
        no_selected_signal_policy='train_with_original_teacher_SD; never_zero_updates',
        matched_update_reference_arm='vanilla_syns_vanilla_sd',
        study_max_model_len=16384,
        phase_control_path=str(output/'phase.json'),
        development_cases=160,holdout_cases=840,
        development_source_questions=40,holdout_source_questions=210,
        protocol_frozen_utc=datetime.now(timezone.utc).isoformat(),
        optimization_policy='develop_on_160_cases; enable_840_case_holdout_only_after_frozen_development_decision',
        deprecated_comparison_root=previous['output'],
        new_synthetic_generation=False,teacher_targets_reused_from_full_document_protocol=False,
        old_trained_cartridges_reused=False,
        question_budget_policy='unchanged_saved_1000_case_synthetic_sets; matched_per_case_split_counts_and_actual_SD_updates',
        validation_policy='same_heldout_source_Teacher_targets_within_each_SD_pair; no_benchmark_answer_supervision')
    previous_root=Path(previous['output']);reused=Counter()
    for item in config['cases']:
        source=previous_root/'cases'/item['key'];target=output/'cases'/item['key']
        initial=source/'initial.safetensors'
        if initial.is_file():
            destination=target/'initial.safetensors';destination.parent.mkdir(parents=True,exist_ok=True)
            if not destination.exists():
                try:os.link(initial,destination)
                except OSError:shutil.copy2(initial,destination)
            reused['initial_caches']+=1
        baseline=source/'baselines.json'
        if baseline.is_file():
            values=read(baseline);values.pop('initial_kv',None)
            # The full-context/no-context inference prompts are unchanged.
            write(target/'baselines.json',values);reused['baseline_predictions']+=1
    write(output/'phase.json',{'phase':'development','holdout_enabled':False,
        'reason':'repair source-Teacher fidelity and original grounded-veto semantics before heldout evaluation'})
    write(output/'protocol_audit.json',{'previous_protocol':previous['protocol'],
        'corrections':[
            'restore the actual generation source E rather than silently replacing it by the full document',
            'restore official bot-B greedy temperature 0 instead of temperature 0.6',
            'retain original SD under unresolved checks; use source-grounded veto rather than development strict support-required gating',
            'match actual optimizer updates across all four arms, using generic vanilla-SD stopping as the common budget',
            'preserve official truncated Teacher continuations; never abstain or return untrained memory for lack of verifier certainty',
            'use scratch compilation caches; the unchanged old full-document veto critic sees additional context beyond the local Teacher'],
        'native_KV_only':True,'backbone_frozen':True,'student_reads_document_at_inference':False,
        'hard_label_NLL':False,'attention_modified':False,'B_g_LoRA_added':False,
        'synthetic_data_regenerated':False,'old_results_deleted':False,
        'development_cases':160,'holdout_cases':840,'reused':dict(reused),
        'original_deadline_unchanged':config['deadline_utc']})
    write(output/'config.json',config)
    return config,reused


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous-config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();config,reused=prepare(a.previous_config,a.output)
    print({'config':str(Path(a.output)/'config.json'),'development':160,'holdout':840,
           'teacher_context_policy':config['teacher_context_policy'],'selective_gate':config['selective_gate'],
           'reused':dict(reused),'deadline_utc':config['deadline_utc']})


if __name__=='__main__':main()
