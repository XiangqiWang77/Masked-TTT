"""Run only Contextual Cartridges for a Figure-4-style cache/quality study.

The twelve cells are the Cartesian product of LongHealth, MTOB (KE),
QASPER and four native-KV prefix sizes. Paper baselines are deliberately not
executed here.  Every request uses query-guided answer-first question crafting,
ordinary sparse-logit self-distillation, and conservative TracIn nominations
confirmed by actual matched refits. Benchmark answers are excluded from learning.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time
import traceback


DATASETS = ("longhealth", "mtob_ke", "qasper")
PREFIXES = (512, 1024, 2048, 4096)
PROTOCOL = 'contextual-figure4-confirmed-harm-tracin-v1'


def training_config(model, prefix_tokens):
    return {
        'protocol': PROTOCOL, 'model': model, 'prefix_tokens': prefix_tokens,
        'packed_tokens': 512, 'learning_rate': .02, 'train_batch_size': 'full_dataset',
        'official_epochs': 2, 'min_steps': 1, 'max_steps': 2,
        'eval_every': 1, 'patience': 3, 'track_best_before_min': True,
        'tracin_threshold': 0., 'no_heldout_all_arms': True,
        'matched_update_reference_arm': 'specific_syns_vanilla_sd',
        'matched_update_unit': 'packed_optimizer_steps',
        'selective_rule': 'unanimous_negative_TracIn_and_confirmed_deletion_benefit',
        'matched_training_orders': 2,
        'deletion_loss_atol': 1e-6, 'deletion_loss_rtol': 1e-5,
        'keep_uncertain_trajectories': True,
        'sd_target_only_logits': True,
        'sd_offload_saved_tensors': True,
    }


def complete_target_layout(engine, conversations, config):
    """Fit the complete training conversations independently of KV capacity.

    All cache sizes for the same case share the same Teacher trajectories and
    the same resulting packing length. Upstream pad packing never cuts the
    final conversation in a batch. Neither answers nor targets are regenerated.
    """
    from cartridges.datasets import qwen_messages_to_element
    converter = getattr(engine, 'message_converter', qwen_messages_to_element)
    elements = [converter(row.messages, tokenizer=engine.tokenizer)
                for row in conversations]
    if not elements or any(element.topk_token_idxs.numel() == 0 for element in elements):
        raise ValueError('Every Teacher conversation must supply soft targets')
    longest = max(len(element.input_ids) for element in elements)
    packed = max(int(config['packed_tokens']), 1 << (longest - 1).bit_length())
    if packed > 4096:
        raise ValueError('A complete conversation exceeds the 4096-token training budget')
    result = dict(config, packed_tokens=packed, packing_mode='pad')
    audit = {
        'packing_mode': 'pad', 'base_training_tokens': config['packed_tokens'],
        'packed_training_tokens': packed, 'longest_conversation_tokens': longest,
        'teacher_target_terms': sum(e.topk_token_idxs.numel() for e in elements),
        'teacher_conversations': len(elements), 'truncated_target_terms': 0,
        'independent_of_cartridge_capacity': True,
    }
    return result, audit


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def stable_key(*parts):
    return hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()[:20]


def longhealth_prompt(patient_id, patient, question):
    options = "\n".join(question[f"answer_{letter}"] for letter in "abcde")
    patient_info = (f"ID {patient_id}, Name: {patient['name']}, Birthday: "
                    f"{patient['birthday']}, Diagnosis: {patient['diagnosis']}")
    return (
        "Please answer the question below about the following patient: " + patient_info
        + f"\n\n<question>\n{question['question']}\n</question>"
        + f"\n\n<options>\n{options}\n</options>\n"
        + "Please provide your answer exactly as it appears in the options. "
          "Output nothing except <answer>{YOUR_ANSWER}</answer>."
    )


def mtob_prompt(sentence):
    return (
        'You are tasked with translating the following sentence from Kalamang to English: '
        f'"{sentence}".\nI understand that you may not be familiar enough with Kalamang '
        "or English to make a confident translation, but please give your best guess.\n"
        "Make sure you respond with only the translation and absolutely no other text. "
        "Do not repeat the input sentence."
    )


def qasper_prompt(question):
    return (
        "Please write a succinct answer to the following question.\n"
        "You do not need to restate the paper name or answer in complete sentences.\n\n"
        f"<question>\n{question}\n</question>\n\n"
        "Provide your answer in the following format (output nothing else):\n\n"
        "<answer>\n{your answer here}\n</answer>"
    )


def load_longhealth(root):
    data = read(Path(root) / "longhealth" / "benchmark_v5.json")
    cases = []
    for patient_id in [f"patient_{index:02d}" for index in range(1, 11)]:
        patient = data[patient_id]
        document = "\n\n".join(patient["texts"][key] for key in sorted(patient["texts"]))
        for question in patient["questions"]:
            cases.append({
                "id": f"{patient_id}-q{int(question['No']):02d}",
                "document_id": patient_id,
                "document": document,
                "query": question["question"],
                "prompt": longhealth_prompt(patient_id, patient, question),
                "reference": question["correct"],
                "options": [question[f"answer_{letter}"] for letter in "abcde"],
            })
    return cases


def load_mtob(root):
    base = Path(root) / "mtob" / "extracted"
    document = (base / "resources" / "grammar_book.tex").read_text()
    rows = read(base / "splits" / "test_examples_ke.json")
    cases = []
    for index, row in enumerate(rows):
        if "original" not in row or "ground_truth" not in row:
            continue
        cases.append({
            "id": f"mtob-ke-{index:03d}", "document_id": "kalamang-grammar-book",
            "document": document, "query": row["original"],
            "prompt": mtob_prompt(row["original"]), "reference": row["ground_truth"],
        })
    return cases


def qasper_documents(root):
    path = Path(root) / "qasper" / "raw" / "qasper-train-v0.3.json"
    rows = read(path)
    documents = {}
    for paper_id, paper in rows.items():
        sections = []
        for section in paper["full_text"]:
            title = section.get("section_name", "")
            paragraphs = section.get("paragraphs", [])
            sections.append(f"<section-title>{title}</section-title>\n" + "\n\n".join(paragraphs))
        documents[paper_id] = (
            f"<title>{paper['title']}</title>\n<abstract>{paper['abstract']}</abstract>\n"
            + "\n\n".join(sections)
        )
    return documents


def load_qasper(root):
    import pandas as pd
    base = Path(root) / "qasper"
    frame = pd.read_parquet(base / "qasper-rewrite-gpt-4.1" / "data" /
                            "question-00000-of-00001.parquet")
    documents = qasper_documents(root)
    cases = []
    for index, row in frame.iterrows():
        paper_id = str(row["paper_id"])
        if paper_id not in documents:
            raise ValueError(f"QASPER paper {paper_id} is absent from the official train split")
        cases.append({
            "id": f"qasper-{index:03d}-{paper_id}", "document_id": paper_id,
            "document": documents[paper_id], "query": str(row["question"]),
            "prompt": qasper_prompt(str(row["question"])), "reference": str(row["answer"]),
        })
    return cases


def load_cases(dataset, root):
    return {"longhealth": load_longhealth, "mtob_ke": load_mtob,
            "qasper": load_qasper}[dataset](root)


def chunk_document(tokenizer, document, chunk_tokens=640):
    encoded = tokenizer(document, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    chunks = []
    for start in range(0, len(ids), chunk_tokens):
        end = min(start + chunk_tokens, len(ids))
        begin, finish = offsets[start][0], offsets[end - 1][1]
        text = document[begin:finish]
        if text.strip():
            chunks.append({"id": str(len(chunks)), "source_index": len(chunks),
                           "span": [begin, finish], "text": text})
    return chunks


def lexical_candidates(query, chunks, maximum=24):
    stop = {"the", "a", "an", "of", "to", "in", "is", "are", "was", "were",
            "what", "which", "who", "where", "when", "how", "and", "or", "for"}
    words = [word for word in re.findall(r"[^\W_]+", query.casefold())
             if len(word) > 1 and word not in stop]
    counts = Counter(words)
    ranked = []
    for index, chunk in enumerate(chunks):
        text = chunk["text"].casefold()
        score = sum((2.0 if " " in word else 1.0) * weight * text.count(word)
                    for word, weight in counts.items())
        exact = 1 if query.strip().casefold() in text else 0
        ranked.append((exact, score, -index, index))
    ranked_indices = [row[-1] for row in sorted(ranked, reverse=True)]
    selected = ranked_indices[:maximum]
    # Preserve a small global scaffold so retrieval is not purely lexical.
    if len(chunks) > maximum:
        scaffold = [0, len(chunks) // 3, 2 * len(chunks) // 3, len(chunks) - 1]
        selected = ranked_indices[:max(0, maximum - len(scaffold))]
        for index in scaffold + ranked_indices:
            if len(selected) == maximum:
                break
            if index not in selected:
                selected.append(index)
    # Temporal/dependency order is the original document order, not rank order.
    return [chunks[index] for index in sorted(selected)]


def prepare_evidence(reranker, query, chunks):
    from experiments.question_guided_candidates import craft_evidence
    candidates = lexical_candidates(query, chunks)
    passages = [row["text"] for row in candidates]
    spans = [row["span"] for row in candidates]
    policy = {"candidate_pool": 8, "pair_candidates": 8, "max_evidence": 12}
    content = craft_evidence(query, passages, spans, reranker.score, policy,
                             score_pairs=lambda q, p: reranker.score(
                                 q, p, instruction=reranker.model_info["pair_instruction"]))
    groups = content["selected_evidence"]  # already interleaves pairs/singletons
    return groups[:4], {"candidate_chunks": len(candidates),
                       "document_chunks": len(chunks), "reranker": content}


async def craft_questions(engine, query, evidence):
    from contextualized.answer_guided_questions import build_answer_guided_dataset
    return await build_answer_guided_dataset(
        engine, query, evidence, max_examples=12, answer_tokens=384,
        question_tokens=384, questions_per_answer=3, simple_two_stage=True)


def materialize_conversations(engine, dataset_record):
    from experiments.contextual_learning_runtime import _query_conversation
    from contextualized.tracin_sd import structural_source_gate
    conversations = []
    for index, row in enumerate(dataset_record["examples"]):
        generated = engine.generate_teacher(row["question"], row["evidence"]["text"],
                                            max_new_tokens=192, sampling_temperature=0,
                                            topk=20)
        if not generated.answer_ids:
            continue
        conversation = _query_conversation(engine, row["question"],
                                           row["evidence"]["text"], generated)
        conversation.metadata.update(question_index=index, answer_guided=True,
                                     benchmark_answer_used=False)
        conversation.metadata['source_gate'] = structural_source_gate(
            question=row['question'], evidence=row['evidence']['text'],
            candidate=engine.text(generated), candidate_complete=generated.finished)
        conversations.append(conversation)
    if not conversations:
        raise ValueError('No Teacher trajectory: do not report an untrained Cartridge as completed')
    return conversations


def train_selective(engine, initial, conversations, config, seed, directory):
    from experiments.conservative_tracin_runtime import train_conservatively
    return train_conservatively(engine, initial, conversations, config, seed, directory,
        deadline=float(os.environ.get('FIGURE4_DEADLINE_EPOCH', 'inf')))


def normalize_answer(text):
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.I | re.S)
    return (match.group(1) if match else text).strip()


def longhealth_score(prediction, case):
    from difflib import SequenceMatcher
    # Match the released Cartridges scorer, including its first-option
    # fallback when the required answer tags are absent.
    match = re.search(r'<answer>(.*?)</answer>', prediction, flags=re.S)
    if not match:
        return float(case['options'][0].strip().lower() == case['reference'].strip().lower())
    value = match.group(1).strip().lower()
    closest = max(case["options"], key=lambda item: SequenceMatcher(
        None, value, item.strip().lower()).ratio())
    return float(closest.strip().lower() == case["reference"].strip().lower())


def chr_f(prediction, reference, beta=2.0, max_order=6):
    """Sentence chrF compatible in spirit with the official corpus metric."""
    prediction = normalize_answer(prediction)
    scores = []
    for order in range(1, max_order + 1):
        pred = Counter(prediction[i:i + order] for i in range(max(0, len(prediction) - order + 1)))
        ref = Counter(reference[i:i + order] for i in range(max(0, len(reference) - order + 1)))
        overlap = sum((pred & ref).values())
        precision = overlap / max(1, sum(pred.values()))
        recall = overlap / max(1, sum(ref.values()))
        denom = beta * beta * precision + recall
        scores.append((1 + beta * beta) * precision * recall / denom if denom else 0.0)
    return 100.0 * sum(scores) / len(scores)


def qasper_nll(engine, cache, prompt, reference):
    import torch
    prompt_ids = engine._chat_ids(prompt)
    answer_ids = engine.tokenizer.encode(f"<answer>\n{reference}\n</answer>",
                                         add_special_tokens=False)
    raw = prompt_ids + answer_ids
    ids = torch.tensor(raw, device=engine.device)
    positions = torch.arange(len(prompt_ids) - 1,
                             len(prompt_ids) - 1 + len(answer_ids), device=engine.device)
    cache.clear()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = engine.student(input_ids=ids, seq_ids=torch.zeros_like(ids),
            position_ids=torch.arange(len(ids), device=engine.device),
            past_key_values=cache, use_cache=True, mode="train", logits_to_keep=positions)
        target = torch.tensor(answer_ids, device=engine.device)
        loss = torch.nn.functional.cross_entropy(output.logits[0].float(), target)
    cache.clear()
    return float(loss)


def cache_gb(engine, prefix_tokens):
    config = engine.student.config
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    return (2 * config.num_hidden_layers * config.num_key_value_heads * head_dim
            * prefix_tokens * 2) / 1e9


def evaluate_frozen_cartridge(engine, cartridge, dataset, case):
    """Post-training metrics only; no result from here can affect selection."""
    if dataset == 'qasper':
        return None, qasper_nll(engine, cartridge, case['prompt'], case['reference'])
    prediction = engine.generate_student(case['prompt'], cartridge, max_new_tokens=192)[0]
    if dataset == 'mtob_ke':
        prediction = normalize_answer(prediction)
    score = (longhealth_score(prediction, case) if dataset == 'longhealth'
             else chr_f(prediction, case['reference']))
    return prediction, score


def run(args):
    import gc
    import torch
    from cartridges.structs import read_conversations, write_conversations
    from contextualized.cache_io import load_cartridge, save_cartridge
    from experiments.syn_sd_factorial_runtime import load_engine, resident
    from experiments.question_guided_candidates import FrozenReranker
    from experiments.qwen_reranker_candidates import INSTRUCTION, PAIR_INSTRUCTION

    dataset = DATASETS[args.task // len(PREFIXES)]
    prefix_tokens = PREFIXES[args.task % len(PREFIXES)]
    output = Path(args.output)
    task_root = output / dataset / f"p{prefix_tokens}"
    task_root.mkdir(parents=True, exist_ok=True)
    deadline = float(os.environ.get("FIGURE4_DEADLINE_EPOCH", time.time() + 8 * 3600))
    all_cases = load_cases(dataset, args.data_root)
    shard, shards = args.shard, args.shards
    if not 0 <= shard < shards <= 6:
        raise ValueError('At most six additional GPU workers are allowed')
    assigned = list(enumerate(all_cases))[shard::shards]
    if args.case_index is not None:
        if not 0 <= args.case_index < len(all_cases):
            raise ValueError('Requested case index is outside the dataset')
        assigned = [(position, case) for position, case in assigned
                    if position == args.case_index]
        if not assigned:
            raise ValueError('Requested case index does not belong to this shard')
    pending = [(position, case) for position, case in assigned
               if not (task_root / 'cases' / f"{case['id']}.json").exists()]
    def save_status():
        count = sum((task_root / 'cases' / f"{case['id']}.json").exists()
                    for case in all_cases)
        write(task_root / 'worker.json', {'dataset': dataset,
              'prefix_tokens': prefix_tokens, 'planned': len(all_cases),
              'completed': count, 'complete': count == len(all_cases),
              'deadline_epoch': deadline, 'task': args.task})
    if not pending:
        save_status()
        return 0
    if time.time() >= deadline - 180:
        save_status()
        return 75
    model = args.model
    config = training_config(model, prefix_tokens)
    # Only one 4B backbone is GPU-resident at a time. The 0.6B reranker
    # replaces the unused verifier; no extra answer-judging model is loaded.
    engine = load_engine(config)
    model_info = {"model": args.reranker_model, "instruction": INSTRUCTION,
                  "pair_instruction": PAIR_INSTRUCTION, "max_length": 4096}
    reranker = FrozenReranker(model_info, batch_size=2).__enter__().to('cpu')
    document_chunks = {}
    completed = []
    try:
        for position, case in pending:
            destination = task_root / "cases" / f"{case['id']}.json"
            if destination.exists():
                completed.append(case["id"])
                continue
            if time.time() >= deadline - 180:
                break
            started = time.monotonic()
            shared = output / "shared_synthetic" / dataset / f"{case['id']}.json"
            lock_path = shared.with_suffix(".lock")
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open("a+") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                if shared.exists():
                    synthetic = read(shared)
                else:
                    if case['document_id'] not in document_chunks:
                        document_chunks[case['document_id']] = chunk_document(
                            engine.tokenizer, case['document'])
                    chunks = document_chunks[case['document_id']]
                    engine.teacher.to('cpu'); engine.student.to('cpu')
                    gc.collect(); torch.cuda.empty_cache()
                    reranker.to(engine.device)
                    try:
                        evidence, retrieval = prepare_evidence(reranker, case["query"], chunks)
                    finally:
                        reranker.to('cpu')
                        gc.collect(); torch.cuda.empty_cache()
                    resident(engine, 'teacher')
                    learning_query = case['prompt'] if dataset == 'mtob_ke' else case['query']
                    crafted = asyncio.run(craft_questions(engine, learning_query, evidence))
                    examples = crafted['examples']
                    fallback = not examples
                    if fallback:
                        # Still perform ordinary source-conditioned SD when
                        # answer-first crafting produces no usable question.
                        examples = [{'question': case['prompt'], 'evidence': group,
                                     'origin': 'original_query_teacher_fallback'}
                                    for group in evidence]
                    if not examples:
                        raise ValueError('No source material available for a trained fallback')
                    synthetic = {
                        "case": case["id"], "dataset": dataset, "query": case["query"],
                        "examples": examples, "question_audit": crafted["audit"],
                        'original_query_teacher_fallback': fallback, 'protocol': PROTOCOL,
                        "retrieval": retrieval, "benchmark_answer_used": False,
                    }
                    write(shared, synthetic)
                targets_path = shared.with_suffix('.targets.pkl')
                if targets_path.exists():
                    conversations = read_conversations(targets_path)
                else:
                    resident(engine, 'teacher')
                    conversations = materialize_conversations(engine, synthetic)
                    temporary = targets_path.with_name(targets_path.stem + f'.{os.getpid()}.pkl')
                    write_conversations(conversations, temporary)
                    temporary.replace(targets_path)
                if not conversations:
                    raise ValueError('Cached Teacher targets are empty')
            case_config, packing_audit = complete_target_layout(engine, conversations, config)
            case_config['teacher_targets_sha256'] = hashlib.sha256(targets_path.read_bytes()).hexdigest()
            engine.dataset_packing_mode = case_config['packing_mode']
            engine.packed_tokens = case_config['packed_tokens']
            engine.sd_target_only_logits = case_config['sd_target_only_logits']
            engine.sd_offload_saved_tensors = case_config['sd_offload_saved_tensors']
            resident(engine, 'student')
            torch.cuda.reset_peak_memory_stats()
            seed = args.seed + position
            random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            initial = engine.initial_cache(case["document"])
            # Isolate incomplete old packing checkpoints; completed results are
            # still skipped above and their original protocol stays recorded.
            case_dir = task_root / "work" / case["id"] / PROTOCOL
            case_dir.mkdir(parents=True, exist_ok=True)
            cartridge, training = train_selective(engine, initial, conversations, case_config,
                                                  seed, case_dir)
            if cartridge.num_cartridge_tokens() != initial.num_cartridge_tokens():
                raise ValueError('Returned Cartridge has a different KV capacity from initialization')
            returned_training = (training['vanilla_training'] if training['fallback']
                                 else training['selective_training'])
            if returned_training.get('updates', 0) < 1:
                raise RuntimeError('A completed experiment must return a trained Cartridge')
            with torch.no_grad():
                delta_sq = sum(float((parameter.detach().float() - base.detach().float())
                                     .square().sum())
                               for parameter, base in zip(cartridge.parameters(), initial.parameters())
                               if parameter.requires_grad)
            prediction, metric_value = evaluate_frozen_cartridge(engine, cartridge, dataset, case)
            save_cartridge(cartridge, case_dir / "contextual.safetensors", model)
            pilot_prediction, pilot_metric = prediction, metric_value
            if training['removed']:
                # Evaluate the pilot already required by selection. This is
                # solely a post-hoc regression audit, never a gold-based veto
                # or a separately rerun paper baseline.
                cartridge.clear(); initial.clear()
                cartridge.to('cpu'); initial.to('cpu')
                gc.collect(); torch.cuda.empty_cache()
                pilot, _ = load_cartridge(training['vanilla_checkpoint'], device=engine.device,
                    expected_prefix_tokens=cartridge.num_cartridge_tokens(), expected_model=model)
                pilot_prediction, pilot_metric = evaluate_frozen_cartridge(engine, pilot, dataset, case)
                pilot.clear()
                del pilot
            record = {
                "status": "completed", "dataset": dataset, "case": case["id"],
                'protocol': PROTOCOL, 'backbone': model, 'worker_shard': shard,
                "prefix_tokens": prefix_tokens,
                'actual_prefix_tokens': cartridge.num_cartridge_tokens(),
                "cache_gb_bf16": cache_gb(engine, cartridge.num_cartridge_tokens()),
                "metric": "accuracy" if dataset == "longhealth" else
                          "chrF" if dataset == "mtob_ke" else "log_perplexity",
                "metric_value": metric_value, "prediction": prediction,
                'pilot_prediction': pilot_prediction, 'pilot_metric_value': pilot_metric,
                'pilot_evaluation_post_training_only': True,
                'pilot_accuracy_never_used_to_choose_checkpoint': True,
                "reference": case["reference"],
                "reference_used_only_for_post_training_metric": True,
                'longhealth_answer_tags_present': (
                    bool(re.search(r'<answer>(.*?)</answer>', prediction, flags=re.S))
                    if dataset == 'longhealth' else None),
                "benchmark_answer_used_for_training": False,
                "synthetic_examples": len(synthetic["examples"]),
                "teacher_conversations": len(conversations), "training": training,
                'training_packing': packing_audit,
                'peak_student_memory_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_student_memory_reserved_bytes': torch.cuda.max_memory_reserved(),
                'synthetic_dataset_path': str(shared),
                'teacher_targets_sha256': case_config['teacher_targets_sha256'],
                'teacher_targets_shared_across_cache_sizes': True,
                'native_kv_parameter_change_l2': math.sqrt(delta_sq),
                'returned_checkpoint_updates': returned_training.get(
                    'effective_updates', returned_training['updates']),
                'original_query_teacher_fallback': synthetic.get('original_query_teacher_fallback', False),
                "seconds": time.monotonic() - started,
                "native_kv_only": True, "backbone_frozen": True,
            }
            write(destination, record)
            # Audit JSON retains every trial and loss vector. After a result
            # is durable, keep the original pilot plus final native KV rather
            # than accumulating all rejected large-cache trial checkpoints.
            pilot = Path(training['vanilla_checkpoint'])
            for trial_path in (case_dir / 'fits').glob('*/checkpoint.safetensors'):
                if trial_path != pilot:
                    trial_path.unlink()
            completed.append(case["id"])
            cartridge.clear(); initial.clear()
            del cartridge, initial, conversations
            gc.collect(); torch.cuda.empty_cache()
            print(json.dumps({"event": "figure4_case_complete", "dataset": dataset,
                              "prefix": prefix_tokens, "case": case["id"],
                              "metric": metric_value, "seconds": record["seconds"]}), flush=True)
    except TimeoutError as error:
        write(task_root / f'paused-shard{shard}.json', {
            'error_type': type(error).__name__, 'reason': str(error),
            'completed_fit_checkpoints_preserved': True, 'epoch': time.time()})
        save_status()
        return 75
    except Exception as error:
        write(task_root / f"failure-shard{shard}.json", {"error_type": type(error).__name__,
              "error": str(error), "traceback": traceback.format_exc(), "epoch": time.time()})
        raise
    finally:
        reranker.__exit__(None, None, None)
    save_status()
    # A worker advances after its own disjoint shard, not after other GPUs.
    own_complete = all((task_root / 'cases' / f"{case['id']}.json").exists()
                       for _, case in assigned)
    return 0 if own_complete else 75


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--decision-model", required=True)
    parser.add_argument("--reranker-model", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--case-index', type=int, default=None,
                        help='Run one case, enabling a balanced single-job cache sweep')
    args = parser.parse_args()
    if not 0 <= args.task < len(DATASETS) * len(PREFIXES):
        raise ValueError("task must index the 3x4 matrix")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
