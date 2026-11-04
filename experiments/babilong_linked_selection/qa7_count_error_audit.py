#!/usr/bin/env python3
"""Audit saved counting answers and selected source-story coverage after inference."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import re
import statistics
import sys

NUMBERS = {"none":0,"nothing":0,"zero":0,"0":0,"one":1,"1":1,"two":2,"2":2,
           "three":3,"3":3,"four":4,"4":4,"five":5,"5":5}
EMPTY = {"no objects","no object","no items","no item","empty handed"}
QUESTION = re.compile(r"How many objects is ([A-Z][A-Za-z'-]+) carrying\?", re.I)
ARMS = ("whole","teacher","zero","cartridge","zero_teacher_format","cartridge_teacher_format")
EVENT_STATUSES = ("all_present","some_missing","ambiguous","no_extra_events")


def parse_number(text, focused_segment, normalized):
    segment = normalized(focused_segment(text,"qa7"))
    if segment in EMPTY:return 0
    words = segment.split()
    return NUMBERS.get(words[0]) if words else None


def extra_actor_status(document, match, ranges, actor, support):
    """Coverage of non-support source-story sentences mentioning the actor."""
    facts, support_ids = match["facts"], set(match["support"])
    occurrences = support.ordered_occurrences(document,[text for _,text in facts])
    actor_pattern = re.compile(r"\b"+re.escape(actor)+r"\b",re.I)
    statuses = []
    for (number,text),spans in zip(facts,occurrences):
        if number in support_ids or not actor_pattern.search(text):continue
        observed = []
        for left,right in spans:
            while left < right and not document[left].isalnum():left += 1
            while right > left and not document[right-1].isalnum():right -= 1
            if left == right:raise ValueError("Empty actor-story fact")
            observed.append(support.span_covered((left,right),ranges))
        statuses.append("all_present" if all(observed) else "some_missing" if not any(observed) else "ambiguous")
    if not statuses:return "no_extra_events"
    if "some_missing" in statuses:return "some_missing"
    if "ambiguous" in statuses:return "ambiguous"
    return "all_present"


def combine_event_status(statuses):
    if not statuses or any(status not in EVENT_STATUSES for status in statuses):
        raise ValueError("Invalid source-story event coverage")
    return statuses[0] if len(set(statuses)) == 1 else "ambiguous"


def summarize(rows):
    n = len(rows)
    results = {}
    for arm in ARMS:
        parsed = [row for row in rows if row["numbers"][arm] is not None]
        over = sum(row["numbers"][arm] > row["answer"] for row in parsed)
        under = sum(row["numbers"][arm] < row["answer"] for row in parsed)
        equal = sum(row["numbers"][arm] == row["answer"] for row in parsed)
        results[arm] = dict(correct=sum(row["scores"][arm] for row in rows),
            overcount=over, undercount=under, numeric_match=equal, unparsed=n-len(parsed))
    invalid_lengths=[row["study"]["generated_tokens"] for row in rows
                     if row["study"]["rejections"].get("invalid_json",0)]
    return dict(n=n, source_questions=len({row["source"] for row in rows}),
        reference_counts={str(value):sum(row["answer"] == value for row in rows) for value in sorted({row["answer"] for row in rows})},
        support_coverage={name:sum(row["coverage"] == name for row in rows)
                          for name in ("complete","incomplete","ambiguous","unavailable")},
        extra_actor_story_coverage={name:sum(row["actor_coverage"] == name for row in rows) for name in EVENT_STATUSES},
        self_study=dict(verified_at_least_one=sum(row["study"]["verified"]>0 for row in rows),
            invalid_json=sum(bool(row["study"]["rejections"].get("invalid_json")) for row in rows),
            invalid_json_generated_tokens_median=statistics.median(invalid_lengths) if invalid_lengths else None,
            invalid_json_at_token_cap=sum(length==192 for length in invalid_lengths)),
        arms=results)


def collect(workspace, snapshot):
    dev, held = (workspace/name for name in ("query_cartridges_linked_selection_20261104",
                                              "query_cartridges_linked_holdout_20261104"))
    development, holdout = (json.loads((path/"manifest.json").read_text()) for path in (dev,held))
    profile=json.loads((workspace/"query_cartridges_value_lr_20261104/olmo_profile.json").read_text())
    if (snapshot["metric"] != "answer_focused_lexical_v2" or
        snapshot["models"]["qwen"]["state"] != "complete" or
        profile["self_question_max_tokens"] != 192 or
        len(development["case_ids"]) != 200 or len(holdout["case_ids"]) != 800 or
        set(development["case_ids"]) & set(holdout["case_ids"])):
        raise ValueError("The completed Qwen source split changed")
    protocol=json.loads(Path(development["model_protocols"]["qwen"]).read_text())
    if protocol != json.loads(Path(holdout["model_protocols"]["qwen"]).read_text()):
        raise ValueError("The saved model protocol changed")
    index={item["key"]:item for item in json.loads(Path(protocol["index"]).read_text())}
    score_root=Path(protocol["shared_scores"])
    files=json.loads((score_root/"case_to_score_file.json").read_text())
    helper=workspace/"query_cartridges_20261103/audit_babilong_selected_support.py"
    spec=importlib.util.spec_from_file_location("saved_support_audit",helper)
    support=importlib.util.module_from_spec(spec);spec.loader.exec_module(support)
    panel={item["case"]:item for item in json.loads((support.ORIGINAL/"fresh_confirmation.json").read_text())}
    units,matches=support.source_matches(json.loads(support.RESERVATION.read_text()))
    references=json.loads((workspace.parent/"chunked_ntp_query_only_20261029/evaluation_references.json").read_text())
    prior={row["case"]:row for row in json.loads((dev/"coverage.json").read_text())["rows"]}
    sys.path.insert(0,str(workspace/"monitor"))
    from report_hybrid_partial import BABILONG_SCORER_VERSION, focused_segment, normalized, score_prediction
    if BABILONG_SCORER_VERSION != "answer_focused_lexical_v2":raise ValueError("Unexpected answer scorer")
    roots={key:dev for key in development["case_ids"]};roots.update({key:held for key in holdout["case_ids"]})
    cases=[key for key in roots if references[key]["task"] == "qa7"]
    rows=[];checked=0
    with open(protocol["source_cases"],"rb") as source:
        for key in sorted(cases,key=lambda item:index[item]["offset"]):
            entry=index[key];source.seek(entry["offset"]);case=json.loads(source.read(entry["bytes"]))
            result=json.loads((roots[key]/"cases/qwen"/key/"result.json").read_text())
            selected=result["selection"];unit=panel[key]["sampling_unit"]
            match=QUESTION.fullmatch(case["query"].strip())
            if (match is None or case["key"] != key or case["query"].strip() != units[unit]["question"].strip() or
                result["key"] != key or result["task"] != "qa7" or selected["source_case_key"] != key or
                not selected["complete_original_document"] or selected["source_mapping"] is not None or
                selected["selection_policy"] != "decision_entity_link_v1" or selected["reference_answers_used"] or
                selected["online_api_calls"] != 0):
                raise ValueError("The counting question or saved selection changed")
            chosen=sorted(i for group in selected["selected_original_chunks"] for i in group)
            data=json.loads((score_root/"scores"/files[key]).read_text())
            if data["case_key"] != key or data["question"] != case["query"] or data["reference_answers_used"] is not False:
                raise ValueError("The saved Decision scores changed")
            ranges=[tuple(data["source_spans"][i]) for i in chosen]
            support_status=support.combine([support.support_coverage(case["document"],item,ranges,
                strip_boundary_punctuation=True) for item in matches[unit]])["status"]
            actor_status=combine_event_status([extra_actor_status(case["document"],item,ranges,match.group(1),support)
                                               for item in matches[unit]])
            if key in prior:
                original=prior[key]["outcomes"]["entity10"]
                if original["selected"] != chosen or original["status"] != support_status:
                    raise ValueError("The prior QA7 coverage changed")
                checked += 1
            answers=references[key]["answers"]
            possible={NUMBERS[normalized(answer)] for answer in answers if normalized(answer) in NUMBERS}
            if len(possible) != 1:raise ValueError("The counting reference is not a single numerical value")
            answer=possible.pop()
            pair=result["standalone_cartridge"]["paired_prompt_control"]
            if (not pair["same_selected_kv"] or not pair["same_generation_limit"] or
                pair["reference_answers_used"] or pair["teacher_answer_in_student_prompt"]):
                raise ValueError("The saved answer controls changed")
            outputs=dict(whole=result["self_judge"]["native_prediction"],teacher=result["teacher"]["prediction"],
                zero=pair["zero_original"]["prediction"],cartridge=pair["trained_original"]["prediction"],
                zero_teacher_format=pair["zero_teacher_suffix"]["prediction"],
                cartridge_teacher_format=pair["trained_teacher_suffix"]["prediction"])
            numbers={name:parse_number(text,focused_segment,normalized) for name,text in outputs.items()}
            scores={name:int(score_prediction(text,answers,"babilong","qa7")[0]) for name,text in outputs.items()}
            if any(numbers[name] is not None and (numbers[name] == answer) != bool(scores[name]) for name in ARMS):
                raise ValueError("Numeric parsing disagrees with the unified scorer")
            rows.append(dict(source=key.split("-",2)[2],split="development" if roots[key]==dev else "holdout",
                             answer=answer,coverage=support_status,actor_coverage=actor_status,
                             numbers=numbers,scores=scores,study=result["self_study"]))
    if len(rows)!=100 or checked!=20 or len({row["source"] for row in rows})!=25:
        raise ValueError("The 100 counting cases or prior support checks are incomplete")
    overall=summarize(rows);frozen=snapshot["models"]["qwen"]["groups"]["task:qa7"]["accuracy"]
    if any(overall["arms"][arm]["correct"] != round(frozen[arm]) for arm in ARMS):
        raise ValueError("Rescored counting answers differ from the saved 1,000-case report")
    groups={"teacher_wrong":summarize([r for r in rows if not r["scores"]["teacher"]]),
            "teacher_correct":summarize([r for r in rows if r["scores"]["teacher"]]),
            "teacher_wrong_full_correct":summarize([r for r in rows if not r["scores"]["teacher"] and r["scores"]["whole"]])}
    for value in sorted({row["answer"] for row in rows}):
        groups["reference:"+str(value)]=summarize([r for r in rows if r["answer"]==value])
    return dict(metric="answer_focused_lexical_v2",cases=100,source_questions=25,
        question_type="count_objects",prior_development_coverage_checks=checked,
        question_answer_and_annotations_used_only_posthoc=True,no_new_model_calls=True,online_api_calls=0,
        source_story_actor_events_are_descriptive_not_proven_necessary=True,
        overall=overall,groups=groups,
        splits={name:summarize([r for r in rows if r["split"]==name]) for name in ("development","holdout")})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,required=True);parser.add_argument("--snapshot",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    result=collect(args.workspace.resolve(),json.loads(args.snapshot.read_text()))
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+"\n")
    print("overall",json.dumps(result["overall"]));print("teacher_wrong",json.dumps(result["groups"]["teacher_wrong"]))


if __name__ == "__main__":main()
