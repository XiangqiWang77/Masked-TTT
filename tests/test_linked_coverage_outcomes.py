import json

from experiments.babilong_linked_selection.coverage_outcomes import audit, coverage_group, summarize


def test_coverage_groups_distinguish_recovery_loss_and_uncertainty():
    assert coverage_group("incomplete", "complete") == "newly_complete"
    assert coverage_group("complete", "incomplete") == "lost_complete"
    assert coverage_group("complete", "complete") == "both_complete"
    assert coverage_group("incomplete", "incomplete") == "both_incomplete"
    assert coverage_group("unknown", "complete") == "ambiguous"


def test_summary_uses_paired_outcomes():
    rows = [
        dict(score10=0, linked=1, linked_zero=0, linked_teacher=1),
        dict(score10=1, linked=0, linked_zero=0, linked_teacher=0),
        dict(score10=1, linked=1, linked_zero=1, linked_teacher=1),
    ]
    result = summarize(rows)
    assert result["n"] == 3 and result["gains"] == result["losses"] == 1
    assert result["delta_pp"] == 0
    assert summarize([])["delta_pp"] is None


def test_audit_requires_exact_selected_chunks_and_scores_after_generation(tmp_path):
    def saved(chunks, prediction):
        return dict(
            key="case", model="qwen", task="qa1",
            selection=dict(complete_original_document=True,
                selected_original_chunks=[[item] for item in chunks],
                online_api_calls=0, reference_answers_used=False),
            training=dict(selected_transfer_weight=0),
            teacher=dict(prediction="garden"),
            standalone_cartridge=dict(paired_prompt_control=dict(
                trained_original=dict(prediction=prediction),
                zero_original=dict(prediction="office"))),
        )
    old=tmp_path/"old/cases/case/weight_0.00.json"
    new=tmp_path/"new/cases/qwen/case/result.json"
    for path, data in ((old,saved([1,2],"office")),(new,saved([1,3],"garden"))):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(data))
    coverage={"rows":[dict(case="case",outcomes={
        "score10":dict(status="incomplete",selected=[1,2]),
        "entity10":dict(status="complete",selected=[1,3]),
    })]}
    refs={"case":dict(task="qa1",answers=["garden"])}
    grade=lambda text,answers,benchmark,task:(int(text in answers),True)
    result=audit(["case"],coverage,refs,tmp_path/"old",tmp_path/"new",grade)
    assert result["overall"]["n"]==1
    assert result["groups"]["newly_complete"]["gains"]==1
    assert result["online_api_calls"]==result["new_model_calls"]==0
    assert not result["reference_answers_used_in_inference"]
    coverage["rows"][0]["outcomes"]["entity10"]["selected"]=[1,4]
    rejected=audit(["case"],coverage,refs,tmp_path/"old",tmp_path/"new",grade)
    assert rejected["overall"]["n"]==0 and len(rejected["unavailable"])==1
