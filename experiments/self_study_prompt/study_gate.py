"""Only advance the prompt candidate when its saved paired evidence warrants it."""


def probe_decision(report):
    if (report.get("state") != "complete" or report.get("n") != 10 or report.get("expected") != 10 or
        report.get("errors") != 0 or report.get("reference_answers_used") is not False or
        report.get("online_api_calls") != 0 or report.get("training_runtime_changed") is not False or
        report.get("same_selected_teacher_cache") is not True or
        report.get("downstream_accuracy_not_measured") is not True):
        return dict(eligible=False, reason="probe_incomplete_or_invalid")
    a, b = report["base"], report["variant"]
    if a["n"] != 10 or b["n"] != 10:
        raise ValueError("The probe arms are not fully paired")
    eligible = (b["verified_exercises"] > a["verified_exercises"] and
                report["more_verified"] > report["fewer_verified"])
    return dict(eligible=eligible, reason="more_verified_source_exercises" if eligible else
                "no_increase_in_verified_source_exercises", base=a["verified_exercises"],
                candidate=b["verified_exercises"], more=report["more_verified"], fewer=report["fewer_verified"])


def development_decision(report):
    stats = report["overall"]
    if (report["mode"] != "development" or report["expected"] != 200 or stats["n"] != 200 or
        report["metric"] != "answer_focused_lexical_v2" or report["errors"] or report["invalid"] or
        stats["strictly_paired"]["n"] != 200):
        return dict(eligible=False, reason="development_incomplete_or_invalid")
    accuracy = stats["accuracy"]
    difference = {key: accuracy["cartridge"] - accuracy[key] for key in ("whole", "zero", "teacher")}
    difference["original_sd"] = stats["strictly_paired"]["delta_pp"]
    eligible = all(value > 0 for value in difference.values())
    return dict(eligible=eligible, reason="beats_all_paired_development_controls" if eligible else
                "does_not_beat_all_paired_development_controls", delta_pp=difference,
                new_correct=stats["strictly_paired"]["new_correct"],
                new_wrong=stats["strictly_paired"]["new_wrong"])
