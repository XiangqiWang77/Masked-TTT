"""Render results only after every required paired case is complete."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("Qwen3-4B", "OLMo-3-7B", "GPT-OSS-20B")
METHODS = ("full_context_icl", "agnostic_vanilla_sd", "agnostic_selective_sd",
           "specific_vanilla_sd", "specific_selective_sd")
LENGTHS = (8192, 16384, 32768, 65536)


def display(value, latex=False, signed=False):
    if value is None: return r"\textemdash" if latex else "—"
    return f"{value:+.2f}" if signed else f"{value:.2f}"


def accuracy(rows, model, method, length=None):
    selected = [r for r in rows if r["model"] == model and r["method"] == method
                and (length is None or r["length"] == length)]
    if len(selected) != (4 if length is None else 1):
        raise ValueError("Missing or duplicated table row")
    for r in selected:
        if r["status"] != "complete" or r["correct"] is None: return None
        if r["n_completed"] != r["n_expected"] or not r["source"]:
            raise ValueError("Complete accuracy requires full denominator and source")
        if not 0 <= r["correct"] <= r["n_completed"]: raise ValueError("Invalid correct count")
        if r["accuracy_percent"] is not None and abs(r["accuracy_percent"] - 100*r["correct"]/r["n_completed"]) > .005:
            raise ValueError("Accuracy and count disagree")
    return 100*sum(r["correct"] for r in selected)/sum(r["n_completed"] for r in selected)


def table(rows, by_length=False):
    head = ["Model"] + (["Length"] if by_length else []) + ["Full-context ICL", "Agnostic + V-SD", "Agnostic + S-SD", "Specific + V-SD", "Contextual (Specific + S-SD)", "Δ selective (pp)"]
    md = ["| " + " | ".join(head) + " |", "|"+"|".join(["---"]*len(head))+"|"]
    body = []
    for model in MODELS:
        for length in LENGTHS if by_length else (None,):
            prefix = [model]+([f"{length//1024}K"] if by_length else [])
            v = [accuracy(rows,model,m,length) for m in METHODS]
            delta = None if v[-1] is None or v[-2] is None else v[-1]-v[-2]
            md.append("| "+" | ".join(prefix+[display(x) for x in v]+[display(delta,signed=True)])+" |")
            body.append(" & ".join(prefix+[display(x,True) for x in v]+[display(delta,True,True)])+r" \\")
    cols="llrrrrrr" if by_length else "lrrrrrr"
    tex=[r"\begin{table*}[t]",r"\centering",r"\small",r"\setlength{\tabcolsep}{4pt}",
         r"\caption{"+("BABILong accuracy by document length (250 requests per cell). " if by_length else "BABILong accuracy over 1,000 requests per backbone. ")+
         r"V-SD and S-SD denote ordinary and selective self-distillation. Contextual uses query-specific data and S-SD. $\Delta$ is S-SD minus V-SD on the same query-specific data. All entries are pending (\textemdash); historical configurations are not substituted.}",
         r"\label{tab:babilong-"+("length" if by_length else "overall")+"}",
         r"\begin{tabular}{"+cols+"}",r"\toprule",
         (r"Model & Length & ICL & Agn./V-SD & Agn./S-SD & Spec./V-SD & Contextual & $\Delta$ \\" if by_length else
          r"Model & ICL & Agn./V-SD & Agn./S-SD & Spec./V-SD & Contextual & $\Delta$ \\"),
         r"\midrule",*body,r"\bottomrule",r"\end{tabular}",r"\end{table*}"]
    return "\n".join(md)+"\n","\n".join(tex)+"\n"


def main():
    rows=json.loads((ROOT/"data/babilong_results.json").read_text())["rows"]
    for length,stem in ((False,"babilong_overall"),(True,"babilong_by_length")):
        md,tex=table(rows,length)
        (ROOT/"tables"/f"{stem}.md").write_text(md)
        (ROOT/"tables"/f"{stem}.tex").write_text(tex)
    s=json.loads((ROOT/"data/synthetic_summary.json").read_text())["metrics"]
    labels={"no_support":("Zero-support contexts (%)",100),"recall":("Support-fact recall (%)",100),
            "joint":("All-fact co-exposure (%)",100),"context_chars":("Evidence characters",1),
            "question_words":("Question words",1)}
    tex=[r"\begin{table*}[t]",r"\centering",r"\small",
         r"\caption{Matched synthetic-data audit: 26,451 exercises per method across 1,000 requests. Means weight requests equally. All-fact co-exposure uses 352 multi-fact requests; recall uses 994 paired requests. These source-coverage proxies do not measure answer correctness or establish necessary multi-hop reasoning.}",
         r"\label{tab:synthetic}",r"\begin{tabular}{lrrrr}",r"\toprule",
         r"Metric & Paired $n$ & Query-agnostic & Query-specific & Difference \\",r"\midrule"]
    for k,(label,scale) in labels.items():
        v=s[k];label=label.replace('%',r'\%')
        tex.append(f"{label} & {v['n']} & {v['vanilla']*scale:.2f} & {v['query_specific']*scale:.2f} & {v['delta']*scale:+.2f}"+r" \\")
    tex += [r"\bottomrule",r"\end{tabular}",r"\end{table*}"]
    (ROOT/"tables/synthetic.tex").write_text("\n".join(tex)+"\n")


if __name__=="__main__": main()
