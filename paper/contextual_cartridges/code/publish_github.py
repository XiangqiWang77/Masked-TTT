"""Publish this reviewable artifact package without touching experiment jobs.

Uses the already-authorized gh CLI when the connector cannot access this repo.
Only fast-forward branch updates are allowed; unrelated paths are preserved.
"""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
REPO = "XiangqiWang77/Masked-TTT"
BRANCH = "report/query-cartridges-motivation-only"
PREFIX = "paper/contextual_cartridges/"


def api(endpoint, data=None, method=None):
    command = ["gh", "api", "repos/"+REPO+"/"+endpoint]
    if method: command += ["--method", method]
    if data is not None: command += ["--input", "-"]
    p = subprocess.run(command, input=None if data is None else json.dumps(data),
                       capture_output=True, text=True, timeout=90)
    if p.returncode: raise RuntimeError(p.stderr[-1500:])
    return json.loads(p.stdout)


def blob_sha(content):
    return hashlib.sha1(b"blob "+str(len(content)).encode()+b"\0"+content).hexdigest()


def main():
    required = ["paper.pdf", "README.md", "METHOD.zh-CN.md", "RESULTS.md", "review/audit.json"]
    if any(not (ROOT/p).exists() for p in required): raise ValueError("Incomplete paper package")
    audit=json.loads((ROOT/"review/audit.json").read_text())
    if not audit["passed"]: raise ValueError("Artifact audit did not pass")
    entries={}
    extensions={".md", ".json", ".csv", ".py", ".tex", ".sty", ".pdf", ".png", ".svg"}
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.suffix not in extensions: continue
        if path.name == "publication.json" or "__pycache__" in path.parts: continue
        entries[PREFIX+str(path.relative_to(ROOT))]=path.read_bytes()
    head=api("git/ref/heads/"+BRANCH)["object"]["sha"]
    current=api("git/trees/"+head+"?recursive=1")
    existing={r["path"]:r.get("sha") for r in current["tree"]}
    old_readme=api("contents/README.md?ref="+quote(BRANCH,safe=""))
    old=base64.b64decode(old_readme["content"]).decode()
    marker="<!-- contextual-acl-package -->"
    if marker in old:
        old=old.split("<!-- previous-report -->",1)[-1].strip()
    landing="\n".join([
        marker,
        "# Contextual Cartridges",
        "",
        "**Query-Guided Test-Time Learning with Selective Self-Distillation**",
        "",
        f"[ACL paper and complete results package]({PREFIX}README.md) · [Paper PDF]({PREFIX}paper.pdf) · [完整中文方法]({PREFIX}METHOD.zh-CN.md) · [BABILong 待填结果表]({PREFIX}RESULTS.md)",
        "",
        "1,000 BABILong requests; 26,451 count-matched synthetic exercises per method. Three separate experimental figures, using only axes, legends, and the blue–cyan / yellow–orange palette. No bars or confidence bands.",
        "",
        *[f"![{label}]({PREFIX}figures/{name}.png)\n" for label,name in [
          ("Source dilution and all-fact co-exposure","figure1_dilution_fragmentation"),
          ("Task-by-length paired contrasts","figure2_task_length"),
          ("Paired source recall and source footprint","figure3_paired_coverage_footprint")]],
        f"[Definitions, denominators, figure PDFs/SVGs and source JSON]({PREFIX}README.md). These completed data audits measure source coverage and organization; final answer accuracy is not inferred from them.",
        "",
        f"[Three backbones × three datasets cache–performance template]({PREFIX}figures/figure4_cache_grid_pending.pdf). Downstream tables and cache curves are explicitly pending; no missing result is filled with zero or an older run.",
        "",
        f"[Reproduction code and software references]({PREFIX}REPRODUCE.md).",
        "",
        "<!-- previous-report -->",
    ])
    # Preserve the old report as a separate archive rather than mixing versions.
    archive="archive/README_before_acl_20261111.md"
    if archive not in existing:
        entries[archive]=("Historical report preserved from commit "+head+". Relative paths below refer to the repository root at that commit.\n\n"+old).encode()
    entries["README.md"]=(landing+f"\n[Historical report (previous protocols)]({archive})\n").encode()
    # Replace the old top-level method entry with an unambiguous current pointer.
    entries["CONTEXTUAL_CARTRIDGES_METHOD.zh-CN.md"]=(
        "# Contextual Cartridges\n\n当前完整方法已同步到 [ACL 方法稿]("+PREFIX+"METHOD.zh-CN.md)。\n\n"
        "当前版本是 Answer-Guided Query-Specific Dataset Crafting 与 Conservative TracIn-Guided Selective SD：三个检查点一致负向提名，再用两种固定顺序实际删样本重训确认。保留原生 KV、冻结 backbone 和稀疏软目标 SD；无 held-out 标签。守护对象为保留轨迹的 SD 损失，不保证最终答案零退化。\n").encode()
    blobs=[]
    for index,(path,content) in enumerate(entries.items()):
        sha=blob_sha(content)
        if existing.get(path)==sha: continue
        saved=api("git/blobs", {"content":base64.b64encode(content).decode(),"encoding":"base64"})
        if saved["sha"]!=sha: raise ValueError("Upload content hash mismatch")
        blobs.append({"path":path,"mode":"100644","type":"blob","sha":sha})
        if len(blobs)%10==0: print(json.dumps({"uploaded":len(blobs),"files":len(entries)}),flush=True)
    if not blobs: commit=head
    else:
        for attempt in range(3):
            parent=api("git/ref/heads/"+BRANCH)["object"]["sha"]
            base=api("git/commits/"+parent)["tree"]["sha"]
            tree=api("git/trees",{"base_tree":base,"tree":blobs})
            commit=api("git/commits",{"message":"Add clean ACL figures, current method, and pending evaluation tables","tree":tree["sha"],"parents":[parent]})["sha"]
            try:
                api("git/refs/heads/"+BRANCH,{"sha":commit,"force":False},"PATCH")
                break
            except RuntimeError:
                if attempt==2: raise
    tree=api("git/trees/"+commit+"?recursive=1")
    paths={r["path"]:r.get("sha") for r in tree["tree"]}
    if any(paths.get(p)!=blob_sha(v) for p,v in entries.items()):
        raise ValueError("Published files differ from audited artifacts")
    receipt={"complete":True,"commit":commit,"repository":REPO,"branch":BRANCH,"files":len(entries),
             "url":f"https://github.com/{REPO}/tree/{BRANCH}/{PREFIX}","verified_content_hashes":True}
    (ROOT/"publication.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt),flush=True)


if __name__=="__main__": main()
