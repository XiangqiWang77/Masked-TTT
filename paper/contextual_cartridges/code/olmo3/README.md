# OLMo-3-7B-Instruct port (full-table campaign, 2026-11-13)

Code the OLMo-3 campaign runs (snapshot source for `code_main_<n>`); paths mirror the Query_Cart repo.
Campaign: LongHealth (patients 1-5, 100 q), QASPER 50, MTOB-KE 50; Whole KV, Evidence ICL, SnapKV,
FastKV (AM), Vanilla Cartridges, Vanilla + DualJSD, Specific + SD, Contextual Cartridges (specific + DualJSD);
budgets 5/10/15/20% of OLMo Whole-KV prefix. No results yet; this is code only.

OLMo-specific changes:
- Synthesis answer prompt `answer_instruction: nonempty_v2` (OLMo takes the `{"answers": []}` escape with the original prompt); Qwen default unchanged.
- 24/32 layers are sliding-window (4096). SnapKV selects only in-window keys on sliding layers; FastKV runs AM on the 8 full-attention layers and keeps the most recent raw K/V on sliding layers. Plain `DynamicCache()` where all slots are needed (DualJSD scorer, baselines, compaction port).
- `models/olmo3` compaction port: no `head_dim` in config, q_norm before head split, YaRN attention_scaling divided out for pure RoPE shifts, fp32 RoPE.
- AM NNLS / value-fit ridge fallback made scale-aware (absolute 1e-6 was singular on OLMo).
- Native HF engine for Cartridge training (`contextualized/native_ratio.py`, `native_backbone.py`).
- No published OLMo per-head budgets, so uniform.
