"""Official typed Decisions transport; credentials never enter outputs or caches."""

from contextlib import contextmanager
import hashlib
import fcntl
import json
import math
import os
from pathlib import Path
import socket
import time
import urllib.error
import urllib.request
import uuid

POLICY_VERSION = "shared_question_and_evidence_rubric_v1"
SUPPORTED_POLICIES = (POLICY_VERSION,)
DOCUMENT_POLICY_VERSION = "unused_document_policy"


ENDPOINT = "https://api.openai.com/v1/decisions"


MODEL = "gpt-6-luna"


CHOICES = [
    dict(
        value="yes", description="Possibly helpful, directly or indirectly; uncertainty means yes."
    ),
    dict(value="no", description="Clearly unrelated; no plausible helpful connection."),
]


class DecisionsAPIError(RuntimeError):
    """Sanitized error: never includes headers, credentials or server body."""


def _probability(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def apply_drop_confidence_guard(record, minimum):
    """Retain an uncertain negative without changing the judge's raw evidence.

    This is the same decision rule for live and preserved judgments. Zero is
    the explicit historical opt-out; a positive bound requires BOTH a valid
    negative probability and confidence. Neither quantity is claimed to be
    calibrated evidence of correctness or harmlessness.
    """
    if not _probability(minimum):
        raise ValueError("Decision rejection confidence must be a probability")
    result = dict(record)
    if result.get("accepted", True) or minimum == 0:
        return result
    probabilities = result.get("decision_probabilities")
    probability = (
        probabilities.get("no", probabilities.get("DROP"))
        if isinstance(probabilities, dict)
        else None
    )
    confidence = result.get("decision_confidence")
    valid_negative = result.get("utility_output_valid") is True and (
        result.get("decision_choice") == "DROP" or result.get("decision_label") == "no"
    )
    if (
        not valid_negative
        or not _probability(probability)
        or not _probability(confidence)
        or min(probability, confidence) < minimum
    ):
        result.update(
            accepted=True,
            negative_list_member=False,
            reason="API_negative_below_required_confidence_keep",
            API_rejection_abstained=True,
        )
    return result


def parse_decisions(response, count):
    """Malformed, duplicate, missing, refused or uncertain decisions retain text."""
    answers = response.get("answers") if isinstance(response, dict) else None
    by_name = {}
    envelope_valid = isinstance(answers, list)
    if envelope_valid:
        for answer in answers:
            name = answer.get("name") if isinstance(answer, dict) else None
            if not isinstance(name, str) or name not in {f"chunk_{i}" for i in range(count)}:
                envelope_valid = False
                break
            by_name.setdefault(name, []).append(answer)
    records = []
    for i in range(count):
        matches = by_name.get(f"chunk_{i}", []) if envelope_valid else []
        answer = matches[0] if len(matches) == 1 else {}
        raw = answer.get("probabilities", [])
        probabilities = {}
        # Legacy responses remain readable; live v2 requests use exactly yes/no.
        labels = (
            {"yes", "no"}
            if answer.get("choice") in ("yes", "no")
            else {"KEEP", "DROP", "UNCERTAIN"}
        )
        valid = answer.get("type") == "choice" and answer.get("choice") in labels
        valid = valid and isinstance(raw, list) and _probability(answer.get("confidence"))
        if valid:
            for entry in raw:
                if (
                    not isinstance(entry, dict)
                    or entry.get("value") not in labels
                    or entry["value"] in probabilities
                    or not _probability(entry.get("probability"))
                ):
                    valid = False
                    break
                probabilities[entry["value"]] = entry["probability"]
            valid = valid and set(probabilities) == labels
            valid = valid and abs(sum(probabilities.values()) - 1) <= 0.02
            valid = valid and probabilities[answer["choice"]] >= max(probabilities.values()) - 1e-6
        label = answer.get("choice") if valid else None
        choice = {"yes": "KEEP", "no": "DROP"}.get(label, label)
        drop = choice == "DROP"
        records.append(
            dict(
                chunk=i,
                accepted=not drop,
                reason="utility_explicit_useless_exclusion"
                if drop
                else "utility_uncertain_or_useful_keep",
                utility_output_valid=bool(valid),
                negative_list_member=drop,
                model_selected_useful=False if drop else True if choice == "KEEP" else None,
                decision_choice=choice,
                decision_label=label,
                decision_probabilities=probabilities if valid else {},
                decision_confidence=answer.get("confidence") if valid else None,
                decision_answer_type=answer.get("type"),
                judgment_format="typed_decision_choice",
                full_subset_grammar_scored=False,
                semantic_target="any_direct_or_indirect_chunk_question_relevance",
                irrelevant_rejection_only=True,
                fraction_cap=False,
            )
        )
    return records


class DecisionClient:
    def __init__(self, cfg, budget, *, transport=None):
        self.budget = budget
        self.requests = []
        self.set_policy_version(cfg.get("decision_policy_version", POLICY_VERSION))
        self.manifest = dict(repo="OpenAI/gpt-6-luna", revision="served_API_version_not_pinned")
        self.exclusion_threshold = None  # categorical DROP, not an invented score threshold
        self.minimum_drop_confidence = cfg.get("decision_minimum_drop_confidence", 0.0)
        if not _probability(self.minimum_drop_confidence):
            raise ValueError("Rejection confidence must be a probability")
        self.timeout = float(cfg.get("decision_timeout_seconds", 30))
        if not 0 < self.timeout <= 45:
            raise ValueError("Decision timeout must be between 0 and 45 seconds")
        if (
            cfg.get("decision_endpoint", ENDPOINT) != ENDPOINT
            or cfg.get("decision_model", MODEL) != MODEL
        ):
            raise ValueError("Use the documented official Decisions endpoint and model")
        self._transport = transport
        self.cost_ledger = (
            DecisionCostLedger(cfg["decision_budget_ledger"], cfg.get("decision_max_cost_usd", 5))
            if cfg.get("decision_budget_ledger")
            else None
        )
        if transport is None and self.cost_ledger is None:
            raise DecisionsAPIError(
                "Live Decisions calls require the shared authorized cost ledger"
            )
        self._diagnostic_cache = None
        self.shared_cache = (
            Path(cfg["decision_cache_directory"]).resolve()
            if cfg.get("decision_cache_directory")
            else None
        )
        if self.shared_cache is not None:
            self.shared_cache.mkdir(parents=True, exist_ok=True)
        self._key = None
        if transport is None:
            self._key = os.environ.get("OPENAI_API_KEY")
            if not self._key:
                path = cfg.get("decision_api_key_file") or os.environ.get("OPENAI_API_KEY_FILE")
                if not path:
                    raise DecisionsAPIError("Decisions API credential is not configured")
                credential = Path(path).expanduser().resolve()
                if credential.stat().st_mode & 0o077:
                    raise DecisionsAPIError(
                        "Decisions credential file must have private permissions"
                    )
                self._key = credential.read_text().strip()
            if not self._key:
                raise DecisionsAPIError("Decisions API credential is empty")

    def set_policy_version(self, version):
        """Select a frozen policy only between requests, never in a worker."""
        if version not in SUPPORTED_POLICIES:
            raise ValueError("Unknown Decisions policy version")
        self.policy_version = version

    @contextmanager
    def diagnostic_replay(self):
        """Synthetic scheduling check only; real experiment calls are uncached."""
        if self._diagnostic_cache is not None:
            raise RuntimeError("Nested diagnostic replay")
        self._diagnostic_cache = {}
        try:
            yield
        finally:
            self._diagnostic_cache = None

    def _post(self, payload):
        if self._transport is not None:
            return self._transport(payload)
        request = urllib.request.Request(
            ENDPOINT,
            data=json.dumps(payload).encode(),
            headers={"Authorization": "Bearer " + self._key, "Content-Type": "application/json"},
        )
        self.budget.check()
        remaining = (
            getattr(self.budget, "deadline", time.monotonic() + self.timeout) - time.monotonic()
        )
        if remaining <= 0:
            self.budget.check()
        with urllib.request.urlopen(
            request, timeout=min(self.timeout, max(0.1, remaining))
        ) as response:
            return json.load(response)

    def classify_payload(self, payload):
        """Use the existing locked cache and ledger for a typed chunk payload."""
        self.budget.check()
        if (
            not isinstance(payload, dict)
            or payload.get("model") != MODEL
            or not isinstance(payload.get("input"), str)
            or not isinstance(payload.get("questions"), list)
            or not payload["questions"]
        ):
            raise ValueError("A documented Decisions model, text input and questions are required")
        evidence = json.loads(payload["input"])
        if not isinstance(evidence, dict):
            raise ValueError("Chunk decision evidence must be a structured object")
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if self.shared_cache is None or self._diagnostic_cache is not None:
            return self._classify_payload(payload, digest)
        # Two method variations share reference-blind judgments, not answers.
        # Lock per payload so simultaneous workers cannot pay twice.
        file = self.shared_cache / (digest + ".json")
        wait_started = time.perf_counter()
        with (self.shared_cache / (digest + ".lock")).open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            lock_wait = time.perf_counter() - wait_started
            cached = json.loads(file.read_text()) if file.exists() else None
            records = self._classify_payload(payload, digest, shared_response=cached)
            self.requests[-1]["shared_cache_lock_wait_seconds"] = lock_wait
            self.requests[-1]["wall_seconds"] += lock_wait
            if cached is None and self._last_valid_response is not None:
                temporary = file.with_suffix(".tmp")
                temporary.write_text(json.dumps(self._last_valid_response) + "\n")
                os.replace(temporary, file)
            return records

    def _classify_payload(self, payload, digest, *, shared_response=None):
        began = time.perf_counter()
        cache = self._diagnostic_cache
        hit = cache is not None and digest in cache
        shared_hit = shared_response is not None
        self._last_valid_response = None
        error, status, reservation, called = None, 200, None, False
        if hit or shared_hit:
            response = shared_response if shared_hit else cache[digest]
        else:
            if self.cost_ledger is not None:
                reservation = self.cost_ledger.reserve(payload)
            try:
                if self.cost_ledger is not None and reservation is None:
                    response, status, error = {}, None, "cost_budget_unavailable_keep_all"
                else:
                    called = True
                    response = self._post(payload)
            except urllib.error.HTTPError as exc:
                status = exc.code
                if status in (408, 429) or 500 <= status < 600:
                    response, error = {}, f"http_{status}_keep_all"
                else:
                    raise DecisionsAPIError(
                        f"Decisions API rejected request: HTTP {status}"
                    ) from None
            except (urllib.error.URLError, TimeoutError, socket.timeout, json.JSONDecodeError):
                response, status, error = {}, None, "transport_or_response_error_keep_all"
            if cache is not None:
                cache[digest] = response
            if reservation is not None:
                raw_usage = response.get("usage") if isinstance(response, dict) else None
                tokens = raw_usage.get("input_tokens") if isinstance(raw_usage, dict) else None
                self.cost_ledger.settle(reservation, tokens)
        self.budget.check()
        if isinstance(response, dict) and response.get("model", MODEL) != MODEL:
            response, error = {}, "unexpected_model_keep_all"
        records = [
            apply_drop_confidence_guard(record, self.minimum_drop_confidence)
            for record in parse_decisions(response, len(payload["questions"]))
        ]
        if called and not error and all(r["utility_output_valid"] for r in records):
            self._last_valid_response = response
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        usage = (
            {
                k: v
                for k, v in usage.items()
                if k in ("input_tokens", "output_tokens", "total_tokens")
                and isinstance(v, int)
                and not isinstance(v, bool)
                and v >= 0
            }
            if isinstance(usage, dict)
            else {}
        )
        evidence = json.loads(payload["input"])
        complete_document = evidence.get("context_scope") == "complete_original_document"
        self.requests.append(
            dict(
                model=MODEL,
                endpoint=ENDPOINT,
                policy_version=self.policy_version,
                chunk_count=len(payload["questions"]),
                judgment_context=(
                    "complete_chronological_request_text"
                    if complete_document
                    else "whole_window_peers_and_published_history"
                ),
                background_chunk_count=len(evidence.get("background_chunks", [])),
                peer_chunks_per_candidate=len(payload["questions"])
                - 1
                + len(evidence.get("other_published_chunks", [])),
                source_text_truncated=False,
                whole_chunks_only=True,
                gold_answers_used=False,
                minimum_API_drop_confidence=self.minimum_drop_confidence,
                wall_seconds=time.perf_counter() - began,
                http_status=status,
                error=error,
                usage=usage,
                diagnostic_cache_hit=hit,
                diagnostic_cache_enabled=cache is not None,
                shared_judgment_cache_hit=shared_hit,
                shared_judgment_cache_enabled=self.shared_cache is not None,
                transport_called=called,
                request_sha256=digest,
                discarded_local_ids=[r["chunk"] for r in records if not r["accepted"]],
                invalid_or_refused_count=sum(not r["utility_output_valid"] for r in records),
                estimated_cost_usd=(
                    usage.get("input_tokens", 0) * 0.1 / 1_000_000
                    if not hit and not shared_hit
                    else 0
                ),
                cost_reservation_id=reservation,
                decision_format="typed_decision_choice",
                execution="CPU_network_worker",
            )
        )
        return records


class DecisionCostLedger:
    """Shared authorized envelope, locked across workers and restarts.

    Reserve a conservative UTF-8 byte bound plus protocol headroom before calling.
    Unknown usage (timeout/crash) keeps the reservation. It is never refunded on a
    restart. Known usage replaces the reservation at the documented standard rate.
    """

    def __init__(self, path, cap=5):
        if not isinstance(cap, (int, float)) or isinstance(cap, bool) or not 0 < cap <= 50:
            raise ValueError("The authorized Decisions API budget is at most $50")
        self.path = Path(path).expanduser().resolve()
        self.cap = float(cap)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    @contextmanager
    def _locked(self):
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = (
                json.loads(self.path.read_text())
                if self.path.exists()
                else dict(
                    cap_usd=self.cap,
                    billed_estimate_usd=0,
                    reservations={},
                    completed_calls=0,
                    known_input_tokens=0,
                    rate_usd_per_million_input_tokens=0.1,
                )
            )
            if state["cap_usd"] != self.cap:
                raise ValueError("Shared API budget cap changed")
            yield state
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, indent=2) + "\n")
            os.replace(temporary, self.path)
            fcntl.flock(lock, fcntl.LOCK_UN)

    def reserve(self, payload):
        encoded = json.dumps(payload, ensure_ascii=False).encode()
        upper_tokens = 2 * len(encoded) + 8192 + 1024 * len(payload["questions"])
        amount = max(0.01, upper_tokens * 0.1 / 1_000_000)
        with self._locked() as state:
            committed = state["billed_estimate_usd"] + sum(state["reservations"].values())
            if committed + amount > self.cap:
                return None
            identity = uuid.uuid4().hex
            state["reservations"][identity] = amount
            return identity

    def settle(self, identity, tokens):
        with self._locked() as state:
            if identity not in state["reservations"]:
                raise ValueError("Unknown API cost reservation")
            if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0:
                cost = tokens * 0.1 / 1_000_000
                if cost > state["reservations"][identity]:
                    raise DecisionsAPIError(
                        "Actual input usage exceeded conservative request reservation"
                    )
                del state["reservations"][identity]
                state["billed_estimate_usd"] += cost
                state["known_input_tokens"] += tokens
                state["completed_calls"] += 1
