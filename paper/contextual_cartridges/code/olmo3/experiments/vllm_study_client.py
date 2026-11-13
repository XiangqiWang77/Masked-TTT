"""Batched local vLLM transport for the unchanged Cartridges synthesizer.

This module changes inference scheduling, not prompts, sample counts or the
Teacher soft-target format. It is deliberately outside ``contextualized`` so
existing HF-generated datasets retain their original source identity. New
runs must separately record this backend and its version.
"""
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from hashlib import sha256
import asyncio
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cartridges.clients.base import Client, ClientConfig, ClientResponse, ClientSample, TopLogprobs
from cartridges.clients.usage import Usage


_REQUEST_SEED = ContextVar('vllm_study_request_seed', default=None)


@contextmanager
def sampling_seed(seed):
    """Give one logical study task a seed stream independent of queue ordering."""
    if type(seed) is not int or seed < 0:
        raise ValueError('A nonnegative integer seed is required')
    token = _REQUEST_SEED.set((seed, 0))
    try:
        yield
    finally:
        _REQUEST_SEED.reset(token)


def request_seed(runtime, prompt):
    state = _REQUEST_SEED.get()
    if state is None:
        state = (runtime.seed, runtime.request_counter)
        runtime.request_counter += 1
    else:
        _REQUEST_SEED.set((state[0], state[1]+1))
    raw = json.dumps([*state, prompt], separators=(',', ':')).encode()
    return int.from_bytes(sha256(raw).digest()[:8], 'big') % (2**31)


class OfflineMicrobatcher:
    """Merge concurrent requests into bounded offline batches on one engine.

    ``LLM.generate`` is blocking; a single worker thread runs it while the
    event loop collects the next batch. No two generate calls overlap on the
    engine. A request's own sampling parameters travel with its token IDs.
    """
    def __init__(self, llm, *, max_sequences=256, wait_seconds=0.01):
        if type(max_sequences) is not int or max_sequences < 1:
            raise ValueError('max_sequences must be a positive integer')
        if not math.isfinite(wait_seconds) or wait_seconds < 0:
            raise ValueError('wait_seconds must be finite and nonnegative')
        self.llm = llm
        self.max_sequences = max_sequences
        self.wait_seconds = wait_seconds
        self.pending = deque()
        self.worker = None
        self.loop = None
        self.failure = None
        self.batch_sizes = []

    async def submit(self, prompts, parameters):
        if len(prompts) != len(parameters) or not prompts:
            raise ValueError('One sampling-parameter object is required per prompt')
        if self.failure is not None:
            raise RuntimeError('The shared vLLM engine previously failed') from self.failure
        loop = asyncio.get_running_loop()
        if self.loop is not None and self.loop is not loop and (self.pending or self.worker and not self.worker.done()):
            raise RuntimeError('An active microbatcher cannot cross event loops')
        self.loop = loop
        futures = []
        for prompt, parameter in zip(prompts, parameters):
            future = loop.create_future()
            self.pending.append((prompt, parameter, future))
            futures.append(future)
        if self.worker is None or self.worker.done():
            self.worker = loop.create_task(self._drain())
        return await asyncio.gather(*futures)

    async def _drain(self):
        while self.pending:
            if len(self.pending) < self.max_sequences and self.wait_seconds:
                await asyncio.sleep(self.wait_seconds)
            batch = [self.pending.popleft() for _ in range(min(len(self.pending), self.max_sequences))]
            batch = [item for item in batch if not item[2].cancelled()]
            if not batch:
                continue
            prompts = [item[0] for item in batch]
            parameters = [item[1] for item in batch]
            try:
                result = await asyncio.to_thread(self.llm.generate, prompts,
                    sampling_params=parameters, use_tqdm=False)
                if len(result) != len(batch):
                    raise ValueError('vLLM returned a different number of completions than requested')
                self.batch_sizes.append(len(batch))
                for (_, _, future), value in zip(batch, result):
                    if not future.done():
                        future.set_result(value)
            except Exception as error:
                self.failure = error
                outstanding = batch+list(self.pending)
                self.pending.clear()
                for _, _, future in outstanding:
                    if not future.done():
                        future.set_exception(error)
                return

    async def wait_idle(self):
        if self.worker is not None:
            await self.worker
        if self.failure is not None:
            raise RuntimeError('The shared vLLM engine failed') from self.failure


def exact_top_logprobs(logprobs, *, count, width, vocab_size):
    """Convert real vLLM token IDs/probabilities without re-tokenization.

    vLLM may return the sampled token in addition to its top K. Keep exactly
    the highest K raw log probabilities, matching the existing HF client.
    ``FlatLogprobs`` is consumed directly to avoid creating per-token dicts.
    """
    if logprobs is None or len(logprobs) != count:
        raise ValueError('Teacher log probabilities do not cover every generated token')
    flat = all(hasattr(logprobs, key) for key in ('start_indices', 'end_indices', 'token_ids', 'logprobs'))
    if flat:
        if len(logprobs.end_indices) != count or len(logprobs.token_ids) != len(logprobs.logprobs):
            raise ValueError('Malformed flat Teacher log probabilities')
    ids = np.empty((count, width), dtype=np.int32)
    values = np.empty((count, width), dtype=np.float32)
    for position in range(count):
        if flat:
            start, end = logprobs.start_indices[position], logprobs.end_indices[position]
            if not (type(start) is int and type(end) is int and 0 <= start <= end <= len(logprobs.token_ids)):
                raise ValueError('Invalid flat Teacher probability offsets')
            pairs = list(zip(logprobs.token_ids[start:end], logprobs.logprobs[start:end]))
        else:
            row = logprobs[position]
            if not isinstance(row, dict):
                raise ValueError('Each Teacher probability row must contain real vocabulary IDs')
            pairs = [(key, value.logprob) for key, value in row.items()]
        checked = {}
        for token_id, value in pairs:
            if type(token_id) is not int or not 0 <= token_id < vocab_size:
                raise ValueError('Invalid Teacher vocabulary ID; placeholder IDs are forbidden')
            value = float(value)
            if not math.isfinite(value) or value > 1e-5:
                raise ValueError('Teacher probabilities must be finite natural-log probabilities')
            if token_id in checked and checked[token_id] != value:
                raise ValueError('Conflicting probabilities for the same vocabulary ID')
            checked[token_id] = value
        if len(checked) < width:
            raise ValueError('Teacher returned fewer than the requested top-K probabilities')
        ordered = sorted(checked.items(), key=lambda item: (-item[1], item[0]))[:width]
        ids[position] = [item[0] for item in ordered]
        values[position] = [item[1] for item in ordered]
    return TopLogprobs(token_ids=ids, logprobs=values)


class VLLMStudyClient(Client):
    class Config(ClientConfig):
        runtime_key: str

    def __init__(self, config):
        super().__init__(config)
        from contextualized.upstream_study import _RUNTIMES
        self.engine = _RUNTIMES[config.runtime_key]

    async def chat(self, chats, temperature=0.6, max_completion_tokens=1024,
                   top_logprobs=None, enable_thinking=False, stop=None, **kwargs):
        runtime = self.engine
        if not chats or type(max_completion_tokens) is not int or max_completion_tokens < 1:
            raise ValueError('Nonempty chats and a positive completion budget are required')
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError('Temperature must be finite and nonnegative')
        if top_logprobs is not None and (type(top_logprobs) is not int or not 1 <= top_logprobs <= runtime.max_logprobs):
            raise ValueError('Requested Teacher top-K exceeds the registered runtime limit')
        if stop:
            raise ValueError('String-stop truncation is not part of the frozen study protocol')
        if kwargs.get('cartridges') or kwargs.get('frequency_penalty', 0) != 0:
            raise ValueError('This frozen Teacher client does not apply cartridges or penalties')
        prompts = [runtime.tokenizer.apply_chat_template(chat, tokenize=True,
            add_generation_prompt=True, enable_thinking=enable_thinking) for chat in chats]
        if any(not ids or any(type(value) is not int or not 0 <= value < runtime.vocab_size for value in ids)
               for ids in prompts):
            raise ValueError('Tokenizer returned invalid prompt token IDs')
        if any(len(ids)+max_completion_tokens > runtime.max_model_len for ids in prompts):
            raise ValueError('Study prompt plus completion budget exceeds the context limit; no truncation allowed')
        parameters = [runtime.sampling_params_factory(n=1, temperature=temperature, top_p=1.0, top_k=-1,
            min_p=0.0, repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0,
            max_tokens=max_completion_tokens, seed=request_seed(runtime, ids), logprobs=top_logprobs,
            flat_logprobs=True, stop_token_ids=sorted(runtime.eos_ids), ignore_eos=False,
            detokenize=False) for ids in prompts]
        outputs = await runtime.batcher.submit([{'prompt_token_ids': ids} for ids in prompts], parameters)
        samples = []
        for prompt, output in zip(prompts, outputs):
            if list(output.prompt_token_ids) != prompt or len(output.outputs) != 1 or not output.finished:
                raise ValueError('vLLM response does not match its complete original prompt')
            completion = output.outputs[0]
            tokens = list(completion.token_ids)
            if (not tokens or len(tokens) > max_completion_tokens
                    or any(type(value) is not int or not 0 <= value < runtime.vocab_size for value in tokens)):
                raise ValueError('Invalid or empty generated token IDs')
            # The HF client includes the first EOS token, then trims padding.
            end = next((i+1 for i, token in enumerate(tokens) if token in runtime.eos_ids), len(tokens))
            if end != len(tokens):
                raise ValueError('vLLM produced tokens after a registered EOS')
            top = exact_top_logprobs(completion.logprobs, count=len(tokens), width=top_logprobs,
                                    vocab_size=runtime.vocab_size) if top_logprobs is not None else None
            samples.append(ClientSample(text=runtime.tokenizer.decode(tokens, skip_special_tokens=True),
                                        token_ids=tokens, top_logprobs=top))
        usage = Usage(prompt_tokens=sum(map(len, prompts)), completion_tokens=sum(len(s.token_ids) for s in samples))
        runtime.study_usage.append({'prompt_tokens': usage.prompt_tokens, 'completion_tokens': usage.completion_tokens,
                                   'teacher_soft_targets_requested': top_logprobs is not None,
                                   'backend': 'vllm_raw_logprobs'})
        return ClientResponse(samples=samples, usage=usage)

    async def teacher_force(self, chats, answer_token_ids, *, top_logprobs=20,
                            enable_thinking=False):
        """Recompute sparse Teacher targets on fixed assistant trajectories.

        SelfStudySynthesizer saves the assistant turn generated together with
        each synthetic question.  vLLM prompt log-probabilities let us score
        that *same* turn under a chosen source context instead of silently
        replacing it with a second free generation.  The returned probabilities
        cover exactly the answer tokens; the one decode token needed by vLLM is
        discarded and never becomes part of the training conversation.
        """
        runtime = self.engine
        if (not chats or len(chats) != len(answer_token_ids)
                or type(top_logprobs) is not int
                or not 1 <= top_logprobs <= runtime.max_logprobs):
            raise ValueError('Fixed Teacher scoring needs aligned chats, answers, and a valid top-K')
        prefixes = [runtime.tokenizer.apply_chat_template(chat, tokenize=True,
            add_generation_prompt=True, enable_thinking=enable_thinking) for chat in chats]
        answers = [list(map(int, values)) for values in answer_token_ids]
        if any(not values or any(not 0 <= value < runtime.vocab_size for value in values)
               for values in answers):
            raise ValueError('Fixed answer trajectories contain invalid token IDs')
        prompts = [prefix+answer for prefix, answer in zip(prefixes, answers)]
        if any(len(ids)+1 > runtime.max_model_len for ids in prompts):
            raise ValueError('Fixed Teacher prompt plus one scoring token exceeds the context limit; no truncation allowed')
        parameters = [runtime.sampling_params_factory(n=1, temperature=0.0, top_p=1.0,
            top_k=-1, min_p=0.0, repetition_penalty=1.0, presence_penalty=0.0,
            frequency_penalty=0.0, max_tokens=1, seed=request_seed(runtime, ids),
            logprobs=None, prompt_logprobs=top_logprobs, flat_logprobs=True,
            stop_token_ids=sorted(runtime.eos_ids), ignore_eos=False, detokenize=False)
            for ids in prompts]
        outputs = await runtime.batcher.submit(
            [{'prompt_token_ids': ids} for ids in prompts], parameters)
        samples = []
        for prefix, answer, prompt, output in zip(prefixes, answers, prompts, outputs):
            if list(output.prompt_token_ids) != prompt or not output.finished:
                raise ValueError('vLLM fixed-trajectory response differs from its exact prompt')
            probabilities = output.prompt_logprobs
            if probabilities is None or len(probabilities) != len(prompt):
                raise ValueError('Teacher prompt probabilities do not cover the fixed trajectory')
            answer_probabilities = probabilities[len(prefix):len(prefix)+len(answer)]
            top = exact_top_logprobs(answer_probabilities, count=len(answer),
                                     width=top_logprobs, vocab_size=runtime.vocab_size)
            samples.append(ClientSample(
                text=runtime.tokenizer.decode(answer, skip_special_tokens=True),
                token_ids=answer, top_logprobs=top))
        usage = Usage(prompt_tokens=sum(map(len, prompts)), completion_tokens=0)
        runtime.study_usage.append({'prompt_tokens': usage.prompt_tokens,
            'completion_tokens': 0, 'teacher_soft_targets_requested': True,
            'fixed_assistant_trajectory': True, 'backend': 'vllm_raw_prompt_logprobs'})
        return ClientResponse(samples=samples, usage=usage)


def load_vllm_runtime(model, tokenizer=None, *, seed=42, max_batch_sequences=256,
                      batch_wait_seconds=0.01, max_model_len=65536,
                      gpu_memory_utilization=0.9, max_num_batched_tokens=8192,
                      enforce_eager=False):
    """Load one local BF16 Qwen replica. GPU/backend imports are intentionally lazy."""
    model = Path(model).resolve()
    config = json.loads((model/'config.json').read_text())
    if config.get('model_type') not in ('qwen3', 'olmo3'):
        raise ValueError('The accelerated data-only protocol currently supports Qwen3 and OLMo-3 only')
    if type(seed) is not int or seed < 0:
        raise ValueError('A nonnegative integer seed is required')
    if type(max_model_len) is not int or max_model_len < 1:
        raise ValueError('A positive context budget is required')
    if type(enforce_eager) is not bool:
        raise ValueError('enforce_eager must be boolean')
    import os
    import sys
    compiler_tools=Path(__file__).resolve().parents[1]/'.deps/toolchain/bin'
    os.environ['PATH']=f'{Path(sys.executable).parent}:{compiler_tools}:'+os.environ.get('PATH','')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, __version__ as vllm_version
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    if tokenizer.chat_template and 'strftime_now("%Y-%m-%d")' in tokenizer.chat_template:
        tokenizer.chat_template = tokenizer.chat_template.replace('strftime_now("%Y-%m-%d")', '"2026-11-03"')
    generation_path = model/'generation_config.json'
    generation = json.loads(generation_path.read_text()) if generation_path.exists() else {}
    eos = generation.get('eos_token_id', config.get('eos_token_id', tokenizer.eos_token_id))
    eos_ids = set(eos if isinstance(eos, list) else [eos]) | {tokenizer.eos_token_id}
    eos_ids.discard(None)
    kwargs = dict(model=str(model), tokenizer=str(model), trust_remote_code=False, dtype='bfloat16',
        tensor_parallel_size=1, seed=seed, max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization, max_num_seqs=max_batch_sequences,
        max_num_batched_tokens=max_num_batched_tokens, enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_logprobs=20, logprobs_mode='raw_logprobs', generation_config='vllm')
    if enforce_eager:
        kwargs['enforce_eager'] = True
    llm = LLM(**kwargs)
    runtime = SimpleNamespace(tokenizer=tokenizer, eos_ids=eos_ids, vocab_size=config['vocab_size'],
        max_model_len=max_model_len, max_logprobs=20, seed=seed, request_counter=0,
        study_usage=[], study_client_type=VLLMStudyClient, sampling_params_factory=SamplingParams,
        batcher=OfflineMicrobatcher(llm, max_sequences=max_batch_sequences, wait_seconds=batch_wait_seconds),
        backend_metadata={'name': 'vllm', 'version': vllm_version, 'engine_options': kwargs,
                          'sampling_seed_scheme': 'sha256(task_seed, request_index, exact_prompt_token_ids)',
                          'same_seed_is_not_bitwise_equivalent_to_hf': True})
    return runtime
