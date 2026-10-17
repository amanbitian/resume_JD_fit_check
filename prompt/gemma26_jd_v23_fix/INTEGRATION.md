# Integration notes for `resume_JD_fit_check` + dual-GPU llama.cpp service

This ZIP intentionally avoids editing unknown application code. The service and JD project were not locally available as source trees when this ZIP was assembled. The instructions below mark *exact adaptation points* rather than pretending to know your current function signatures.

## A. The critical gateway change

Your current `evaluation` path forwards only `messages`, `temperature`, `top_p`, `max_tokens`, and `seed`. **That path discards `response_format`.** If the new gateway/client cannot pass `response_format` to llama.cpp, JSON-schema constrained decoding will not happen, regardless of prompt wording.

When constructing the OpenAI-compatible `POST /v1/chat/completions` body, add **an opt-in forwarding rule**:

```python
request_body = {
    'model': model_id,
    'messages': messages,
    'temperature': temperature,
    'max_tokens': max_tokens,
    'seed': seed,
}
# Add optional top_p only if one was supplied.
if top_p is not None:
    request_body['top_p'] = top_p
# Forward caller-supplied schema only for the Gemma experiment.
if gemma_schema_trial and response_format is not None:
    request_body['response_format'] = response_format
```

Do not modify Qwen's request body or shared defaults. Verify that the service **actually** forwards this field by inspecting the outbound HTTP payload or the llama.cpp server log. A gateway returning HTTP 200 is not enough proof.

The adapter in `gemma_jd_fix/core.py` supports both of these envelope variants:

```json
{"response_format": {"type": "json_schema", "schema": {"type": "object", "properties": {}, "additionalProperties": false}}}
```

```json
{"response_format": {"type": "json_schema", "json_schema": {"name": "jd_rubric_v23", "strict": true, "schema": {"type": "object", "properties": {}, "additionalProperties": false}}}}
```

Start with `--schema-mode llamacpp`. These snippets illustrate envelopes, not a valid rubric schema or a guarantee that all JSON Schema keywords are enforced. The wrapper **must use your existing actual v2.3 schema**. The runner will use `jsonschema` to independently verify returned content.

## B. Integrate *real* v2.3 semantic checks (fail closed)

The runner cannot infer the validator's import path or callable signature from benchmark logs alone. Create `your_real_v23_adapter.py` in your own project by editing `examples/validator_hook_TEMPLATE.py`.

It must expose:

```python
def validate(rubric: dict, jd: str) -> dict:
    # Call exactly your existing project's v2.3 semantic validation function.
    # Return its authoritative decision:
    return {'accepted': bool_value, 'errors': list_of_errors, 'warnings': list_of_warnings}
```

**Do not** return `accepted=True` merely because JSON parses or the schema passes. Keep original hard gate, skill coverage, exact quote, logistic condition and priority validation rules intact. All logs record whether a semantic hook was provided.

## C. Schema limitations

The grammar compiler supports only a JSON-Schema subset in some releases; unsupported features may be ignored. The client warns for common problematic keys (`uniqueItems`, conditionals, `contains`, etc.). Do not automatically delete unsupported schema properties to make compilation pass, because that silently weakens the validation contract. If you need a grammar-compatible version, build a separate inference-only schema with documented differences, and **continue validation with the untouched full v2.3 schema** after generation.

llama.cpp has publicly documented Gemma 4 grammar issues. Some reports concern E2B/E4B (different architecture), and others mention 26B. Therefore do not assume your specific build is affected—or safe—without testing.

## D. Running and measurement

Use the same 20 JDs, prompt, temperature, seed, output cap, same semantic validator and context sizing as the previous run, changing only the `response_format` forwarding. Make a separate run with `--prompt-mode targeted` only if the constrained baseline still has failures. Record per-JD accepted rate, true semantic checks, VRAM headroom, and total time per *accepted* rubric. Do not report schema-only success as an accepted JD.

Do not fix `JD-015413` by mutating model output. Resolve the `mustskills` / `goodskills` overlap in JD interpretation and validator policies first, with regression tests.

## E. Optional: source-aware cache identity

A stable JD identifier alone is not enough to reuse extraction. The package's `content_fingerprint()` hashes the JD text, prompt, schema, model, and generation settings. Before importing this behavior into the project's real batch-runner cache, confirm all existing cache invariants and storage migration behavior. The standalone benchmark logs its fingerprint but never modifies your production cache.
