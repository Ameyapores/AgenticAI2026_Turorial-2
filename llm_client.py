"""Small provider adapter for the U of T Fall School labs.

Every notebook requires REAL API access. There is deliberately no hidden offline
fallback or paid call inside a repeated RL training loop.
"""
from __future__ import annotations
import os, json, re
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()
try:
    import api_config_local as _private_config
except ModuleNotFoundError:
    _private_config = None
try:
    import api_config as _template_config
except ModuleNotFoundError:
    _template_config = None


def _setting(name: str, *environment_names: str) -> str:
    """Read the private override, template, then legacy environment settings."""
    for config_module in (_private_config, _template_config):
        value = getattr(config_module, name, '') if config_module else ''
        if value and not str(value).startswith(('PASTE_', 'REPLACE_')):
            return str(value).strip()
    for environment_name in environment_names:
        value = os.getenv(environment_name, '')
        if value:
            return value.strip()
    return ''


_REQUESTS = 0
MAX_REQUESTS = int(_setting('MAX_MODEL_CALLS', 'LAB_MAX_MODEL_CALLS') or '12')

@dataclass
class ModelConfig:
    provider: str
    model: str
    key: str
    base_url: str | None


def config():
    p = _setting('PROVIDER', 'LAB_LLM_PROVIDER').lower()
    model = _setting('MODEL', 'LAB_LLM_MODEL')
    if p not in ('anthropic', 'openai', 'openai_compatible'):
        raise RuntimeError('Set PROVIDER in api_config.py to anthropic, openai, or openai_compatible.')
    if not model:
        raise RuntimeError('Set MODEL in api_config.py to the model name supplied by the school.')
    if p == 'anthropic':
        key = _setting('API_KEY', 'ANTHROPIC_API_KEY', 'SCHOOL_LLM_API_KEY')
        base = _setting('BASE_URL', 'ANTHROPIC_BASE_URL') or None
    elif p == 'openai':
        key = _setting('API_KEY', 'OPENAI_API_KEY', 'SCHOOL_LLM_API_KEY')
        base = _setting('BASE_URL', 'OPENAI_BASE_URL') or None
    else:
        key = _setting('API_KEY', 'SCHOOL_LLM_API_KEY', 'OPENAI_API_KEY')
        base = _setting('BASE_URL', 'SCHOOL_LLM_API_BASE', 'OPENAI_BASE_URL')
        if not base:
            raise RuntimeError('OpenAI-compatible mode requires BASE_URL in api_config.py.')
    if not key:
        raise RuntimeError('Set API_KEY in api_config.py to your provider key.')
    return ModelConfig(p, model, key, base)


def ask_text(prompt: str, *, max_tokens: int=800, system: str | None=None) -> str:
    global _REQUESTS
    cfg = config()
    if _REQUESTS >= MAX_REQUESTS:
        raise RuntimeError(f'Classroom safety cap reached: {MAX_REQUESTS} model calls. Increase LAB_MAX_MODEL_CALLS explicitly if needed.')
    _REQUESTS += 1
    system = system or ('You are assisting a fictional scientific research tutorial. '
                        'Use only evidence supplied by the user, distinguish predictions from '
                        'measurements, and do not invent replicated results.')
    if cfg.provider == 'anthropic':
        from anthropic import Anthropic
        kwargs = dict(api_key=cfg.key)
        if cfg.base_url: kwargs['base_url'] = cfg.base_url
        client = Anthropic(**kwargs)
        response = client.messages.create(model=cfg.model, max_tokens=max_tokens,
             system=system, messages=[{'role':'user','content':prompt}])
        result = '\n'.join(piece.text for piece in response.content if getattr(piece, 'type', '') == 'text')
    elif cfg.provider == 'openai':
        from openai import OpenAI
        kwargs = dict(api_key=cfg.key)
        if cfg.base_url: kwargs['base_url'] = cfg.base_url
        client = OpenAI(**kwargs)
        response = client.responses.create(model=cfg.model,
            instructions=system, input=prompt, max_output_tokens=max_tokens)
        result = response.output_text or ''
    else:
        from openai import OpenAI
        client = OpenAI(api_key=cfg.key, base_url=cfg.base_url)
        messages=[{'role':'system','content':system},{'role':'user','content':prompt}]
        request = dict(model=cfg.model, messages=messages, max_tokens=max_tokens)
        # Short action-selection requests need a visible JSON answer. On OpenRouter,
        # disabling hidden reasoning prevents the output allowance being exhausted
        # before the model emits that answer.
        if cfg.base_url and 'openrouter.ai' in cfg.base_url:
            request['reasoning_effort'] = 'none'
        try:
            r=client.chat.completions.create(**request)
        except Exception as exc:
            msg=str(exc).lower()
            if ('reasoning' in msg or 'unsupported' in msg) and 'reasoning_effort' in request:
                request.pop('reasoning_effort')
                r=client.chat.completions.create(**request)
            elif 'max_tokens' in msg:
                request.pop('max_tokens')
                request['max_completion_tokens'] = max_tokens
                r=client.chat.completions.create(**request)
            else:
                raise
        result=r.choices[0].message.content or ''
    if not result.strip():
        raise RuntimeError('The model returned no text. Check the model and token budget.')
    return result


def read_json(text: str) -> dict:
    text=text.strip()
    match = re.search(r'```(?:json)?\s*(.*?)```', text, flags=re.DOTALL | re.IGNORECASE)
    if match: text = match.group(1).strip()
    try:
        obj=json.loads(text)
    except json.JSONDecodeError:
        a=text.find('{'); b=text.rfind('}')
        if a == -1 or b == -1 or b < a: raise
        obj=json.loads(text[a:b+1])
    if not isinstance(obj,dict): raise ValueError('Expected a JSON object from the model.')
    return obj


def ask_json(prompt: str, *, max_tokens: int=1000) -> dict:
    text=ask_text(prompt + '\nReturn ONLY a JSON object. No markdown.', max_tokens=max_tokens)
    try:
        return read_json(text)
    except Exception:
        # One additional, explicit paid call to repair output. No silent placeholder data.
        fixed = ask_text('Convert the following text to one valid JSON object. Keep its values, '
              'do not add unsupported facts. Text:\n' + text, max_tokens=max_tokens)
        return read_json(fixed)


def connection_check():
    cfg=config()
    # A genuine call is required to confirm the provided credentials work.
    reply=ask_text('Reply with the single word READY.', max_tokens=400)
    print(f'Live model connection: {cfg.provider} / {cfg.model}')
    print('Model replied:', reply.strip()[:120])
    return cfg
