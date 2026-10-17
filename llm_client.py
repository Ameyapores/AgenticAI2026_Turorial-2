"""Small provider adapter for the U of T Fall School labs.

Every notebook requires REAL API access. There is deliberately no hidden offline
fallback or paid call inside a repeated RL training loop.
"""
from __future__ import annotations
import os, json, re
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()
_REQUESTS = 0
MAX_REQUESTS = int(os.getenv('LAB_MAX_MODEL_CALLS', '12'))

@dataclass
class ModelConfig:
    provider: str
    model: str
    key: str
    base_url: str | None


def config():
    p = os.getenv('LAB_LLM_PROVIDER', '').strip().lower()
    model = os.getenv('LAB_LLM_MODEL', '').strip()
    if p not in ('anthropic', 'openai', 'openai_compatible'):
        raise RuntimeError('Set LAB_LLM_PROVIDER to anthropic, openai, or openai_compatible in your .env file.')
    if not model:
        raise RuntimeError('Set LAB_LLM_MODEL to the exact model name supplied by the school.')
    if p == 'anthropic':
        key = os.getenv('ANTHROPIC_API_KEY', '') or os.getenv('SCHOOL_LLM_API_KEY','')
        base = os.getenv('ANTHROPIC_BASE_URL', '') or None
    elif p == 'openai':
        key = os.getenv('OPENAI_API_KEY','') or os.getenv('SCHOOL_LLM_API_KEY','')
        base = os.getenv('OPENAI_BASE_URL','') or None
    else:
        key = os.getenv('SCHOOL_LLM_API_KEY','') or os.getenv('OPENAI_API_KEY','')
        base = os.getenv('SCHOOL_LLM_API_BASE','') or os.getenv('OPENAI_BASE_URL','')
        if not base:
            raise RuntimeError('OpenAI-compatible mode requires SCHOOL_LLM_API_BASE or OPENAI_BASE_URL.')
    if not key:
        raise RuntimeError('No API key. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or SCHOOL_LLM_API_KEY.')
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
        try:
            r=client.chat.completions.create(model=cfg.model, messages=messages,
                 max_tokens=max_tokens)
        except Exception as exc:
            msg=str(exc).lower()
            if 'max_tokens' not in msg and 'unsupported' not in msg:
                raise
            r=client.chat.completions.create(model=cfg.model, messages=messages,
                 max_completion_tokens=max_tokens)
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
