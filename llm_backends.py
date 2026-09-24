from __future__ import annotations

import os
from functools import lru_cache

import torch
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import build_feedback_prompt

load_dotenv()

LOCAL_DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LOCAL_MODEL_OPTIONS = [
    LOCAL_DEFAULT_MODEL,
    "Qwen/Qwen2.5-3B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
]
API_DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


@lru_cache(maxsize=2)
def load_local_model(model_name: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float16

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=True,
            **model_kwargs,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


def generate_local_answer(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_new_tokens: int,
) -> str:
    tokenizer, model = load_local_model(model_name)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    model_inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {key: value.to(model.device) for key, value in model_inputs.items()}

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature

    with torch.no_grad():
        outputs = model.generate(**model_inputs, **generation_kwargs)

    prompt_length = model_inputs["input_ids"].shape[-1]
    generated_tokens = outputs[0][prompt_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def generate_openai_answer(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_new_tokens: int,
) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model_name,
        store=False,
        temperature=temperature,
        max_output_tokens=max_new_tokens,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.output_text.strip()


def answer_question(
    backend: str,
    model_name: str,
    dataset_label: str,
    question: str,
    retrieved_rows: list[dict],
    temperature: float,
    max_new_tokens: int,
) -> tuple[str, str]:
    system_prompt, user_prompt = build_feedback_prompt(
        dataset_label, question, retrieved_rows
    )
    prompt_text = f"[system]\n{system_prompt}\n\n[user]\n{user_prompt}"

    if backend == "Local Hugging Face":
        answer = generate_local_answer(
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
    else:
        answer = generate_openai_answer(
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    return answer, prompt_text
