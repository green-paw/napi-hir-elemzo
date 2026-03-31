import time
import random
from typing import Any, Callable
from google.genai import types, Client
from token_logger import TokenLogger

import config

client = Client(api_key=config.GOOGLE_API_KEY)
logger = TokenLogger()

def execute_with_retry(func: Callable, *args, max_retries: int = 5, **kwargs) -> Any:
    """Univerzális retry logika."""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err = str(e).lower()
            if any(x in err for x in ["429", "500", "503", "quota"]) and attempt < max_retries - 1:
                wait = ((2 ** attempt) * 5) + random.uniform(0, 3)
                time.sleep(wait)
            else:
                raise e

def generate(contents: Any, sys_instr: str, schema: Any = None, model: str = config.MODEL_LITE_ID) -> Any:
    final_contents = contents
    final_sys_instr = sys_instr

    gen_config = types.GenerateContentConfig(
        system_instruction=final_sys_instr,
        cached_content=None,
        response_mime_type="application/json" if schema else "text/plain",
        response_schema=schema,
        temperature=0.1,
    )
    
    # Itt már a final_contents-t küldjük be
    response = execute_with_retry(
        client.models.generate_content, 
        model=model, 
        contents=final_contents, 
        config=gen_config
    )
    
    logger.add(model, response)

    try:
        if response is not None:
            input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
            cached_tokens = getattr(response.usage_metadata, 'cached_content_token_count', 0) or 0
            output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
            finish_reason = response.candidates[0].finish_reason if hasattr(response, 'candidates') and response.candidates else 'N/A'
            print(f"📊 {model} | Input tokens: {input_tokens} | Output tokens: {output_tokens} | Cached tokens: {cached_tokens} | Finish reason: {finish_reason}")
    except:
        pass
    
    try:
        if schema:
            if schema != "json": 
                return response.parsed
            return response.text # amikor json-t kérünk tőle de nem akarjuk pydantic-ba kényszeríteni
        return response.text
    except Exception as e:
        print(f"⚠️ Válasz feldolgozási hiba: {e}")
        return response.text if hasattr(response, 'text') else response

def embed(texts: list) -> Any:
    return execute_with_retry(client.models.embed_content, model="gemini-embedding-001", contents=texts, config=types.EmbedContentConfig(task_type="CLUSTERING"))

