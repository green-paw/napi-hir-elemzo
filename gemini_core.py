import time
import random
from typing import Any, Callable, List, Dict, Optional, Union
from google.genai import types, Client
from google import genai
import os

import config
from token_logger import TokenLogger
from concurrent.futures import ThreadPoolExecutor

def get_gemini_client() -> genai.Client:
    """Inicializálja a Vertex AI klienst."""
    project_id: str = os.getenv("GCP_PROJECT_ID", "your-project-id")
    location: str = "us-central1"

    return genai.Client(
        vertexai=True,
        project=project_id,
        location=location
    )
client = get_gemini_client()

logger = TokenLogger()

def execute_with_retry(func: Callable, *args, max_retries: int = 5, **kwargs) -> Any:
    """Univerzális retry logika exponential backoff-fal."""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err = str(e).lower()
            if any(x in err for x in ["429", "500", "503", "quota"]) and attempt < max_retries - 1:
                wait = ((2 ** attempt) * 5) + random.uniform(0, 3)
                print(f"🔄 API hiba ({err[:40]}...). Újrapróbálkozás {wait:.1f} mp múlva (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise e

def generate(
    contents: Any, 
    sys_instr: str, 
    schema: Any = None, 
    model: str = config.MODEL_LITE_ID,
    max_output_tokens: int = 1024
) -> Any:
    """Szöveg vagy strukturált adat generálása az LLM-mel."""
    
    # A Pydantic sémák és a nyers JSON megkülönböztetése
    response_mime_type = "text/plain"
    response_schema = None
    
    if schema:
        response_mime_type = "application/json"
        if schema != "json":
            response_schema = schema

    gen_config = types.GenerateContentConfig(
        system_instruction=sys_instr,
        response_mime_type=response_mime_type,
        response_schema=response_schema,
        temperature=0.0,
        max_output_tokens=max_output_tokens
    )
    
    response = execute_with_retry(
        client.models.generate_content, 
        model=model, 
        contents=contents, 
        config=gen_config
    )
    
    logger.add(model, response)

    # Logging a konzolra (ahogy te írtad, nagyon hasznos debuggoláshoz)
    try:
        if response is not None and hasattr(response, 'usage_metadata'):
            logger.add(model, response)

            usage = response.usage_metadata
            input_tokens = getattr(usage, 'prompt_token_count', 0) or 0
            cached_tokens = getattr(usage, 'cached_content_token_count', 0) or 0
            output_tokens = getattr(usage, 'candidates_token_count', 0) or 0
            
            finish_reason = 'N/A'
            if hasattr(response, 'candidates') and response.candidates:
                finish_reason = response.candidates[0].finish_reason
                
            print(f"📊 {model} | In: {input_tokens} | Out: {output_tokens} | Cache: {cached_tokens} | Reason: {finish_reason}")
    except Exception as e:
        print(f"⚠️ Logger hiba a konzolos kiírásnál: {e}")
    
    # Biztonságos visszatérési érték feldolgozás
    try:
        if schema and schema != "json" and hasattr(response, 'parsed') and response.parsed:
            return response.parsed
        return response.text
    except Exception as e:
        print(f"⚠️ Válasz feldolgozási hiba: {e}")
        return response.text if hasattr(response, 'text') else response

def embed(texts: List[str], task_type: str = "CLUSTERING") -> List[List[float]]:
    if not texts:
        return []

    batch_size: int = 100
    # Felosztjuk a szövegeket batch-ekre
    batches: List[List[str]] = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    
    def process_batch(batch: List[str]) -> List[List[float]]:
        """Egyetlen batch feldolgozása egy szálon."""
        try:
            response = execute_with_retry(
                client.models.embed_content, 
                model="gemini-embedding-001", 
                contents=batch, 
                config=types.EmbedContentConfig(task_type=task_type)
            )
            
            if hasattr(response, 'embeddings'):
                # Visszaadjuk a float listák listáját
                return [emb.values for emb in response.embeddings]
            return []
        except Exception as e:
            print(f"⚠️ Hiba az embedding batch feldolgozásakor: {e}")
            return []

    all_embeddings: List[List[float]] = []
    
    # Max workers: érdemes korlátozni, hogy ne fussunk bele azonnal Quota limitbe (pl. 5 szál)
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Az executor.map megőrzi a beküldött sorrendet a válaszoknál
        results = executor.map(process_batch, batches)
        
        for result_list in results:
            all_embeddings.extend(result_list)
            # Mivel a szálak párhuzamosan futnak, a belső sleep-et kivettem, 
            # az execute_with_retry-nek kell kezelnie a sebességkorlátot.

    return all_embeddings