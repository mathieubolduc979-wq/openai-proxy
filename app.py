# app.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import asyncio, json, os
from openai import AsyncOpenAI

app = FastAPI()

# --- OpenAI client ---
client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Limite de parallélisme (évite de heurter les quotas tout en restant rapide)
SEMAPHORE = asyncio.Semaphore(6)

def _clean_json_bytes(b: bytes) -> str:
    """
    Supprime BOM et espaces invisibles que Make peut ajouter (ZWSP, ZWNJ, ZWJ, etc.)
    puis trim. On parse ensuite ce string propre.
    """
    s = b.decode("utf-8", errors="ignore")
    # BOM + divers zero-width / word-joiner
    for ch in ("\ufeff", "\u200b", "\u200c", "\u200d", "\u2060"):
        s = s.replace(ch, "")
    return s.strip()

@app.get("/")
async def root():
    return {"ok": True, "hint": "POST /batch avec {'prompts': ['...','...']}"}

@app.post("/batch")
async def batch(req: Request):
    # 1) Lire le body BRUT, le nettoyer, parser JSON
    raw = await req.body()
    clean = _clean_json_bytes(raw)

    try:
        data = json.loads(clean)
    except json.JSONDecodeError as e:
        # Retourne clairement où ça casse
        raise HTTPException(status_code=422, detail=f"Invalid JSON: {e.msg} at pos {e.pos}")

    prompts = data.get("prompts")
    if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
        raise HTTPException(status_code=400, detail="`prompts` must be a list of strings.")

    # 2) Lancer les appels OpenAI en parallèle
    async def run_query(prompt: str, idx: int):
        try:
            async with SEMAPHORE:
                resp = await client.chat.completions.create(
                    model="gpt-5",          # rapide et dispo pour tous
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=250,
                    temperature=0.2,
                )
                text = resp.choices[0].message.content
                return {"index": idx, "output": text}
        except Exception as e:
            # On capture l'erreur pour ne pas faire planter tout le batch
            return {"index": idx, "error": str(e)}

    tasks = [run_query(p, i) for i, p in enumerate(prompts)]
    results = await asyncio.gather(*tasks)

    # (gather conserve déjà l'ordre, mais on re-trie par sécurité)
    results.sort(key=lambda x: x["index"])
    outputs = [r.get("output") for r in results]

    return JSONResponse({"responses": outputs, "details": results})
