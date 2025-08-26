from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
from openai import AsyncOpenAI
import os

app = FastAPI()
client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Définir un schéma Pydantic pour le body
class BatchRequest(BaseModel):
    prompts: list[str]

@app.post("/batch")
async def batch(request: BatchRequest):
    async def run_query(prompt: str):
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",  # modèle stable
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
        )
        return resp.choices[0].message.content

    results = await asyncio.gather(*(run_query(p) for p in request.prompts))
    return {"responses": results}
