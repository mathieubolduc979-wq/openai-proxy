from fastapi import FastAPI, Request
import asyncio
from openai import AsyncOpenAI
import os

app = FastAPI()
client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

@app.post("/batch")
async def batch(request: Request):
    # Log le raw body reçu
    raw_body = await request.body()
    print("==== RAW BODY ====")
    print(raw_body.decode("utf-8", errors="ignore"))
    print("==================")

    # Essaie de parser en JSON
    data = await request.json()
    prompts = data["prompts"]

    async def run_query(prompt: str):
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
        )
        return resp.choices[0].message.content

    results = await asyncio.gather(*(run_query(p) for p in prompts))
    return {"responses": results}
