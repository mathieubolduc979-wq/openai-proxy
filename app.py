from fastapi import FastAPI, Request
import asyncio
from openai import AsyncOpenAI
import os

app = FastAPI()
client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

@app.post("/batch")
async def batch(request: Request):
    data = await request.json()
    prompts = data["prompts"]

    async def run_query(prompt):
        resp = await client.chat.completions.create(
            model="gpt-4.1",  # change en "gpt-5" si dispo
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
        )
        return resp.choices[0].message.content

    results = await asyncio.gather(*(run_query(p) for p in prompts))
    return {"responses": results}
