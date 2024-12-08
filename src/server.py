from fastapi import FastAPI, Query
from .agent import create_agent

app = FastAPI()
agent = create_agent()


# def stream_to_string(stream):
#     res = []
#     for s in stream:
#         message = s["messages"][-1]
#         if isinstance(message, tuple):
#             res.append(str(message))
#         else:
#             res.append(message.pretty())
#     return "\n".join(res)
#

@app.get("/search/")
async def search_articles(query: str = Query(..., description="Ключевые слова для поиска статей")):
    """
    Поиск статей с помощью ReAct-агента.
    """
    inputs = {"messages": [("user", f"Find articles matching this query: '{query}'")]}
    res = agent.invoke(inputs, stream_mode="values")
    return {"response": res}
