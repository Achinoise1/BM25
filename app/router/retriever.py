from fastapi import APIRouter
from pydantic import BaseModel
from extension.bm25_algo import retrieve_context
from extension.RoBERTa import extract_answer_from_context

router = APIRouter()


class Query(BaseModel):
    data: str


@router.post("/retrieve")
async def retrieve_data(query: Query):
    dict_list = retrieve_context(query.data)
    ans = extract_answer_from_context(query.data, dict_list.get("text"))
    return {"code": 200, "msg": "OK", "data": ans}
