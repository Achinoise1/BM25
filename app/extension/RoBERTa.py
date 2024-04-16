import os
import sys
import json
import time

start = time.time()

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline


def extract_answer_from_context(question="", context=""):
    model = AutoModelForQuestionAnswering.from_pretrained(
        "uer/roberta-base-chinese-extractive-qa"
    )
    tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-chinese-extractive-qa")
    QA = pipeline("question-answering", model=model, tokenizer=tokenizer)

    # QA_input = {
    #     "question": "著名诗歌《假如生活欺骗了你》的作者是",
    #     "context": "普希金从那里学习人民的语言，吸取了许多有益的养料，这一切对普希金后来的创作产生了很大的影响。这两年里，普希金创作了不少优秀的作品，如《囚徒》、《致大海》、《致凯恩》和《假如生活欺骗了你》等几十首抒情诗，叙事诗《努林伯爵》，历史剧《鲍里斯·戈都诺夫》，以及《叶甫盖尼·奥涅金》前六章。",
    # }

    QA_input = {"question": question, "context": context}

    return QA(QA_input)


# if __name__ == "__main__":
#     # ans = extract_answer_from_context(sys.argv[1], sys.argv[2])
#     ans = extract_answer_from_context(sys.argv[1], sys.argv[2])
#     print(json.dumps(ans))
#     # end = time.time()
#     # print(end - start)
