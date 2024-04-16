import json
import math
import os
import jieba
import pickle


class BM25Param(object):
    def __init__(
        self,
        f,
        df,
        idf,
        length,
        avg_length,
        docs_list,
        line_length_list,
        k1=1.5,
        k2=1.0,
        b=0.75,
    ):
        """

        :param f:
        :param df:
        :param idf:
        :param length:
        :param avg_length:
        :param docs_list:
        :param line_length_list:
        :param k1: 可调整参数，[1.2, 2.0]
        :param k2: 可调整参数，[1.2, 2.0]
        :param b:
        """
        self.f = f
        self.df = df
        self.k1 = k1
        self.k2 = k2
        self.b = b
        self.idf = idf
        self.length = length
        self.avg_length = avg_length
        self.docs_list = docs_list
        self.line_length_list = line_length_list

    def __str__(self):
        return f"k1:{self.k1}, k2:{self.k2}, b:{self.b}"


# 预处理，处理的内容存到文件中，后续直接读取文件过算法
# def content_preprocess(
#     input_file="E:/0Projs/BM25/extension/wikipedia-cn-20230720-filtered.json",
# ):
#     with open(input_file, "r", encoding="utf-8") as f:
#         content_return = []
#         content_whole = json.loads(f.read())
#         for cw in content_whole:
#             content_return.append(cw["completion"])
#         with open("preprocessed.json", "w", encoding="utf-8") as writer:
#             writer.write(json.dumps(content_return))
#         return content_return


def read_preprocessed_data(file="app/extension/preprocessed.json"):
    with open(file, "r", encoding="utf-8") as reader:
        content_return = json.loads(reader.read())
        return content_return


class BM25(object):
    _param_pkl = "E:/0Projs/BM25/app/data/param.pkl"
    _stop_words_path = "E:/0Projs/BM25/app/data/stop_words.txt"
    _stop_words = []

    def __init__(self):
        self.param: BM25Param = self._load_param()

    def _load_stop_words(self):
        if not os.path.exists(self._stop_words_path):
            raise Exception(f"system stop words: {self._stop_words_path} not found")
        stop_words = []
        with open(self._stop_words_path, "r", encoding="utf8") as reader:
            for line in reader:
                line = line.strip()
                stop_words.append(line)
        return stop_words

    def _build_param(self):

        def _cal_param():
            f = []  # 列表的每一个元素是一个dict，dict存储着一个文档中每个词的出现次数
            df = {}  # 存储每个词及出现了该词的文档数量
            idf = {}  # 存储每个词的idf值
            lines = self.all_text
            length = len(lines)
            words_count = 0
            docs_list = []
            line_length_list = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                words = [
                    word
                    for word in jieba.lcut(line)
                    if word and word not in self._stop_words
                ]
                line_length_list.append(len(words))
                docs_list.append(line)
                words_count += len(words)
                tmp_dict = {}
                for word in words:
                    tmp_dict[word] = tmp_dict.get(word, 0) + 1
                f.append(tmp_dict)
                for word in tmp_dict.keys():
                    df[word] = df.get(word, 0) + 1
            for word, num in df.items():
                idf[word] = math.log(length - num + 0.5) - math.log(num + 0.5)
            param = BM25Param(
                f, df, idf, length, words_count / length, docs_list, line_length_list
            )
            return param

        param = _cal_param()

        with open(self._param_pkl, "wb") as writer:
            pickle.dump(param, writer)
        return param

    def _load_param(self):
        self._stop_words = self._load_stop_words()

        if not os.path.exists(self._param_pkl):
            param = self._build_param()
        else:
            with open(self._param_pkl, "rb") as reader:
                param = pickle.load(reader)
        return param

    def _cal_similarity(self, words, index):
        score = 0
        for word in words:
            if word not in self.param.f[index]:
                continue
            molecular = (
                self.param.idf[word] * self.param.f[index][word] * (self.param.k1 + 1)
            )
            denominator = self.param.f[index][word] + self.param.k1 * (
                1
                - self.param.b
                + self.param.b
                * self.param.line_length_list[index]
                / self.param.avg_length
            )
            score += molecular / denominator
        return score

    def cal_similarity(self, query: str):
        """
        相似度计算，无排序结果
        :param query: 待查询结果
        :return: [(doc, score), ..]
        """
        words = [
            word for word in jieba.lcut(query) if word and word not in self._stop_words
        ]
        score_list = []
        for index in range(self.param.length):
            score = self._cal_similarity(words, index)
            score_list.append((self.param.docs_list[index], score))
        return score_list

    def cal_similarity_rank(self, query: str):
        """
        相似度计算，排序
        :param query: 待查询结果
        :return: [(doc, score), ..]
        """
        result = self.cal_similarity(query)
        result.sort(key=lambda x: -x[1])
        return result


def retrieve_context(query_content):
    """
        [('自然语言处理并不是一般地研究自然语言，', 3.2072055608059036), ('因此，这一领域的研究将涉及自然语言，即人们日常使用的语言，', 2.2068046420905443), ('它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方
    法。', 2.0911221271793545), ('所以它与语言学的研究有着密切的联系，但又有重要的区别。', 1.4616618736274032), ('而在于研制能有效地实现自然语言通信的计算机系统，', 1.201522188129132), ('自然语言处理是计算机科学领域与
    人工智能领域中的一个重要方向。', 1.012567843290477), ('自然语言处理是一门融语言学、计算机科学、数学于一体的科学。', 1.012567843290477), ('特别是其中的软件系统。因而它是计算机科学的一部分。', 0), ('在信息搜索中，我
    们做的第一步就是检索。', 0), ('再延展一下，搜索这项功能在我们生活中也是太多太多。', 0), ('大众一点就是搜索
    引擎，商品搜索等，在问题系统中可以匹配相似的问题，然后返回对应答案等。', 0), ('文本匹配包括监督学习方法以及非监督学习方法。', 0), ('或者分为传统方法和深度学习方法。', 0), ('BM25 在 20 世纪 70 年代到 80 年代被提出，到目前为止已经过去二三十年了，但是这个算法依然在很多信息检索的任务中表现优异，是很多工程师首选的算法之一。', 0), ('有时候全称是 Okapi BM25，这里的“BM”是“最佳匹配”（Best Match）的简称。', 0), ('那么，当通过使用不同
    的语素分析方法，语素权重判定方法以及语素与文档的相关性判定方法，可以衍生很多不同的搜索相关性计算方法，灵活
    性也比较大。', 0)]
    """
    bm25 = BM25()
    result = bm25.cal_similarity_rank(query_content)[0]
    dict_list = {"text": result[0], "score": result[1]}
    ans = dict_list.get("text").replace("\n", "")
    return {"text": ans, "score": dict_list.get("score")}
