from pathlib import Path
from typing import Dict, List, Tuple
import jieba.posseg as pseg
import lancedb
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import LanceDB
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field


class VectorDatabase:
    def __init__(self):
        self.embedding = DashScopeEmbeddings(
            model="text-embedding-v3",
            dashscope_api_key="sk-b2817c33fdd64d3189582e100e1c0617",
        )

        db_path = Path(__file__).resolve().parents[1] / "lance_db"
        self.vectordb = LanceDB(
            embedding=self.embedding,
            connection=lancedb.connect(str(db_path)),
            table_name="lance",
        )

        self.allowed_pos = {"n", "v", "a", "vn", "an", "nz", "ns", "nt", "nr"}

    def _chinese_preprocessor(self, text: str) -> List[str]:
        result: List[str] = []
        for word, flag in pseg.cut(text):
            if flag in self.allowed_pos:
                result.append(word)
        return result

    @staticmethod
    def _doc_key(doc: Document) -> Tuple[str, Tuple[Tuple[str, object], ...]]:
        metadata_items = tuple(sorted(doc.metadata.items()))
        return doc.page_content, metadata_items

    def _merge_ranked_results(self, ranked_sources: List[Tuple[List[Document], float]], k: int) -> List[Document]:
        # 融合算法 权限越大 -> 分数越高 排名越前 -> 分数越高
        # score(doc) = sum(weight_i / rank_i(doc))
        k = max(1, k)

        score_dict: Dict[Tuple[str, Tuple[Tuple[str, object], ...]], float] = {}
        results_dict: Dict[Tuple[str, Tuple[Tuple[str, object], ...]], Document] = {}

        for results, weight in ranked_sources:
            if not results or weight <= 0:
                continue
            for rank, result in enumerate(results, start=1):
                key = self._doc_key(result)
                current_score = score_dict.get(key, 0.0)
                wrrf_score = weight / rank
                score_dict[key] = current_score + wrrf_score
                results_dict.setdefault(key, result)

        ordered_keys = sorted(score_dict, key=score_dict.get, reverse=True)

        return [results_dict[key] for key in ordered_keys[:k]]

    def fetch_all_documents(self) -> List[Document]:
        # 遍历
        table = self.vectordb.get_table()
        if table is None:
            return []

        return self.vectordb.results_to_docs(table.to_arrow()) or []

    def add_records(self, records: List[Dict[str, str]], text_key: str = "text") -> List[str]:
        # 新增
        if not records:
            return []

        texts: List[str] = []
        metadatas: List[Dict[str, str]] = []

        for record in records:
            text = record.get(text_key)
            if not text:
                continue
            texts.append(text)
            metadatas.append({k: v for k, v in record.items() if k != text_key})

        if not texts:
            return []

        return self.vectordb.add_texts(texts=texts, metadatas=metadatas)

    def staged_search(self, query: str, k1: int, k2: int, k1_weight: float, k2_weight: float) -> List[Document]:
        # 融合查询
        if not query:
            return []

        db_documents = self.fetch_all_documents()
        if not db_documents:
            return []

        k1_retriever = self.vectordb.as_retriever(search_kwargs={"k": k1})
        k2_retriever = BM25Retriever.from_documents(
            documents=db_documents,
            preprocess_func=self._chinese_preprocessor,
            k=k1,
        )

        k1_results = k1_retriever.invoke(query)
        k2_results = k2_retriever.invoke(query)

        return self._merge_ranked_results(
            ranked_sources=[(k1_results, k1_weight), (k2_results, k2_weight)],
            k=k2,
        )


class CustomStagedRetriever(BaseRetriever):
    vector_db: VectorDatabase = Field(default_factory=VectorDatabase)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return self.vector_db.staged_search(query=query, k1=4, k1_weight=0.3, k2=2, k2_weight=0.7)


# if __name__ == "__main__":
#     vd = VectorDatabase()
#     sample_records = [
#         # 订单相关
#         {"text": "查询订单明细，需要提供订单号"},
#         # 公司信息相关
#         {"text": "查询公司信息，请提供公司全称或统一社会信用代码"},
#     ]
#     vd.add_records(records=sample_records)
#     print(vd.fetch_all_documents())
#     print(vd.staged_search(query="我怎么查订单", k1=4, k1_weight=0.3, k2=2, k2_weight=0.7))
