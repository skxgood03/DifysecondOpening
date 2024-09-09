import threading
from typing import Optional

from flask import Flask, current_app

from core.rag.data_post_processor.data_post_processor import DataPostProcessor
from core.rag.datasource.keyword.keyword_factory import Keyword
from core.rag.datasource.vdb.vector_factory import Vector
from core.rag.rerank.constants.rerank_mode import RerankMode
from core.rag.retrieval.retrieval_methods import RetrievalMethod
from extensions.ext_database import db
from models.dataset import Dataset

default_retrieval_model = {
    'search_method': RetrievalMethod.SEMANTIC_SEARCH.value,
    'reranking_enable': False,
    'reranking_model': {
        'reranking_provider_name': '',
        'reranking_model_name': ''
    },
    'top_k': 2,
    'score_threshold_enabled': False
}


class RetrievalService:

    @classmethod
    def retrieve(cls, retrieval_method: str, dataset_id: str, query: str,
                 top_k: int, score_threshold: Optional[float] = .0,
                 reranking_model: Optional[dict] = None, reranking_mode: Optional[str] = 'reranking_model',
                 weights: Optional[dict] = None):
        """
        根据指定的检索方法执行检索操作。

        :param retrival_method: 检索方法类型。 semantic_search full_text_search keyword_search,hybrid_search
        :param dataset_id: 数据集ID。
        :param query: 检索查询字符串。
        :param top_k: 返回结果的最大数量。
        :param score_threshold: 分数阈值，用于过滤结果。
        :param reranking_model: 重排序模型配置，用于对结果进行二次排序。
        :return: 检索结果列表。
        """
        dataset = db.session.query(Dataset).filter(
            Dataset.id == dataset_id
        ).first()
        if not dataset or dataset.available_document_count == 0 or dataset.available_segment_count == 0:
            return []
        all_documents = []
        threads = []
        exceptions = []
        # retrieval_model source with keyword
        if retrieval_method == 'keyword_search':
            keyword_thread = threading.Thread(target=RetrievalService.keyword_search, kwargs={
                'flask_app': current_app._get_current_object(),
                'dataset_id': dataset_id,
                'query': query,
                'top_k': top_k,
                'all_documents': all_documents,
                'exceptions': exceptions,
            })
            threads.append(keyword_thread)
            keyword_thread.start()
        # retrieval_model source with semantic
        if RetrievalMethod.is_support_semantic_search(retrieval_method):
            embedding_thread = threading.Thread(target=RetrievalService.embedding_search, kwargs={
                'flask_app': current_app._get_current_object(),
                'dataset_id': dataset_id,
                'query': query,
                'top_k': top_k,
                'score_threshold': score_threshold,
                'reranking_model': reranking_model,
                'all_documents': all_documents,
                'retrieval_method': retrieval_method,
                'exceptions': exceptions,
            })
            threads.append(embedding_thread)
            embedding_thread.start()

        # retrieval source with full text
        if RetrievalMethod.is_support_fulltext_search(retrieval_method):
            full_text_index_thread = threading.Thread(target=RetrievalService.full_text_index_search, kwargs={
                'flask_app': current_app._get_current_object(),
                'dataset_id': dataset_id,
                'query': query,
                'retrieval_method': retrieval_method,
                'score_threshold': score_threshold,
                'top_k': top_k,
                'reranking_model': reranking_model,
                'all_documents': all_documents,
                'exceptions': exceptions,
            })
            threads.append(full_text_index_thread)
            full_text_index_thread.start()
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        # 如果执行过程中有异常，则合并异常信息并抛出
        if exceptions:
            exception_message = ';\n'.join(exceptions)
            raise Exception(exception_message)

        if retrieval_method == RetrievalMethod.HYBRID_SEARCH.value:
            data_post_processor = DataPostProcessor(str(dataset.tenant_id), reranking_mode,
                                                    reranking_model, weights, False)
            all_documents = data_post_processor.invoke(
                query=query,
                documents=all_documents,
                score_threshold=score_threshold,
                top_n=top_k
            )
            # print("混合搜索",all_documents)
        return all_documents

    @classmethod
    def keyword_search(cls, flask_app: Flask, dataset_id: str, query: str,
                       top_k: int, all_documents: list, exceptions: list):
        with flask_app.app_context():
            try:
                dataset = db.session.query(Dataset).filter(
                    Dataset.id == dataset_id
                ).first()

                keyword = Keyword(
                    dataset=dataset
                )

                documents = keyword.search(
                    cls.escape_query_for_search(query),
                    top_k=top_k
                )
                all_documents.extend(documents)
            except Exception as e:
                exceptions.append(str(e))

    @classmethod
    def embedding_search(cls, flask_app: Flask, dataset_id: str, query: str,
                         top_k: int, score_threshold: Optional[float], reranking_model: Optional[dict],
                         all_documents: list, retrieval_method: str, exceptions: list):
        with flask_app.app_context():
            try:
                # 从数据库中查询指定ID的数据集
                dataset = db.session.query(Dataset).filter(
                    Dataset.id == dataset_id
                ).first()
                # 初始化向量检索器
                vector = Vector(
                    dataset=dataset
                )
                # 执行向量搜索，根据相似度分数阈值筛选结果
                documents = vector.search_by_vector(
                    cls.escape_query_for_search(query),
                    search_type='similarity_score_threshold',
                    top_k=top_k,
                    score_threshold=score_threshold,
                    filter={
                        'group_id': [dataset.id]
                    }
                )
                # 如果有搜索结果
                if documents:
                    if reranking_model and reranking_model.get('reranking_model_name') and reranking_model.get('reranking_provider_name') and retrieval_method == RetrievalMethod.SEMANTIC_SEARCH.value:
                        data_post_processor = DataPostProcessor(str(dataset.tenant_id),
                                                                RerankMode.RERANKING_MODEL.value,
                                                                reranking_model, None, False)
                        # 调用数据后处理器对结果进行重排序
                        all_documents.extend(data_post_processor.invoke(
                            query=query,
                            documents=documents,
                            score_threshold=score_threshold,
                            top_n=len(documents)
                        ))
                    else:
                        # 否则直接将结果添加到总结果列表中
                        all_documents.extend(documents)
            except Exception as e:
                # 如果执行过程中出现异常，则记录异常信息
                exceptions.append(str(e))

    @classmethod
    def full_text_index_search(cls, flask_app: Flask, dataset_id: str, query: str,
                               top_k: int, score_threshold: Optional[float], reranking_model: Optional[dict],
                               all_documents: list, retrieval_method: str, exceptions: list):
        with flask_app.app_context():
            try:
                dataset = db.session.query(Dataset).filter(
                    Dataset.id == dataset_id
                ).first()

                vector_processor = Vector(
                    dataset=dataset,
                )

                documents = vector_processor.search_by_full_text(
                    cls.escape_query_for_search(query),
                    top_k=top_k
                )
                if documents:
                    if reranking_model and reranking_model.get('reranking_model_name') and reranking_model.get('reranking_provider_name') and retrieval_method == RetrievalMethod.FULL_TEXT_SEARCH.value:
                        data_post_processor = DataPostProcessor(str(dataset.tenant_id),
                                                                RerankMode.RERANKING_MODEL.value,
                                                                reranking_model, None, False)
                        all_documents.extend(data_post_processor.invoke(
                            query=query,
                            documents=documents,
                            score_threshold=score_threshold,
                            top_n=len(documents)
                        ))
                    else:
                        all_documents.extend(documents)
            except Exception as e:
                exceptions.append(str(e))

    @staticmethod
    def escape_query_for_search(query: str) -> str:
        return query.replace('"', '\\"')