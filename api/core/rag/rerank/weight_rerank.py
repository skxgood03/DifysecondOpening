import math
from collections import Counter
from typing import Optional

import numpy as np

from core.embedding.cached_embedding import CacheEmbedding
from core.model_manager import ModelManager
from core.model_runtime.entities.model_entities import ModelType
from core.rag.datasource.keyword.jieba.jieba_keyword_table_handler import JiebaKeywordTableHandler
from core.rag.models.document import Document
from core.rag.rerank.entity.weight import VectorSetting, Weights


class WeightRerankRunner:
    """
    这个类实现了基于权重的重排序功能，用于对搜索结果中的文档进行重新排序，以提高相关性。
    """

    def __init__(self, tenant_id: str, weights: Weights) -> None:
        self.tenant_id = tenant_id #租户ID。
        self.weights = weights #权重配置

    def run(self, query: str, documents: list[Document], score_threshold: Optional[float] = None,
            top_n: Optional[int] = None, user: Optional[str] = None) -> list[Document]:
        """
       执行基于权重的重排序

        :param query:  搜索查询
        :param documents:  文档列表
        :param score_threshold:  分数阈值
        :param top_n: 返回的文档数量上限
        :param user: unique user id if needed

        :return:  重排序后的文档列表
        """
        docs = [] # 用于存储文档内容
        doc_id = []  # 用于存储文档ID
        unique_documents = []  # 用于存储去重后的文档
        for document in documents:
            if document.metadata['doc_id'] not in doc_id:
                doc_id.append(document.metadata['doc_id'])  # 添加文档ID
                docs.append(document.page_content) # 添加文档内容
                unique_documents.append(document) # 添加文档

        documents = unique_documents  # 更新文档列表为去重后的列表

        rerank_documents = []   # 用于存储重排序后的文档
        # 计算关键词
        query_scores = self._calculate_keyword_score(query, documents)
        # 计算向量
        query_vector_scores = self._calculate_cosine(self.tenant_id, query, documents, self.weights.vector_setting)
        # 合并
        for document, query_score, query_vector_score in zip(documents, query_scores, query_vector_scores):
            # 计算文档的最终
            score = self.weights.vector_setting.vector_weight * query_vector_score + \
                    self.weights.keyword_setting.keyword_weight * query_score
            # 应用分数阈值
            if score_threshold and score < score_threshold:
                continue
            # 更新文档元数据中的得分
            document.metadata['score'] = score
            # 添加文档到重排序列表
            rerank_documents.append(document)
        # 排序文档
        rerank_documents = sorted(rerank_documents, key=lambda x: x.metadata['score'], reverse=True)
        # 返回指定数量的文档
        return rerank_documents[:top_n] if top_n else rerank_documents

    def _calculate_keyword_score(self, query: str, documents: list[Document]) -> list[float]:
        """
          计算文档与查询之间的余弦相似度得分。
        :param query: search query
        :param documents: documents for reranking

        :return:   每个文档的BM25得分列表
        """
        keyword_table_handler = JiebaKeywordTableHandler() # 创建关键词处理实例
        query_keywords = keyword_table_handler.extract_keywords(query, None) # 提取查询关键词
        documents_keywords = [] # 用于存储文档关键词
        for document in documents:
            # # 提取文档关键词
            document_keywords = keyword_table_handler.extract_keywords(document.page_content, None)
            document.metadata['keywords'] = document_keywords  # 将关键词存入文档元数据
            documents_keywords.append(document_keywords)  # 添加文档关键词到列表

        # 统计查询关键词的词频（TF）
        query_keyword_counts = Counter(query_keywords)
        # 总文档数
        total_documents = len(documents)

        #  计算所有文档关键词的逆文档频率（IDF）
        all_keywords = set()  # 用于存储所有文档中的关键词
        for document_keywords in documents_keywords:
            all_keywords.update(document_keywords) # 更新所有关键词集合

        keyword_idf = {}   # 用于存储关键词的IDF值
        for keyword in all_keywords:
            # 计算包含特定关键词的文档数量
            doc_count_containing_keyword = sum(1 for doc_keywords in documents_keywords if keyword in doc_keywords)
            # 计算IDF
            keyword_idf[keyword] = math.log((1 + total_documents) / (1 + doc_count_containing_keyword)) + 1

        query_tfidf = {} # 用于存储查询关键词的TF-IDF值

        for keyword, count in query_keyword_counts.items():
            tf = count  # 查询关键词的词频
            idf = keyword_idf.get(keyword, 0)  # 查询关键词的IDF
            query_tfidf[keyword] = tf * idf # 计算查询关键词的TF-IDF值

        # 计算所有文档的TF-IDF值
        documents_tfidf = []
        for document_keywords in documents_keywords:
            document_keyword_counts = Counter(document_keywords) # 统计文档关键词的词频
            document_tfidf = {} # 用于存储文档关键词的TF-IDF值
            for keyword, count in document_keyword_counts.items():
                tf = count # 文档关键词的词频
                idf = keyword_idf.get(keyword, 0) # 文档关键词的IDF
                document_tfidf[keyword] = tf * idf  # 计算文档关键词的TF-IDF值
            documents_tfidf.append(document_tfidf) # 添加文档TF-IDF值到列表

        # 定义余弦相似度计算函数
        def cosine_similarity(vec1, vec2):
            # 计算两个向量的交集
            intersection = set(vec1.keys()) & set(vec2.keys())
            # 计算分子
            numerator = sum(vec1[x] * vec2[x] for x in intersection)
            # 计算分母
            sum1 = sum(vec1[x] ** 2 for x in vec1.keys())
            sum2 = sum(vec2[x] ** 2 for x in vec2.keys())
            denominator = math.sqrt(sum1) * math.sqrt(sum2)
            # 避免除以零的情况
            if not denominator:
                return 0.0
            else:
                return float(numerator) / denominator

        # 初始化用于存储文档相似度得分的列表
        similarities = []
        # 计算每个文档与查询之间的相似度得分
        for document_tfidf in documents_tfidf:
            similarity = cosine_similarity(query_tfidf, document_tfidf)
            similarities.append(similarity)

        # for idx, similarity in enumerate(similarities):
        #     print(f"Document {idx + 1} similarity: {similarity}")
        # 返回相似度得分列表
        return similarities

    def _calculate_cosine(self, tenant_id: str, query: str, documents: list[Document],
                          vector_setting: VectorSetting) -> list[float]:
        """
        计算查询与文档之间的余弦相似度得分。

        :param tenant_id: 租户ID，用于获取特定租户的模型实例
        :param query: 搜索查询字符串
        :param documents: 包含多个文档的列表
        :param vector_setting: 向量设置对象，包含嵌入模型的信息
        :return: 每个文档与查询之间的余弦相似度得分列表
        """
        # 初始化用于存储查询与文档之间余弦相似度得分的列表
        query_vector_scores = []
        # 创建模型管理器实例
        model_manager = ModelManager()
        # 获取指定租户的嵌入模型实例
        embedding_model = model_manager.get_model_instance(
            tenant_id=tenant_id, # 租户ID
            provider=vector_setting.embedding_provider_name, # 嵌入模型提供商名称
            model_type=ModelType.TEXT_EMBEDDING, # 模型类型为文本嵌入
            model=vector_setting.embedding_model_name # 嵌入模型名称

        )
        # 创建缓存嵌入实例
        cache_embedding = CacheEmbedding(embedding_model)
        # 为查询生成嵌入向量
        query_vector = cache_embedding.embed_query(query)
        # 遍历每个文档
        for document in documents:
            # 如果文档元数据中已经存在得分，则直接使用
            if 'score' in document.metadata:
                query_vector_scores.append(document.metadata['score'])
            else:
                # 获取文档的嵌入向量
                content_vector = document.metadata['vector']
                # 将查询向量和文档向量转换为NumPy数组
                vec1 = np.array(query_vector)
                vec2 = np.array(document.metadata['vector'])

                # 计算点积
                dot_product = np.dot(vec1, vec2)

                # 计算向量范数
                norm_vec1 = np.linalg.norm(vec1)
                norm_vec2 = np.linalg.norm(vec2)

                # 计算余弦相似度
                cosine_sim = dot_product / (norm_vec1 * norm_vec2)
                # 将余弦相似度得分添加到列表中
                query_vector_scores.append(cosine_sim)

        return query_vector_scores
