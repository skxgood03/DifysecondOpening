from typing import Optional

from core.model_manager import ModelManager
from core.model_runtime.entities.model_entities import ModelType
from core.model_runtime.errors.invoke import InvokeAuthorizationError
from core.rag.data_post_processor.reorder import ReorderRunner
from core.rag.models.document import Document
from core.rag.rerank.constants.rerank_mode import RerankMode
from core.rag.rerank.entity.weight import KeywordSetting, VectorSetting, Weights
from core.rag.rerank.rerank_model import RerankModelRunner
from core.rag.rerank.weight_rerank import WeightRerankRunner


class DataPostProcessor:
    """Interface for data post-processing document.
    """

    def __init__(self, tenant_id: str, reranking_mode: str,
                 reranking_model: Optional[dict] = None, weights: Optional[dict] = None,
                 reorder_enabled: bool = False):
        self.rerank_runner = self._get_rerank_runner(reranking_mode, tenant_id, reranking_model, weights)
        self.reorder_runner = self._get_reorder_runner(reorder_enabled)

    def invoke(self, query: str, documents: list[Document], score_threshold: Optional[float] = None,
               top_n: Optional[int] = None, user: Optional[str] = None) -> list[Document]:
        #根据返回的实例WeightRerankRunner/ModelManager调用对应的run方法
        if self.rerank_runner:
            documents = self.rerank_runner.run(query, documents, score_threshold, top_n, user)

        if self.reorder_runner:
            documents = self.reorder_runner.run(documents)

        return documents

    def _get_rerank_runner(self, reranking_mode: str, tenant_id: str, reranking_model: Optional[dict] = None,
                           weights: Optional[dict] = None) -> Optional[RerankModelRunner | WeightRerankRunner]:
        # 根据提供的配置来创建一个重排序（reranking）运行器实例。重排序通常用于调整搜索结果的顺序以提高相关性。

        # 判断是否使用基于权重的重排序模式
        if reranking_mode == RerankMode.WEIGHTED_SCORE.value and weights:
            # 创建 WeightRerankRunner 实例
            return WeightRerankRunner(
                tenant_id,  # 租户ID
                Weights( # 创建 Weights 对象
                    vector_setting=VectorSetting( # 创建 VectorSetting 对象
                        vector_weight=weights['vector_setting']['vector_weight'],  # 向量权重
                        embedding_provider_name=weights['vector_setting']['embedding_provider_name'], # 嵌入式提供商名称
                        embedding_model_name=weights['vector_setting']['embedding_model_name'],  # 嵌入式模型名称
                    ),
                    keyword_setting=KeywordSetting( # 创建 KeywordSetting 对象
                        keyword_weight=weights['keyword_setting']['keyword_weight'], # 关键词权重
                    )
                )
            )
        # 判断是否使用基于模型的重排序模式
        elif reranking_mode == RerankMode.RERANKING_MODEL.value:
            if reranking_model:
                try:
                    # 创建 ModelManager 实例
                    model_manager = ModelManager()
                    # 获取模型实例
                    rerank_model_instance = model_manager.get_model_instance(
                        tenant_id=tenant_id, # 租户ID
                        provider=reranking_model['reranking_provider_name'], # 模型提供商名称
                        model_type=ModelType.RERANK,  # 模型类型
                        model=reranking_model['reranking_model_name'] # 模型名称
                    )
                except InvokeAuthorizationError:
                    return None
                return RerankModelRunner(rerank_model_instance)
            return None
        # 如果既不是基于权重也不是基于模型的重排序模式，则返回 None
        return None

    def _get_reorder_runner(self, reorder_enabled) -> Optional[ReorderRunner]:
        if reorder_enabled:
            return ReorderRunner()
        return None


