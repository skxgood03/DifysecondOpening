"""Paragraph index processor."""
import time
import uuid
from typing import Optional

import meilisearch
import openai
from meilisearch.models.task import Task
from tqdm import tqdm

from core.rag.cleaner.clean_processor import CleanProcessor
from core.rag.datasource.keyword.keyword_factory import Keyword
from core.rag.datasource.retrieval_service import RetrievalService
from core.rag.datasource.vdb.vector_factory import Vector
from core.rag.extractor.entity.extract_setting import ExtractSetting
from core.rag.extractor.extract_processor import ExtractProcessor
from core.rag.index_processor.index_processor_base import BaseIndexProcessor
from core.rag.models.document import Document
from libs import helper
from models.dataset import Dataset

meili_client = meilisearch.Client("http://127.0.0.1:7700", "LD9I0_C10_Ee68J0HsztV3B_gO8eITzk2yNaruep_I")
openai_client = openai.OpenAI(api_key="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",
                              base_url="https://api-openai-proxy.gempoll.com/v1")


class ParagraphIndexProcessor(BaseIndexProcessor):

    def extract(self, extract_setting: ExtractSetting, **kwargs) -> list[Document]:

        text_docs = ExtractProcessor.extract(extract_setting=extract_setting,
                                             is_automatic=kwargs.get('process_rule_mode') == "automatic")

        return text_docs

    def transform(self, documents: list[Document], **kwargs) -> list[Document]:
        """
           将文本文档分割成节点，并对每个节点进行清理和元数据处理。

           :param documents: 待处理的文档列表
           :param kwargs: 关键字参数，包括处理规则、嵌入模型实例等
           :return: 处理后的文档节点列表
           这段代码实现了将一系列文档分割成更小的节点，并对这些节点进行清理和元数据处理的功能。
           它首先根据传入的处理规则和嵌入模型实例选择一个适当的文档分割器。
           然后，遍历每个文档，先清理文档内容，再使用分割器将其分割成多个节点。
           对于每个节点，它生成一个唯一的文档ID和内容的哈希值，更新节点的元数据，并清除可能存在的分割符。
           最后，将处理后的文档节点添加到结果列表中并返回。
           """
        #  # 选择文档分割器
        splitter = self._get_splitter(processing_rule=kwargs.get('process_rule'),  # 处理规则
                                      embedding_model_instance=kwargs.get('embedding_model_instance'))  # 嵌入模型实例
        all_documents = []  # 初始化所有文档节点列表
        for document in documents:  # 遍历每个文档
            # 清理文档内容
            document_text = CleanProcessor.clean(document.page_content, kwargs.get('process_rule'))
            document.page_content = document_text  # 更新文档内容
            #  # 将文档分割成节点
            document_nodes = splitter.split_documents([document])  # 为文档的切片具体实现
            split_documents = []  # 初始化分割后的文档节点列表
            for document_node in document_nodes:  # 遍历每个文档节点

                if document_node.page_content.strip():  # 如果节点内容非空
                    # 生成唯一文档ID和哈希值
                    doc_id = str(uuid.uuid4())
                    hash = helper.generate_text_hash(document_node.page_content)
                    # 更新文档节点元数据
                    document_node.metadata['doc_id'] = doc_id
                    document_node.metadata['doc_hash'] = hash
                    # delete Splitter character
                    page_content = document_node.page_content
                    if page_content.startswith(".") or page_content.startswith("。"):
                        page_content = page_content[1:].strip()  # 去除开头的点或句号
                    else:
                        page_content = page_content
                    if len(page_content) > 0:  # 如果处理后的内容长度大于0
                        document_node.page_content = page_content  # 更新文档节点内容
                        split_documents.append(document_node)  # 添加到分割文档列表
            all_documents.extend(split_documents)  # 将分割后的文档添加到总列表
        return all_documents  # 返回处理后的文档节点列表

    def get_task(self, task: Task, wait_time: float = 0.5):
        time.sleep(wait_time)
        task = meili_client.get_task(task.task_uid)
        return task

    def embed(self, text: str):

        emb = openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        return emb.data[0].embedding

    def load(self, dataset: Dataset, documents: list[Document], with_keywords: bool = True):

        """
        根据数据集的索引技术和需求加载文档的向量或关键词索引。

        :param dataset: 数据集对象，包含索引技术等信息。
        :param documents: 待加载索引的文档列表。
        :param with_keywords: 是否创建关键词索引，默认为True。
        """

        # 如果数据集的索引技术设置为'high_quality'
        if dataset.indexing_technique == 'high_quality':
            # if dataset.retrieval_model['search_method'] == 'meili_search':
            #     pass
            # else:
            # handler = MeiliSearchHandler("config.json")
            # data = handler.prepare_data([document.page_content for document in documents])
            # print('向量数据', data)
            # documents = [
            #     {
            #         "id": record["id"],
            #         "text": record["text"],
            #         "_vectors": {
            #             handler.embedder_name: record["embedding"]
            #         }
            #     }
            #     for record in data
            # ]
            # handler.add_documents(documents)
            # print('-----向量数据存储完成')
            # 创建Vector实例，传入数据集
            vector = Vector(dataset)

            # 使用Vector实例创建文档的向量索引
            vector.create(documents)

        #
        # load_meili(documents)

        # 如果需要创建关键词索引
        if with_keywords:
            # 创建Keyword实例，传入数据集
            keyword = Keyword(dataset)

            # 使用Keyword实例创建文档的关键词索引
            keyword.create(documents)

    def load_meili(self, dataset: Dataset, documents: list[Document]):

        content_list = [document.page_content for document in documents]
        print(content_list)
        data = []
        for i, record in enumerate(tqdm(content_list)):
            dic = {}
            dic["id"] = i + 100
            embedding = self.embed(record)
            dic["embedding"] = embedding
            dic["text"] = record
            data.append(dic)

        # 构建文档列表
        documents = [
            {
                "id": record["id"],
                "text": record["text"],
                "_vectors": {
                    "demo": record["embedding"]
                }
            }
            for record in data
        ]
        meili_client = meilisearch.Client("http://127.0.0.1:7700", "LD9I0_C10_Ee68J0HsztV3B_gO8eITzk2yNaruep_I")
        add_documents_task = meili_client.index("xt81_db").add_documents(documents)

        while True:
            print('进入任务查询循环')
            be_task = self.get_task(add_documents_task, wait_time=2)
            print(be_task)
            if be_task.status == "succeeded":
                print('任务完成')
                break
            elif be_task.status == "failed":
                print('任务失败')
                break

    def clean(self, dataset: Dataset, node_ids: Optional[list[str]], with_keywords: bool = True):
        if dataset.indexing_technique == 'high_quality':
            vector = Vector(dataset)
            if node_ids:
                vector.delete_by_ids(node_ids)
            else:
                vector.delete()
        if with_keywords:
            keyword = Keyword(dataset)
            if node_ids:
                keyword.delete_by_ids(node_ids)
            else:
                keyword.delete()

    def retrieve(self, retrieval_method: str, query: str, dataset: Dataset, top_k: int,
                 score_threshold: float, reranking_model: dict) -> list[Document]:
        # Set search parameters.
        results = RetrievalService.retrieve(retrieval_method=retrieval_method, dataset_id=dataset.id, query=query,
                                            top_k=top_k, score_threshold=score_threshold,
                                            reranking_model=reranking_model)
        # Organize results.
        print('进来看了')
        docs = []
        for result in results:
            metadata = result.metadata
            metadata['score'] = result.score
            if result.score > score_threshold:
                doc = Document(page_content=result.page_content, metadata=metadata)
                docs.append(doc)
        return docs
