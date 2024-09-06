import concurrent.futures
import datetime
import json
import logging
import re
import threading
import time
import uuid
from typing import Optional, cast

from flask import Flask, current_app
from flask_login import current_user
from sqlalchemy.orm.exc import ObjectDeletedError

from configs import dify_config
from core.errors.error import ProviderTokenNotInitError
from core.llm_generator.llm_generator import LLMGenerator
from core.model_manager import ModelInstance, ModelManager
from core.model_runtime.entities.model_entities import ModelType
from core.rag.datasource.keyword.keyword_factory import Keyword
from core.rag.docstore.dataset_docstore import DatasetDocumentStore
from core.rag.extractor.entity.extract_setting import ExtractSetting
from core.rag.index_processor.index_processor_base import BaseIndexProcessor
from core.rag.index_processor.index_processor_factory import IndexProcessorFactory
from core.rag.models.document import Document
from core.rag.splitter.fixed_text_splitter import (
    EnhanceRecursiveCharacterTextSplitter,
    FixedRecursiveCharacterTextSplitter,
)
from core.rag.splitter.text_splitter import TextSplitter
from extensions.ext_database import db
from extensions.ext_redis import redis_client
from extensions.ext_storage import storage
from libs import helper
from models.dataset import Dataset, DatasetProcessRule, DocumentSegment
from models.dataset import Document as DatasetDocument
from models.model import UploadFile
from services.feature_service import FeatureService


class IndexingRunner:

    def __init__(self):
        self.storage = storage
        self.model_manager = ModelManager()

    def run(self, dataset_documents: list[DatasetDocument]):
        """Run the indexing process."""
        """
        运行索引过程，对每个提供的数据集文档进行处理。
        """
        for dataset_document in dataset_documents:
            try:
                # get dataset
                dataset = Dataset.query.filter_by(
                    id=dataset_document.dataset_id
                ).first()

                if not dataset:
                    raise ValueError("no dataset found")

                # 获取处理规则
                processing_rule = db.session.query(DatasetProcessRule). \
                    filter(DatasetProcessRule.id == dataset_document.dataset_process_rule_id). \
                    first()
                index_type = dataset_document.doc_form  # 文档的形式，用于确定索引处理器类型
                index_processor = IndexProcessorFactory(index_type).init_index_processor()  # 创建索引处理器实例
                # 提取文本数据
                text_docs = self._extract(index_processor, dataset_document, processing_rule.to_dict())
                # print('提取文本数据', text_docs)
                print('提取文本数据')
                # [Document(page_content='问题：Dify是什么？\n答案：Dify是一个LLMOps平台', metadata={
                #     'source': 'C:\\Users\\Administrator\\AppData\\Local\\Temp\\tmpf_jug27j/m07u889n.txt',
                #     'document_id': '36657e00-7def-4974-b348-fb0de5257af0',
                #     'dataset_id': 'd9e421d7-13ec-4946-98f3-96e505ff5005'})]
                # 转换数据
                documents = self._transform(index_processor, dataset, text_docs, dataset_document.doc_language,
                                            processing_rule.to_dict())
                # print('切片数据', documents)
                print('切片数据')
                """
                [Document(page_content='问题：Dify是什么？', metadata={'source': 'C:\\Users\\Administrator\\AppData\\Local\\Temp\\tmpktjcz598/qwrh3kmh.txt', 'document_id': 'e4c4bb02-6c6
                    b1e0-0c59085a5b06', 'dataset_id': '0d251c8a-96de-4a5b-bf0c-9799e2721e8d', 'doc_id': '9d3ecec9-b3e6-424b-9e22-41e8f479d748', 'doc_hash': '3b904654106717ebf7e178618c9551e5f22a8558c343c8a449bf881b7080d149'
                        }), 
                        Document(page_content='答案：Dify是一个LLMOps平台', metadata={'source': 'C:\\Users\\Administrator\\AppData\\Local\\Temp\\tmpktjcz598/qwrh3kmh.txt', 'document_id': 'e4c4bb02-6c6d-4e57-b1e0-0c59085a5b06', 'dataset_id': '0
                    d251c8a-96de-4a5b-bf0c-9799e2721e8d', 'doc_id': 'b9f751ff-064c-47c6-adb6-d282ddca8e9d', 'doc_hash': '0bb44ba60eb393aa868562cc0fbd01e7c2bb8d9feefe1842984eeea20d4d7bf7'
                        }), 
                        Document(page_content='#在api/core/indexing_ru
                    nner.py的 self._load加入 meiliserach的添加数据逻辑 创建线程--等待线程完成', metadata={'source': 'C:\\Users\\Administrator\\AppData\\Local\\Temp\\tmpktjcz598/qwrh3kmh.txt', 'document_id': 'e4c4bb02-6c6d-4e57-b1e0-0c59
                    085a5b06', 'dataset_id': '0d251c8a-96de-4a5b-bf0c-9799e2721e8d', 'doc_id': '15f196a1-ecca-4394-9d1b-ca910e7a0018', 'doc_hash': 'c65e1d31fb32a422c04f9b71c05210398e41540aac3917572cdd92d7c051df55'
                        }), 
                        Document(page_conte
                    nt='# 在api/core/rag/datasource/retrieval_service.py 中加入 Meilisearch选项判断 然后再写查询逻辑，最后返回结果', metadata={'source': 'C:\\Users\\Administrator\\AppData\\Local\\Temp\\tmpktjcz598/qwrh3kmh.txt', 'docume
                    nt_id': 'e4c4bb02-6c6d-4e57-b1e0-0c59085a5b06', 'dataset_id': '0d251c8a-96de-4a5b-bf0c-9799e2721e8d', 'doc_id': '2fbe8675-61cf-45c2-9b22-e3092e67d23f', 'doc_hash': '102a7dbe28185d361e32aa053e5cf505c9b2b67ccf2a8d633e3e2837a447efd3'
                        })
                    ]
                """

                # 保存片段 将最终切片后的 chunks 构造 document_segment 入库
                self._load_segments(dataset, dataset_document, documents)

                # load
                self._load(
                    index_processor=index_processor,
                    dataset=dataset,
                    dataset_document=dataset_document,
                    documents=documents
                )
            except DocumentIsPausedException:
                # 如果文档被暂停，抛出异常
                raise DocumentIsPausedException('Document paused, document id: {}'.format(dataset_document.id))
            except ProviderTokenNotInitError as e:
                # 如果提供商令牌未初始化，更新文档状态并提交更改
                dataset_document.indexing_status = 'error'
                dataset_document.error = str(e.description)
                dataset_document.stopped_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
                db.session.commit()
            except ObjectDeletedError:
                # 如果对象被删除，记录警告日志
                logging.warning('Document deleted, document id: {}'.format(dataset_document.id))
            except Exception as e:  # 对于其他异常，记录异常信息并更新文档状态
                logging.exception("consume document failed")
                dataset_document.indexing_status = 'error'
                dataset_document.error = str(e)
                dataset_document.stopped_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
                db.session.commit()

    def run_in_splitting_status(self, dataset_document: DatasetDocument):
        """Run the indexing process when the index_status is splitting."""
        try:
            # get dataset
            dataset = Dataset.query.filter_by(
                id=dataset_document.dataset_id
            ).first()

            if not dataset:
                raise ValueError("no dataset found")

            # get exist document_segment list and delete
            document_segments = DocumentSegment.query.filter_by(
                dataset_id=dataset.id,
                document_id=dataset_document.id
            ).all()

            for document_segment in document_segments:
                db.session.delete(document_segment)
            db.session.commit()
            # get the process rule
            processing_rule = db.session.query(DatasetProcessRule). \
                filter(DatasetProcessRule.id == dataset_document.dataset_process_rule_id). \
                first()

            index_type = dataset_document.doc_form
            index_processor = IndexProcessorFactory(index_type).init_index_processor()
            # extract
            text_docs = self._extract(index_processor, dataset_document, processing_rule.to_dict())

            # transform
            documents = self._transform(index_processor, dataset, text_docs, dataset_document.doc_language,
                                        processing_rule.to_dict())
            # save segment
            self._load_segments(dataset, dataset_document, documents)

            # load
            self._load(
                index_processor=index_processor,
                dataset=dataset,
                dataset_document=dataset_document,
                documents=documents
            )
        except DocumentIsPausedException:
            raise DocumentIsPausedException('Document paused, document id: {}'.format(dataset_document.id))
        except ProviderTokenNotInitError as e:
            dataset_document.indexing_status = 'error'
            dataset_document.error = str(e.description)
            dataset_document.stopped_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
            db.session.commit()
        except Exception as e:
            logging.exception("consume document failed")
            dataset_document.indexing_status = 'error'
            dataset_document.error = str(e)
            dataset_document.stopped_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
            db.session.commit()

    def run_in_indexing_status(self, dataset_document: DatasetDocument):
        """Run the indexing process when the index_status is indexing."""
        try:
            # get dataset
            dataset = Dataset.query.filter_by(
                id=dataset_document.dataset_id
            ).first()

            if not dataset:
                raise ValueError("no dataset found")

            # get exist document_segment list and delete
            document_segments = DocumentSegment.query.filter_by(
                dataset_id=dataset.id,
                document_id=dataset_document.id
            ).all()

            documents = []
            if document_segments:
                for document_segment in document_segments:
                    # transform segment to node
                    if document_segment.status != "completed":
                        document = Document(
                            page_content=document_segment.content,
                            metadata={
                                "doc_id": document_segment.index_node_id,
                                "doc_hash": document_segment.index_node_hash,
                                "document_id": document_segment.document_id,
                                "dataset_id": document_segment.dataset_id,
                            }
                        )

                        documents.append(document)

            # build index
            # get the process rule
            processing_rule = db.session.query(DatasetProcessRule). \
                filter(DatasetProcessRule.id == dataset_document.dataset_process_rule_id). \
                first()

            index_type = dataset_document.doc_form
            index_processor = IndexProcessorFactory(index_type).init_index_processor()
            self._load(
                index_processor=index_processor,
                dataset=dataset,
                dataset_document=dataset_document,
                documents=documents
            )
        except DocumentIsPausedException:
            raise DocumentIsPausedException('Document paused, document id: {}'.format(dataset_document.id))
        except ProviderTokenNotInitError as e:
            dataset_document.indexing_status = 'error'
            dataset_document.error = str(e.description)
            dataset_document.stopped_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
            db.session.commit()
        except Exception as e:
            logging.exception("consume document failed")
            dataset_document.indexing_status = 'error'
            dataset_document.error = str(e)
            dataset_document.stopped_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
            db.session.commit()

    def indexing_estimate(self, tenant_id: str, extract_settings: list[ExtractSetting], tmp_processing_rule: dict,
                          doc_form: str = None, doc_language: str = 'English', dataset_id: str = None,
                          indexing_technique: str = 'economy') -> dict:
        """
        Estimate the indexing for the document.
        """
        # check document limit
        features = FeatureService.get_features(tenant_id)
        if features.billing.enabled:
            count = len(extract_settings)
            batch_upload_limit = dify_config.BATCH_UPLOAD_LIMIT
            if count > batch_upload_limit:
                raise ValueError(f"You have reached the batch upload limit of {batch_upload_limit}.")

        embedding_model_instance = None
        if dataset_id:
            dataset = Dataset.query.filter_by(
                id=dataset_id
            ).first()
            if not dataset:
                raise ValueError('Dataset not found.')
            if dataset.indexing_technique == 'high_quality' or indexing_technique == 'high_quality':
                if dataset.embedding_model_provider:
                    embedding_model_instance = self.model_manager.get_model_instance(
                        tenant_id=tenant_id,
                        provider=dataset.embedding_model_provider,
                        model_type=ModelType.TEXT_EMBEDDING,
                        model=dataset.embedding_model
                    )
                else:
                    embedding_model_instance = self.model_manager.get_default_model_instance(
                        tenant_id=tenant_id,
                        model_type=ModelType.TEXT_EMBEDDING,
                    )
        else:
            if indexing_technique == 'high_quality':
                embedding_model_instance = self.model_manager.get_default_model_instance(
                    tenant_id=tenant_id,
                    model_type=ModelType.TEXT_EMBEDDING,
                )
        preview_texts = []
        total_segments = 0
        index_type = doc_form
        index_processor = IndexProcessorFactory(index_type).init_index_processor()
        all_text_docs = []
        for extract_setting in extract_settings:
            # extract
            text_docs = index_processor.extract(extract_setting, process_rule_mode=tmp_processing_rule["mode"])
            all_text_docs.extend(text_docs)
            processing_rule = DatasetProcessRule(
                mode=tmp_processing_rule["mode"],
                rules=json.dumps(tmp_processing_rule["rules"])
            )

            # get splitter
            splitter = self._get_splitter(processing_rule, embedding_model_instance)

            # split to documents
            documents = self._split_to_documents_for_estimate(
                text_docs=text_docs,
                splitter=splitter,
                processing_rule=processing_rule
            )

            total_segments += len(documents)
            for document in documents:
                if len(preview_texts) < 5:
                    preview_texts.append(document.page_content)

        if doc_form and doc_form == 'qa_model':

            if len(preview_texts) > 0:
                # qa model document
                response = LLMGenerator.generate_qa_document(current_user.current_tenant_id, preview_texts[0],
                                                             doc_language)
                document_qa_list = self.format_split_text(response)

                return {
                    "total_segments": total_segments * 20,
                    "qa_preview": document_qa_list,
                    "preview": preview_texts
                }
        return {
            "total_segments": total_segments,
            "preview": preview_texts
        }

    def _extract(self, index_processor: BaseIndexProcessor, dataset_document: DatasetDocument, process_rule: dict) \
            -> list[Document]:
        """
            根据数据源类型提取文档内容。
            :param index_processor: 索引处理器实例
            :param dataset_document: 数据集文档对象
            :param process_rule: 处理规则字典
            :return: 包含提取后文档内容的列表
            这段代码定义了一个名为_extract的方法，用于根据不同的数据源类型（上传文件、Notion导入、网站爬取）提取文档内容。
            它首先检查数据源类型，然后根据类型创建相应的ExtractSetting对象，
            并调用index_processor的extract方法来提取文档。
            之后，它会更新文档的状态为“分割”，计算并更新文档的词数，以及完成解析的时间。
            最后，它会更新提取的文档元数据中的文档ID和数据集ID，以关联到正确的数据集文档。

        """
        # 加载文件，如果数据源类型不是上传文件、Notion导入或网站爬取，则返回空列表
        if dataset_document.data_source_type not in ["upload_file", "notion_import", "website_crawl"]:
            return []

        data_source_info = dataset_document.data_source_info_dict
        text_docs = []  # 初始化文本文档列表
        # 处理上传文件数据源
        if dataset_document.data_source_type == 'upload_file':
            if not data_source_info or 'upload_file_id' not in data_source_info:
                raise ValueError("no upload file found")

            # 查询上传文件详情
            file_detail = db.session.query(UploadFile). \
                filter(UploadFile.id == data_source_info['upload_file_id']). \
                one_or_none()

            if file_detail:
                # 创建提取设置
                extract_setting = ExtractSetting(
                    datasource_type="upload_file",
                    upload_file=file_detail,
                    document_model=dataset_document.doc_form
                )
                # 使用索引处理器提取文档
                text_docs = index_processor.extract(extract_setting, process_rule_mode=process_rule['mode'])
        # 处理Notion导入数据源
        elif dataset_document.data_source_type == 'notion_import':
            if (not data_source_info or 'notion_workspace_id' not in data_source_info
                    or 'notion_page_id' not in data_source_info):
                raise ValueError("no notion import info found")
            # 创建提取设置
            extract_setting = ExtractSetting(
                datasource_type="notion_import",
                notion_info={
                    "notion_workspace_id": data_source_info['notion_workspace_id'],
                    "notion_obj_id": data_source_info['notion_page_id'],
                    "notion_page_type": data_source_info['type'],
                    "document": dataset_document,
                    "tenant_id": dataset_document.tenant_id
                },
                document_model=dataset_document.doc_form
            )
            # 使用索引处理器提取文档
            text_docs = index_processor.extract(extract_setting, process_rule_mode=process_rule['mode'])
        # 处理网站爬取数据源
        elif dataset_document.data_source_type == 'website_crawl':
            if (not data_source_info or 'provider' not in data_source_info
                    or 'url' not in data_source_info or 'job_id' not in data_source_info):
                raise ValueError("no website import info found")
            # 创建提取设置
            extract_setting = ExtractSetting(
                datasource_type="website_crawl",
                website_info={
                    "provider": data_source_info['provider'],
                    "job_id": data_source_info['job_id'],
                    "tenant_id": dataset_document.tenant_id,
                    "url": data_source_info['url'],
                    "mode": data_source_info['mode'],
                    "only_main_content": data_source_info['only_main_content']
                },
                document_model=dataset_document.doc_form
            )
            # 使用索引处理器提取文档
            text_docs = index_processor.extract(extract_setting, process_rule_mode=process_rule['mode'])
        # update document status to splitting
        # 更新文档状态为“分割”阶段
        self._update_document_index_status(
            document_id=dataset_document.id,
            after_indexing_status="splitting",
            extra_update_params={
                DatasetDocument.word_count: sum(len(text_doc.page_content) for text_doc in text_docs),
                DatasetDocument.parsing_completed_at: datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
            }
        )

        ## 替换文档ID为数据集文档模型ID
        text_docs = cast(list[Document], text_docs)  # 类型断言，确保text_docs为Document列表
        for text_doc in text_docs:
            text_doc.metadata['document_id'] = dataset_document.id
            text_doc.metadata['dataset_id'] = dataset_document.dataset_id

        return text_docs

    @staticmethod
    def filter_string(text):
        text = re.sub(r'<\|', '<', text)
        text = re.sub(r'\|>', '>', text)
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\xEF\xBF\xBE]', '', text)
        # Unicode  U+FFFE
        text = re.sub('\uFFFE', '', text)
        return text

    @staticmethod
    def _get_splitter(processing_rule: DatasetProcessRule,
                      embedding_model_instance: Optional[ModelInstance]) -> TextSplitter:
        """
        Get the NodeParser object according to the processing rule.
        """
        if processing_rule.mode == "custom":
            # The user-defined segmentation rule
            rules = json.loads(processing_rule.rules)
            segmentation = rules["segmentation"]
            max_segmentation_tokens_length = dify_config.INDEXING_MAX_SEGMENTATION_TOKENS_LENGTH
            if segmentation["max_tokens"] < 50 or segmentation["max_tokens"] > max_segmentation_tokens_length:
                raise ValueError(f"Custom segment length should be between 50 and {max_segmentation_tokens_length}.")

            separator = segmentation["separator"]
            if separator:
                separator = separator.replace('\\n', '\n')

            if segmentation.get('chunk_overlap'):
                chunk_overlap = segmentation['chunk_overlap']
            else:
                chunk_overlap = 0

            character_splitter = FixedRecursiveCharacterTextSplitter.from_encoder(
                chunk_size=segmentation["max_tokens"],
                chunk_overlap=chunk_overlap,
                fixed_separator=separator,
                separators=["\n\n", "。", ". ", " ", ""],
                embedding_model_instance=embedding_model_instance
            )
        else:
            # Automatic segmentation
            character_splitter = EnhanceRecursiveCharacterTextSplitter.from_encoder(
                chunk_size=DatasetProcessRule.AUTOMATIC_RULES['segmentation']['max_tokens'],
                chunk_overlap=DatasetProcessRule.AUTOMATIC_RULES['segmentation']['chunk_overlap'],
                separators=["\n\n", "。", ". ", " ", ""],
                embedding_model_instance=embedding_model_instance
            )

        return character_splitter

    def _step_split(self, text_docs: list[Document], splitter: TextSplitter,
                    dataset: Dataset, dataset_document: DatasetDocument, processing_rule: DatasetProcessRule) \
            -> list[Document]:
        """
        Split the text documents into documents and save them to the document segment.
        """
        documents = self._split_to_documents(
            text_docs=text_docs,
            splitter=splitter,
            processing_rule=processing_rule,
            tenant_id=dataset.tenant_id,
            document_form=dataset_document.doc_form,
            document_language=dataset_document.doc_language
        )

        # save node to document segment
        doc_store = DatasetDocumentStore(
            dataset=dataset,
            user_id=dataset_document.created_by,
            document_id=dataset_document.id
        )

        # add document segments
        doc_store.add_documents(documents)

        # update document status to indexing
        cur_time = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        self._update_document_index_status(
            document_id=dataset_document.id,
            after_indexing_status="indexing",
            extra_update_params={
                DatasetDocument.cleaning_completed_at: cur_time,
                DatasetDocument.splitting_completed_at: cur_time,
            }
        )

        # update segment status to indexing
        self._update_segments_by_document(
            dataset_document_id=dataset_document.id,
            update_params={
                DocumentSegment.status: "indexing",
                DocumentSegment.indexing_at: datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
            }
        )

        return documents

    def _split_to_documents(self, text_docs: list[Document], splitter: TextSplitter,
                            processing_rule: DatasetProcessRule, tenant_id: str,
                            document_form: str, document_language: str) -> list[Document]:
        """
        Split the text documents into nodes.
        """
        all_documents = []
        all_qa_documents = []
        for text_doc in text_docs:
            # document clean
            document_text = self._document_clean(text_doc.page_content, processing_rule)
            text_doc.page_content = document_text

            # parse document to nodes
            documents = splitter.split_documents([text_doc])
            split_documents = []
            for document_node in documents:

                if document_node.page_content.strip():
                    doc_id = str(uuid.uuid4())
                    hash = helper.generate_text_hash(document_node.page_content)
                    document_node.metadata['doc_id'] = doc_id
                    document_node.metadata['doc_hash'] = hash
                    # delete Spliter character
                    page_content = document_node.page_content
                    if page_content.startswith(".") or page_content.startswith("。"):
                        page_content = page_content[1:]
                    else:
                        page_content = page_content
                    document_node.page_content = page_content

                    if document_node.page_content:
                        split_documents.append(document_node)
            all_documents.extend(split_documents)
        # processing qa document
        if document_form == 'qa_model':
            for i in range(0, len(all_documents), 10):
                threads = []
                sub_documents = all_documents[i:i + 10]
                for doc in sub_documents:
                    document_format_thread = threading.Thread(target=self.format_qa_document, kwargs={
                        'flask_app': current_app._get_current_object(),
                        'tenant_id': tenant_id, 'document_node': doc, 'all_qa_documents': all_qa_documents,
                        'document_language': document_language})
                    threads.append(document_format_thread)
                    document_format_thread.start()
                for thread in threads:
                    thread.join()
            return all_qa_documents
        return all_documents

    def format_qa_document(self, flask_app: Flask, tenant_id: str, document_node, all_qa_documents, document_language):
        format_documents = []
        if document_node.page_content is None or not document_node.page_content.strip():
            return
        with flask_app.app_context():
            try:
                # qa model document
                response = LLMGenerator.generate_qa_document(tenant_id, document_node.page_content, document_language)
                document_qa_list = self.format_split_text(response)
                qa_documents = []
                for result in document_qa_list:
                    qa_document = Document(page_content=result['question'],
                                           metadata=document_node.metadata.model_copy())
                    doc_id = str(uuid.uuid4())
                    hash = helper.generate_text_hash(result['question'])
                    qa_document.metadata['answer'] = result['answer']
                    qa_document.metadata['doc_id'] = doc_id
                    qa_document.metadata['doc_hash'] = hash
                    qa_documents.append(qa_document)
                format_documents.extend(qa_documents)
            except Exception as e:
                logging.exception(e)

            all_qa_documents.extend(format_documents)

    def _split_to_documents_for_estimate(self, text_docs: list[Document], splitter: TextSplitter,
                                         processing_rule: DatasetProcessRule) -> list[Document]:
        """
        Split the text documents into nodes.
        """
        all_documents = []
        for text_doc in text_docs:
            # document clean
            document_text = self._document_clean(text_doc.page_content, processing_rule)
            text_doc.page_content = document_text

            # parse document to nodes
            documents = splitter.split_documents([text_doc])

            split_documents = []
            for document in documents:
                if document.page_content is None or not document.page_content.strip():
                    continue
                doc_id = str(uuid.uuid4())
                hash = helper.generate_text_hash(document.page_content)

                document.metadata['doc_id'] = doc_id
                document.metadata['doc_hash'] = hash

                split_documents.append(document)

            all_documents.extend(split_documents)

        return all_documents

    @staticmethod
    def _document_clean(text: str, processing_rule: DatasetProcessRule) -> str:
        """
        Clean the document text according to the processing rules.
        """
        if processing_rule.mode == "automatic":
            rules = DatasetProcessRule.AUTOMATIC_RULES
        else:
            rules = json.loads(processing_rule.rules) if processing_rule.rules else {}

        if 'pre_processing_rules' in rules:
            pre_processing_rules = rules["pre_processing_rules"]
            for pre_processing_rule in pre_processing_rules:
                if pre_processing_rule["id"] == "remove_extra_spaces" and pre_processing_rule["enabled"] is True:
                    # Remove extra spaces
                    pattern = r'\n{3,}'
                    text = re.sub(pattern, '\n\n', text)
                    pattern = r'[\t\f\r\x20\u00a0\u1680\u180e\u2000-\u200a\u202f\u205f\u3000]{2,}'
                    text = re.sub(pattern, ' ', text)
                elif pre_processing_rule["id"] == "remove_urls_emails" and pre_processing_rule["enabled"] is True:
                    # Remove email
                    pattern = r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'
                    text = re.sub(pattern, '', text)

                    # Remove URL
                    pattern = r'https?://[^\s]+'
                    text = re.sub(pattern, '', text)

        return text

    @staticmethod
    def format_split_text(text):
        regex = r"Q\d+:\s*(.*?)\s*A\d+:\s*([\s\S]*?)(?=Q\d+:|$)"
        matches = re.findall(regex, text, re.UNICODE)

        return [
            {
                "question": q,
                "answer": re.sub(r"\n\s*", "\n", a.strip())
            }
            for q, a in matches if q and a
        ]

    def _load(self, index_processor: BaseIndexProcessor, dataset: Dataset,
              dataset_document: DatasetDocument, documents: list[Document]) -> None:
        """
        插入索引并更新文档/片段状态至已完成。
        """
        # 如果索引技术设置为'high_quality'，则获取相应的嵌入模型实例
        embedding_model_instance = None
        if dataset.indexing_technique == 'high_quality':
            print('高质量')
            embedding_model_instance = self.model_manager.get_model_instance(
                tenant_id=dataset.tenant_id,
                provider=dataset.embedding_model_provider,
                model_type=ModelType.TEXT_EMBEDDING,
                model=dataset.embedding_model
            )

        # 记录索引开始时间
        indexing_start_at = time.perf_counter()
        # 初始化计数器，用于统计处理的token数量
        tokens = 0
        # 设置每次处理的文档块大小
        chunk_size = 10

        # 创建关键词索引的线程
        create_keyword_thread = threading.Thread(target=self._process_keyword_index,
                                                 args=(current_app._get_current_object(),
                                                       dataset.id, dataset_document.id, documents))
        create_keyword_thread.start()  # 启动线程
        # 如果索引技术为'high_quality'，则并行处理文档块
        if dataset.indexing_technique == 'high_quality':
            # print('高质量块')
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:# 线程池执行器
                futures = []  # 存储Future对象的列表
                # 将文档列表分割成多个块

                for i in range(0, len(documents), chunk_size):
                    chunk_documents = documents[i:i + chunk_size]  # 块文档


                    # 提交任务到线程池
                    futures.append(
                        executor.submit(self._process_chunk, current_app._get_current_object(), index_processor,
                                        chunk_documents, dataset,
                                        dataset_document, embedding_model_instance))
                # 收集并处理所有Future的结果
                for future in futures:
                    tokens += future.result()  # 累加处理的token数量
                print('块文件结束')
        # 等待关键词索引线程完成
        create_keyword_thread.join()
        # 记录索引结束时间
        indexing_end_at = time.perf_counter()


        # create_meilisearch_thread = threading.Thread(target=self._meilisearch_index,
        #                                          args=(current_app._get_current_object(),
        #                                                dataset.id, dataset_document.id, documents))
        # create_meilisearch_thread.start()  # 启动线程
        # create_meilisearch_thread.join()
        # 更新文档状态至已完成
        self._update_document_index_status(
            document_id=dataset_document.id,
            after_indexing_status="completed",
            extra_update_params={
                DatasetDocument.tokens: tokens,
                DatasetDocument.completed_at: datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None),
                DatasetDocument.indexing_latency: indexing_end_at - indexing_start_at,
                DatasetDocument.error: None,
            }
        )

    @staticmethod
    def _process_keyword_index(flask_app, dataset_id, document_id, documents):
        with flask_app.app_context():
            # 查询数据集中指定ID的数据集
            dataset = Dataset.query.filter_by(id=dataset_id).first()
            if not dataset:
                raise ValueError("no dataset found")
            # 创建Keyword实例，传入数据集
            keyword = Keyword(dataset)
            # 使用Keyword实例创建关键词索引
            keyword.create(documents)
            # 如果索引技术不是高质素
            if dataset.indexing_technique != 'high_quality':
                # 从文档元数据中提取文档ID列表
                document_ids = [document.metadata['doc_id'] for document in documents]
                # 更新数据库中满足条件的文档片段状态
                db.session.query(DocumentSegment).filter(
                    DocumentSegment.document_id == document_id,
                    DocumentSegment.dataset_id == dataset_id,
                    DocumentSegment.index_node_id.in_(document_ids),
                    DocumentSegment.status == "indexing"
                ).update({
                    DocumentSegment.status: "completed",  # 更新状态为已完成
                    DocumentSegment.enabled: True,  # 启用文档片段
                    DocumentSegment.completed_at: datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
                    # 更新完成时间为当前时间
                })
                # 提交数据库会话，保存更改
                db.session.commit()


    # def embed(self,text: str):
    #
    #     emb = openai_client.embeddings.create(
    #         model="text-embedding-3-large",
    #         input=text
    #     )
    #     return emb.data[0].embedding
    #
    # def get_task(self,task: Task, wait_time: float = 0.5):
    #     time.sleep(wait_time)
    #     task = meili_client.get_task(task.task_uid)
    #
    #     return task
    # def _meilisearch_index(self, flask_app, dataset_id, document_id,  documents):
    #     with flask_app.app_context():
    #         content_list = [document.page_content for document in documents]
    #         print(content_list)
    #         data = []
    #         for i, record in enumerate(tqdm(content_list)):
    #             dic = {}
    #             dic["id"] = i + 100
    #             embedding = self.embed(record)
    #             dic["embedding"] = embedding
    #             dic["text"] = record
    #             data.append(dic)
    #
    #         # 构建文档列表
    #         documents = [
    #             {
    #                 "id": record["id"],
    #                 "text": record["text"],
    #                 "_vectors": {
    #                     "demo": record["embedding"]
    #                 }
    #             }
    #             for record in data
    #         ]
    #         meili_client = meilisearch.Client("http://127.0.0.1:7700", "LD9I0_C10_Ee68J0HsztV3B_gO8eITzk2yNaruep_I")
    #         add_documents_task = meili_client.index("xt56_db").add_documents(documents)
    #
    #         while True:
    #             print('进入任务查询循环')
    #             be_task = self.get_task(add_documents_task, wait_time=2)
    #             print(be_task)
    #             if be_task.status == "succeeded":
    #                 print('任务完成')
    #                 break
    #             elif be_task.status == "failed":
    #                 print('任务失败')
    #                 break

    def _process_chunk(self, flask_app, index_processor, chunk_documents, dataset, dataset_document,
                       embedding_model_instance):
        """
        处理文档块，包括嵌入和索引更新。

        :param flask_app: Flask应用实例，用于上下文管理
        :param index_processor: 索引处理器，用于加载文档到索引
        :param chunk_documents: 当前处理的文档块列表
        :param dataset: 数据集对象，包含文档所属的数据集信息
        :param dataset_document: 数据集文档对象，当前处理的文档元信息
        :param embedding_model_instance: 嵌入模型实例，用于文本嵌入
        :return: 总的token数，用于计费或资源统计
        """
        with flask_app.app_context():
            # 检查文档是否处于暂停状态，如果是则抛出异常
            self._check_document_paused_status(dataset_document.id)

            tokens = 0
            if embedding_model_instance:
                # 计算所有文档块的总token数，用于后续的计费或资源统计
                tokens += sum(
                    embedding_model_instance.get_text_embedding_num_tokens(
                        [document.page_content]
                    )
                    for document in chunk_documents
                )
            # 加载文档到索引，不包含关键词
            index_processor.load(dataset, chunk_documents, with_keywords=False)
            # 提取所有文档块的ID
            document_ids = [document.metadata['doc_id'] for document in chunk_documents]
            # 更新数据库中对应文档段的状态，设置为已完成，启用，并记录完成时间
            db.session.query(DocumentSegment).filter(
                DocumentSegment.document_id == dataset_document.id,
                DocumentSegment.dataset_id == dataset.id,
                DocumentSegment.index_node_id.in_(document_ids),
                DocumentSegment.status == "indexing"
            ).update({
                DocumentSegment.status: "completed",
                DocumentSegment.enabled: True,
                DocumentSegment.completed_at: datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
            })
            # 提交数据库会话，保存更改
            db.session.commit()
            # 返回处理的总token数
            return tokens

    @staticmethod
    def _check_document_paused_status(document_id: str):
        indexing_cache_key = 'document_{}_is_paused'.format(document_id)
        result = redis_client.get(indexing_cache_key)
        if result:
            raise DocumentIsPausedException()

    @staticmethod
    def _update_document_index_status(document_id: str, after_indexing_status: str,
                                      extra_update_params: Optional[dict] = None) -> None:
        """
        Update the document indexing status.
        """
        count = DatasetDocument.query.filter_by(id=document_id, is_paused=True).count()
        if count > 0:
            raise DocumentIsPausedException()
        document = DatasetDocument.query.filter_by(id=document_id).first()
        if not document:
            raise DocumentIsDeletedPausedException()

        update_params = {
            DatasetDocument.indexing_status: after_indexing_status
        }

        if extra_update_params:
            update_params.update(extra_update_params)

        DatasetDocument.query.filter_by(id=document_id).update(update_params)
        db.session.commit()

    @staticmethod
    def _update_segments_by_document(dataset_document_id: str, update_params: dict) -> None:
        """
        Update the document segment by document id.
        """
        DocumentSegment.query.filter_by(document_id=dataset_document_id).update(update_params)
        db.session.commit()

    @staticmethod
    def batch_add_segments(segments: list[DocumentSegment], dataset: Dataset):
        """
        Batch add segments index processing
        """
        documents = []
        for segment in segments:
            document = Document(
                page_content=segment.content,
                metadata={
                    "doc_id": segment.index_node_id,
                    "doc_hash": segment.index_node_hash,
                    "document_id": segment.document_id,
                    "dataset_id": segment.dataset_id,
                }
            )
            documents.append(document)
        # save vector index
        index_type = dataset.doc_form
        index_processor = IndexProcessorFactory(index_type).init_index_processor()
        index_processor.load(dataset, documents)

    def _transform(self, index_processor: BaseIndexProcessor, dataset: Dataset,
                   text_docs: list[Document], doc_language: str, process_rule: dict) -> list[Document]:
        """
        对提取的文档进行转换，包括生成嵌入向量等操作。

        :param index_processor: 索引处理器实例
        :param dataset: 数据集实例
        :param text_docs: 提取后的文档列表
        :param doc_language: 文档的语言
        :param process_rule: 处理规则字典
        :return: 转换后的文档列表
        此段代码定义了一个名为_transform的方法，其功能是对已经提取的文档进行进一步的转换处理，这通常涉及到生成文档的嵌入向量，
        以便后续可以进行更高效的检索和相似度计算。方法首先根据数据集的索引技术和配置，选择合适的嵌入模型实例，
        然后调用index_processor的transform方法，传入提取后的文档、嵌入模型实例、处理规则、租户ID和文档语言等参数，
        完成文档的转换。最终返回转换后的文档列表。
        """
        # 获取嵌入模型实例
        embedding_model_instance = None
        if dataset.indexing_technique == 'high_quality':  # 高质量索引技术
            if dataset.embedding_model_provider:  # 如果指定了嵌入模型提供商
                embedding_model_instance = self.model_manager.get_model_instance(
                    tenant_id=dataset.tenant_id,  # 租户ID
                    provider=dataset.embedding_model_provider,  # 提供商名称
                    model_type=ModelType.TEXT_EMBEDDING,  # 模型类型：文本嵌入
                    model=dataset.embedding_model  # 嵌入模型名称
                )
            else:
                # 如果没有指定提供商，则获取默认的嵌入模型实例
                embedding_model_instance = self.model_manager.get_default_model_instance(
                    tenant_id=dataset.tenant_id,  # 租户ID
                    model_type=ModelType.TEXT_EMBEDDING,  # 模型类型：文本嵌入
                )
        # 使用索引处理器进行文档转换
        documents = index_processor.transform(text_docs, embedding_model_instance=embedding_model_instance,
                                              process_rule=process_rule, tenant_id=dataset.tenant_id,
                                              doc_language=doc_language)

        return documents

    def _load_segments(self, dataset, dataset_document, documents):
        # 创建一个DatasetDocumentStore实例，用于存储文档片段
        doc_store = DatasetDocumentStore(
            dataset=dataset,
            user_id=dataset_document.created_by,
            document_id=dataset_document.id
        )

        # 将文档片段添加到文档存储中
        doc_store.add_documents(documents)

        # 获取当前时间，用于记录文档处理的完成时间
        cur_time = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        # 更新文档的状态至“索引中”，并记录文档的清洗和分割完成时间
        self._update_document_index_status(
            document_id=dataset_document.id,
            after_indexing_status="indexing",
            extra_update_params={
                DatasetDocument.cleaning_completed_at: cur_time,
                DatasetDocument.splitting_completed_at: cur_time,
            }
        )

        ## 更新文档片段的状态至“索引中”，并记录片段的索引开始时间
        self._update_segments_by_document(
            dataset_document_id=dataset_document.id,
            update_params={
                DocumentSegment.status: "indexing",
                DocumentSegment.indexing_at: datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
            }
        )
        pass


class DocumentIsPausedException(Exception):
    pass


class DocumentIsDeletedPausedException(Exception):
    pass
