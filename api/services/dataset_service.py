import datetime
import json
import logging
import random
import time
import uuid
from typing import Optional

from flask_login import current_user
from sqlalchemy import func

from configs import dify_config
from core.errors.error import LLMBadRequestError, ProviderTokenNotInitError
from core.model_manager import ModelManager
from core.model_runtime.entities.model_entities import ModelType
from core.rag.datasource.keyword.keyword_factory import Keyword
from core.rag.models.document import Document as RAGDocument
from core.rag.retrieval.retrival_methods import RetrievalMethod
from events.dataset_event import dataset_was_deleted
from events.document_event import document_was_deleted
from extensions.ext_database import db
from extensions.ext_redis import redis_client
from libs import helper
from models.account import Account, TenantAccountRole
from models.dataset import (
    AppDatasetJoin,
    Dataset,
    DatasetCollectionBinding,
    DatasetPermission,
    DatasetPermissionEnum,
    DatasetProcessRule,
    DatasetQuery,
    Document,
    DocumentSegment,
)
from models.model import UploadFile
from models.source import DataSourceOauthBinding
from services.errors.account import NoPermissionError
from services.errors.dataset import DatasetNameDuplicateError
from services.errors.document import DocumentIndexingError
from services.errors.file import FileNotExistsError
from services.feature_service import FeatureModel, FeatureService
from services.tag_service import TagService
from services.vector_service import VectorService
from tasks.clean_notion_document_task import clean_notion_document_task
from tasks.deal_dataset_vector_index_task import deal_dataset_vector_index_task
from tasks.delete_segment_from_index_task import delete_segment_from_index_task
from tasks.disable_segment_from_index_task import disable_segment_from_index_task
from tasks.document_indexing_task import document_indexing_task
from tasks.document_indexing_update_task import document_indexing_update_task
from tasks.duplicate_document_indexing_task import duplicate_document_indexing_task
from tasks.recover_document_indexing_task import recover_document_indexing_task
from tasks.retry_document_indexing_task import retry_document_indexing_task
from tasks.sync_website_document_indexing_task import sync_website_document_indexing_task


class DatasetService:
    @staticmethod
    def get_datasets(page, per_page, provider="vendor", tenant_id=None, user=None, search=None, tag_ids=None):
        query = Dataset.query.filter(Dataset.provider == provider, Dataset.tenant_id == tenant_id).order_by(
            Dataset.created_at.desc()
        )

        if user:
            # get permitted dataset ids
            dataset_permission = DatasetPermission.query.filter_by(account_id=user.id, tenant_id=tenant_id).all()
            permitted_dataset_ids = {dp.dataset_id for dp in dataset_permission} if dataset_permission else None

            if user.current_role == TenantAccountRole.DATASET_OPERATOR:
                # only show datasets that the user has permission to access
                if permitted_dataset_ids:
                    query = query.filter(Dataset.id.in_(permitted_dataset_ids))
                else:
                    return [], 0
            else:
                # show all datasets that the user has permission to access
                if permitted_dataset_ids:
                    query = query.filter(
                        db.or_(
                            Dataset.permission == DatasetPermissionEnum.ALL_TEAM,
                            db.and_(Dataset.permission == DatasetPermissionEnum.ONLY_ME, Dataset.created_by == user.id),
                            db.and_(
                                Dataset.permission == DatasetPermissionEnum.PARTIAL_TEAM,
                                Dataset.id.in_(permitted_dataset_ids),
                            ),
                        )
                    )
                else:
                    query = query.filter(
                        db.or_(
                            Dataset.permission == DatasetPermissionEnum.ALL_TEAM,
                            db.and_(Dataset.permission == DatasetPermissionEnum.ONLY_ME, Dataset.created_by == user.id),
                        )
                    )
        else:
            # if no user, only show datasets that are shared with all team members
            query = query.filter(Dataset.permission == DatasetPermissionEnum.ALL_TEAM)

        if search:
            query = query.filter(Dataset.name.ilike(f"%{search}%"))

        if tag_ids:
            target_ids = TagService.get_target_ids_by_tag_ids("knowledge", tenant_id, tag_ids)
            if target_ids:
                query = query.filter(Dataset.id.in_(target_ids))
            else:
                return [], 0

        datasets = query.paginate(page=page, per_page=per_page, max_per_page=100, error_out=False)

        return datasets.items, datasets.total

    @staticmethod
    def get_process_rules(dataset_id):
        # get the latest process rule
        dataset_process_rule = (
            db.session.query(DatasetProcessRule)
            .filter(DatasetProcessRule.dataset_id == dataset_id)
            .order_by(DatasetProcessRule.created_at.desc())
            .limit(1)
            .one_or_none()
        )
        if dataset_process_rule:
            mode = dataset_process_rule.mode
            rules = dataset_process_rule.rules_dict
        else:
            mode = DocumentService.DEFAULT_RULES["mode"]
            rules = DocumentService.DEFAULT_RULES["rules"]
        return {"mode": mode, "rules": rules}

    @staticmethod
    def get_datasets_by_ids(ids, tenant_id):
        datasets = Dataset.query.filter(Dataset.id.in_(ids), Dataset.tenant_id == tenant_id).paginate(
            page=1, per_page=len(ids), max_per_page=len(ids), error_out=False
        )
        return datasets.items, datasets.total

    @staticmethod
    def create_empty_dataset(
        tenant_id: str, name: str, indexing_technique: Optional[str], account: Account, permission: Optional[str] = None
    ):
        # check if dataset name already exists
        if Dataset.query.filter_by(name=name, tenant_id=tenant_id).first():
            raise DatasetNameDuplicateError(f"Dataset with name {name} already exists.")
        embedding_model = None
        if indexing_technique == "high_quality":
            model_manager = ModelManager()
            embedding_model = model_manager.get_default_model_instance(
                tenant_id=tenant_id, model_type=ModelType.TEXT_EMBEDDING
            )
        dataset = Dataset(name=name, indexing_technique=indexing_technique)
        # dataset = Dataset(name=name, provider=provider, config=config)
        dataset.created_by = account.id
        dataset.updated_by = account.id
        dataset.tenant_id = tenant_id
        dataset.embedding_model_provider = embedding_model.provider if embedding_model else None
        dataset.embedding_model = embedding_model.model if embedding_model else None
        dataset.permission = permission if permission else DatasetPermissionEnum.ONLY_ME
        db.session.add(dataset)
        db.session.commit()
        return dataset

    @staticmethod
    def get_dataset(dataset_id):
        return Dataset.query.filter_by(id=dataset_id).first()

    @staticmethod
    def check_dataset_model_setting(dataset):
        if dataset.indexing_technique == "high_quality":
            try:
                model_manager = ModelManager()
                model_manager.get_model_instance(
                    tenant_id=dataset.tenant_id,
                    provider=dataset.embedding_model_provider,
                    model_type=ModelType.TEXT_EMBEDDING,
                    model=dataset.embedding_model,
                )
            except LLMBadRequestError:
                raise ValueError(
                    "No Embedding Model available. Please configure a valid provider "
                    "in the Settings -> Model Provider."
                )
            except ProviderTokenNotInitError as ex:
                raise ValueError(f"The dataset in unavailable, due to: " f"{ex.description}")

    @staticmethod
    def check_embedding_model_setting(tenant_id: str, embedding_model_provider: str, embedding_model: str):
        try:
            model_manager = ModelManager()
            model_manager.get_model_instance(
                tenant_id=tenant_id,
                provider=embedding_model_provider,
                model_type=ModelType.TEXT_EMBEDDING,
                model=embedding_model,
            )
        except LLMBadRequestError:
            raise ValueError(
                "No Embedding Model available. Please configure a valid provider " "in the Settings -> Model Provider."
            )
        except ProviderTokenNotInitError as ex:
            raise ValueError(f"The dataset in unavailable, due to: " f"{ex.description}")

    @staticmethod
    def update_dataset(dataset_id, data, user):
        # 清理数据中的'partial_member_list'键，如果存在
        data.pop('partial_member_list', None)
        # 过滤data字典，移除值为None的键，除了'description'键
        filtered_data = {k: v for k, v in data.items() if v is not None or k == 'description'}
        # 从DatasetService获取指定ID的数据集
        dataset = DatasetService.get_dataset(dataset_id)
        # 检查用户是否有权限操作此数据集
        DatasetService.check_dataset_permission(dataset, user)
        # 初始化action变量
        action = None
        # 检查indexing_technique是否改变
        if dataset.indexing_technique != data['indexing_technique']:
            # 如果切换到'economy'模式
            if data['indexing_technique'] == 'economy':
                action = 'remove'
                # 清空与embedding相关的字段
                filtered_data['embedding_model'] = None
                filtered_data['embedding_model_provider'] = None
                filtered_data['collection_binding_id'] = None
            # 如果切换到'high_quality'模式
            elif data['indexing_technique'] == 'high_quality':
                action = 'add'
                # 获取embedding模型设置
                try:
                    model_manager = ModelManager()
                    embedding_model = model_manager.get_model_instance(
                        tenant_id=current_user.current_tenant_id,
                        provider=data["embedding_model_provider"],
                        model_type=ModelType.TEXT_EMBEDDING,
                        model=data["embedding_model"],
                    )
                    # 更新filtered_data字典中的embedding相关字段
                    filtered_data['embedding_model'] = embedding_model.model
                    filtered_data['embedding_model_provider'] = embedding_model.provider
                    dataset_collection_binding = DatasetCollectionBindingService.get_dataset_collection_binding(
                        embedding_model.provider, embedding_model.model
                    )
                    filtered_data["collection_binding_id"] = dataset_collection_binding.id
                except LLMBadRequestError:
                    raise ValueError(
                        "No Embedding Model available. Please configure a valid provider "
                        "in the Settings -> Model Provider."
                    )
                except ProviderTokenNotInitError as ex:
                    raise ValueError(ex.description)
        # 如果indexing_technique未改变，但embedding模型或提供商改变
        else:
            if (
                data["embedding_model_provider"] != dataset.embedding_model_provider
                or data["embedding_model"] != dataset.embedding_model
            ):
                action = "update"
                try:
                    # 更新embedding模型设置
                    model_manager = ModelManager()
                    embedding_model = model_manager.get_model_instance(
                        tenant_id=current_user.current_tenant_id,
                        provider=data["embedding_model_provider"],
                        model_type=ModelType.TEXT_EMBEDDING,
                        model=data["embedding_model"],
                    )
                    # 更新filtered_data字典中的embedding相关字段
                    filtered_data['embedding_model'] = embedding_model.model
                    filtered_data['embedding_model_provider'] = embedding_model.provider
                    dataset_collection_binding = DatasetCollectionBindingService.get_dataset_collection_binding(
                        embedding_model.provider, embedding_model.model
                    )
                    filtered_data["collection_binding_id"] = dataset_collection_binding.id
                except LLMBadRequestError:
                    raise ValueError(
                        "No Embedding Model available. Please configure a valid provider "
                        "in the Settings -> Model Provider."
                    )
                except ProviderTokenNotInitError as ex:
                    raise ValueError(ex.description)
        # 更新更新者和更新时间
        filtered_data['updated_by'] = user.id
        filtered_data['updated_at'] = datetime.datetime.now()

        # 更新Retrieval model
        filtered_data['retrieval_model'] = data['retrieval_model']
        # 使用过滤后的数据更新数据集
        dataset.query.filter_by(id=dataset_id).update(filtered_data)

        db.session.commit()
        # 如果有action，则异步处理向量索引任务
        if action:
            deal_dataset_vector_index_task.delay(dataset_id, action)
        return dataset

    @staticmethod
    def delete_dataset(dataset_id, user):
        dataset = DatasetService.get_dataset(dataset_id)

        if dataset is None:
            return False

        DatasetService.check_dataset_permission(dataset, user)

        dataset_was_deleted.send(dataset)

        db.session.delete(dataset)
        db.session.commit()
        return True

    @staticmethod
    def dataset_use_check(dataset_id) -> bool:
        count = AppDatasetJoin.query.filter_by(dataset_id=dataset_id).count()
        if count > 0:
            return True
        return False

    @staticmethod
    def check_dataset_permission(dataset, user):
        if dataset.tenant_id != user.current_tenant_id:
            logging.debug(f"User {user.id} does not have permission to access dataset {dataset.id}")
            raise NoPermissionError("You do not have permission to access this dataset.")
        if dataset.permission == DatasetPermissionEnum.ONLY_ME and dataset.created_by != user.id:
            logging.debug(f"User {user.id} does not have permission to access dataset {dataset.id}")
            raise NoPermissionError("You do not have permission to access this dataset.")
        if dataset.permission == "partial_members":
            user_permission = DatasetPermission.query.filter_by(dataset_id=dataset.id, account_id=user.id).first()
            if not user_permission and dataset.tenant_id != user.current_tenant_id and dataset.created_by != user.id:
                logging.debug(f"User {user.id} does not have permission to access dataset {dataset.id}")
                raise NoPermissionError("You do not have permission to access this dataset.")

    @staticmethod
    def check_dataset_operator_permission(user: Account = None, dataset: Dataset = None):
        if dataset.permission == DatasetPermissionEnum.ONLY_ME:
            if dataset.created_by != user.id:
                raise NoPermissionError("You do not have permission to access this dataset.")

        elif dataset.permission == DatasetPermissionEnum.PARTIAL_TEAM:
            if not any(
                dp.dataset_id == dataset.id for dp in DatasetPermission.query.filter_by(account_id=user.id).all()
            ):
                raise NoPermissionError("You do not have permission to access this dataset.")

    @staticmethod
    def get_dataset_queries(dataset_id: str, page: int, per_page: int):
        dataset_queries = (
            DatasetQuery.query.filter_by(dataset_id=dataset_id)
            .order_by(db.desc(DatasetQuery.created_at))
            .paginate(page=page, per_page=per_page, max_per_page=100, error_out=False)
        )
        return dataset_queries.items, dataset_queries.total

    @staticmethod
    def get_related_apps(dataset_id: str):
        return (
            AppDatasetJoin.query.filter(AppDatasetJoin.dataset_id == dataset_id)
            .order_by(db.desc(AppDatasetJoin.created_at))
            .all()
        )


class DocumentService:
    DEFAULT_RULES = {
        "mode": "custom",
        "rules": {
            "pre_processing_rules": [
                {"id": "remove_extra_spaces", "enabled": True},
                {"id": "remove_urls_emails", "enabled": False},
            ],
            "segmentation": {"delimiter": "\n", "max_tokens": 500, "chunk_overlap": 50},
        },
    }

    DOCUMENT_METADATA_SCHEMA = {
        "book": {
            "title": str,
            "language": str,
            "author": str,
            "publisher": str,
            "publication_date": str,
            "isbn": str,
            "category": str,
        },
        "web_page": {
            "title": str,
            "url": str,
            "language": str,
            "publish_date": str,
            "author/publisher": str,
            "topic/keywords": str,
            "description": str,
        },
        "paper": {
            "title": str,
            "language": str,
            "author": str,
            "publish_date": str,
            "journal/conference_name": str,
            "volume/issue/page_numbers": str,
            "doi": str,
            "topic/keywords": str,
            "abstract": str,
        },
        "social_media_post": {
            "platform": str,
            "author/username": str,
            "publish_date": str,
            "post_url": str,
            "topic/tags": str,
        },
        "wikipedia_entry": {
            "title": str,
            "language": str,
            "web_page_url": str,
            "last_edit_date": str,
            "editor/contributor": str,
            "summary/introduction": str,
        },
        "personal_document": {
            "title": str,
            "author": str,
            "creation_date": str,
            "last_modified_date": str,
            "document_type": str,
            "tags/category": str,
        },
        "business_document": {
            "title": str,
            "author": str,
            "creation_date": str,
            "last_modified_date": str,
            "document_type": str,
            "department/team": str,
        },
        "im_chat_log": {
            "chat_platform": str,
            "chat_participants/group_name": str,
            "start_date": str,
            "end_date": str,
            "summary": str,
        },
        "synced_from_notion": {
            "title": str,
            "language": str,
            "author/creator": str,
            "creation_date": str,
            "last_modified_date": str,
            "notion_page_link": str,
            "category/tags": str,
            "description": str,
        },
        "synced_from_github": {
            "repository_name": str,
            "repository_description": str,
            "repository_owner/organization": str,
            "code_filename": str,
            "code_file_path": str,
            "programming_language": str,
            "github_link": str,
            "open_source_license": str,
            "commit_date": str,
            "commit_author": str,
        },
        "others": dict,
    }

    @staticmethod
    def get_document(dataset_id: str, document_id: str) -> Optional[Document]:
        document = (
            db.session.query(Document).filter(Document.id == document_id, Document.dataset_id == dataset_id).first()
        )

        return document

    @staticmethod
    def get_document_by_id(document_id: str) -> Optional[Document]:
        document = db.session.query(Document).filter(Document.id == document_id).first()

        return document

    @staticmethod
    def get_document_by_dataset_id(dataset_id: str) -> list[Document]:
        documents = db.session.query(Document).filter(Document.dataset_id == dataset_id, Document.enabled == True).all()

        return documents

    @staticmethod
    def get_error_documents_by_dataset_id(dataset_id: str) -> list[Document]:
        documents = (
            db.session.query(Document)
            .filter(Document.dataset_id == dataset_id, Document.indexing_status.in_(["error", "paused"]))
            .all()
        )
        return documents

    @staticmethod
    def get_batch_documents(dataset_id: str, batch: str) -> list[Document]:
        documents = (
            db.session.query(Document)
            .filter(
                Document.batch == batch,
                Document.dataset_id == dataset_id,
                Document.tenant_id == current_user.current_tenant_id,
            )
            .all()
        )

        return documents

    @staticmethod
    def get_document_file_detail(file_id: str):
        file_detail = db.session.query(UploadFile).filter(UploadFile.id == file_id).one_or_none()
        return file_detail

    @staticmethod
    def check_archived(document):
        if document.archived:
            return True
        else:
            return False

    @staticmethod
    def delete_document(document):
        # trigger document_was_deleted signal
        file_id = None
        if document.data_source_type == "upload_file":
            if document.data_source_info:
                data_source_info = document.data_source_info_dict
                if data_source_info and "upload_file_id" in data_source_info:
                    file_id = data_source_info["upload_file_id"]
        document_was_deleted.send(
            document.id, dataset_id=document.dataset_id, doc_form=document.doc_form, file_id=file_id
        )

        db.session.delete(document)
        db.session.commit()

    @staticmethod
    def rename_document(dataset_id: str, document_id: str, name: str) -> Document:
        dataset = DatasetService.get_dataset(dataset_id)
        if not dataset:
            raise ValueError("Dataset not found.")

        document = DocumentService.get_document(dataset_id, document_id)

        if not document:
            raise ValueError("Document not found.")

        if document.tenant_id != current_user.current_tenant_id:
            raise ValueError("No permission.")

        document.name = name

        db.session.add(document)
        db.session.commit()

        return document

    @staticmethod
    def pause_document(document):
        if document.indexing_status not in ["waiting", "parsing", "cleaning", "splitting", "indexing"]:
            raise DocumentIndexingError()
        # update document to be paused
        document.is_paused = True
        document.paused_by = current_user.id
        document.paused_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)

        db.session.add(document)
        db.session.commit()
        # set document paused flag
        indexing_cache_key = "document_{}_is_paused".format(document.id)
        redis_client.setnx(indexing_cache_key, "True")

    @staticmethod
    def recover_document(document):
        if not document.is_paused:
            raise DocumentIndexingError()
        # update document to be recover
        document.is_paused = False
        document.paused_by = None
        document.paused_at = None

        db.session.add(document)
        db.session.commit()
        # delete paused flag
        indexing_cache_key = "document_{}_is_paused".format(document.id)
        redis_client.delete(indexing_cache_key)
        # trigger async task
        recover_document_indexing_task.delay(document.dataset_id, document.id)

    @staticmethod
    def retry_document(dataset_id: str, documents: list[Document]):
        for document in documents:
            # add retry flag
            retry_indexing_cache_key = "document_{}_is_retried".format(document.id)
            cache_result = redis_client.get(retry_indexing_cache_key)
            if cache_result is not None:
                raise ValueError("Document is being retried, please try again later")
            # retry document indexing
            document.indexing_status = "waiting"
            db.session.add(document)
            db.session.commit()

            redis_client.setex(retry_indexing_cache_key, 600, 1)
        # trigger async task
        document_ids = [document.id for document in documents]
        retry_document_indexing_task.delay(dataset_id, document_ids)

    @staticmethod
    def sync_website_document(dataset_id: str, document: Document):
        # add sync flag
        sync_indexing_cache_key = "document_{}_is_sync".format(document.id)
        cache_result = redis_client.get(sync_indexing_cache_key)
        if cache_result is not None:
            raise ValueError("Document is being synced, please try again later")
        # sync document indexing
        document.indexing_status = "waiting"
        data_source_info = document.data_source_info_dict
        data_source_info["mode"] = "scrape"
        document.data_source_info = json.dumps(data_source_info, ensure_ascii=False)
        db.session.add(document)
        db.session.commit()

        redis_client.setex(sync_indexing_cache_key, 600, 1)

        sync_website_document_indexing_task.delay(dataset_id, document.id)

    @staticmethod
    def get_documents_position(dataset_id):
        document = Document.query.filter_by(dataset_id=dataset_id).order_by(Document.position.desc()).first()
        if document:
            return document.position + 1
        else:
            return 1

    @staticmethod
    def save_document_with_dataset_id(
        dataset: Dataset,
        document_data: dict,
        account: Account,
        dataset_process_rule: Optional[DatasetProcessRule] = None,
        created_from: str = "web",
    ):
        """
           保存文档到指定数据集，并处理各种数据源类型，包括上传文件、Notion导入和网站抓取。

           :param dataset: 数据集对象。
           :param document_data: 包含文档数据和元数据的字典。
           :param account: 用户账户对象。
           :param dataset_process_rule: 可选的数据集处理规则对象。
           :param created_from: 创建文档的来源，例如 'web'。
           :return: 一个包含已保存文档的列表和批次标识符的元组。
           """
        # 验证并处理文档数量限制
        features = FeatureService.get_features(current_user.current_tenant_id)

        if features.billing.enabled:
            if "original_document_id" not in document_data or not document_data["original_document_id"]:
                count = 0
                if document_data["data_source"]["type"] == "upload_file":
                    upload_file_list = document_data["data_source"]["info_list"]["file_info_list"]["file_ids"]
                    count = len(upload_file_list)
                elif document_data["data_source"]["type"] == "notion_import":
                    notion_info_list = document_data["data_source"]["info_list"]["notion_info_list"]
                    for notion_info in notion_info_list:
                        count = count + len(notion_info["pages"])
                elif document_data["data_source"]["type"] == "website_crawl":
                    website_info = document_data["data_source"]["info_list"]["website_info_list"]
                    count = len(website_info["urls"])
                batch_upload_limit = int(dify_config.BATCH_UPLOAD_LIMIT)
                if count > batch_upload_limit:
                    raise ValueError(f"You have reached the batch upload limit of {batch_upload_limit}.")

                DocumentService.check_documents_upload_quota(count, features)

        #  # 更新数据集的数据源类型
        if not dataset.data_source_type:
            dataset.data_source_type = document_data["data_source"]["type"]
        # 设置索引技术
        if not dataset.indexing_technique:
            if (
                "indexing_technique" not in document_data
                or document_data["indexing_technique"] not in Dataset.INDEXING_TECHNIQUE_LIST
            ):
                raise ValueError("Indexing technique is required")

            dataset.indexing_technique = document_data["indexing_technique"]
            if document_data["indexing_technique"] == 'high_quality':
                # 获取模型管理器实例
                model_manager = ModelManager()
                # 获取默认的文本嵌入模型实例
                embedding_model = model_manager.get_default_model_instance(
                    tenant_id=current_user.current_tenant_id, model_type=ModelType.TEXT_EMBEDDING
                )
                # 设置数据集的嵌入模型和提供商信息
                dataset.embedding_model = embedding_model.model
                dataset.embedding_model_provider = embedding_model.provider
                # 获取与嵌入模型绑定的数据集集合绑定
                dataset_collection_binding = DatasetCollectionBindingService.get_dataset_collection_binding(
                    embedding_model.provider, embedding_model.model
                )
                # 设置数据集的集合绑定ID
                dataset.collection_binding_id = dataset_collection_binding.id
                # 如果数据集的检索模型未设置，则设置默认的检索模型配置
                if not dataset.retrieval_model:
                    default_retrieval_model = {
                        "search_method": RetrievalMethod.SEMANTIC_SEARCH.value,
                        "reranking_enable": False,
                        "reranking_model": {"reranking_provider_name": "", "reranking_model_name": ""},
                        "top_k": 2,
                        "score_threshold_enabled": False,
                    }
                    # 使用文档数据中的检索模型配置覆盖默认配置，如果存在的话
                    dataset.retrieval_model = document_data.get('retrieval_model') if document_data.get(
                        'retrieval_model'
                    ) else default_retrieval_model
        # 初始化文档列表和批次标识符
        documents = []
        batch = time.strftime('%Y%m%d%H%M%S') + str(random.randint(100000, 999999))

        # 根据document_data是否存在"original_document_id"字段判断是否更新已有文档
        if document_data.get("original_document_id"):
            document = DocumentService.update_document_with_dataset_id(dataset, document_data, account)
            documents.append(document)
        else:
            #如果没有，则需要创建或者更新数据集规则
            if not dataset_process_rule:
                process_rule = document_data["process_rule"]

                # 根据process_rule的模式创建DatasetProcessRule实例
                if process_rule["mode"] == "custom":
                    dataset_process_rule = DatasetProcessRule(
                        dataset_id=dataset.id,
                        mode=process_rule["mode"],
                        rules=json.dumps(process_rule["rules"]),
                        created_by=account.id,
                    )
                elif process_rule["mode"] == "automatic":
                    dataset_process_rule = DatasetProcessRule(
                        dataset_id=dataset.id,
                        mode=process_rule["mode"],
                        rules=json.dumps(DatasetProcessRule.AUTOMATIC_RULES),
                        created_by=account.id,
                    )
                # 将新创建的DatasetProcessRule添加到数据库会话中
                db.session.add(dataset_process_rule)
                db.session.commit()
            # 获取数据集中文档的位置信息
            position = DocumentService.get_documents_position(dataset.id)
            document_ids = []
            duplicate_document_ids = []
            if document_data["data_source"]["type"] == "upload_file":
                # 获取文件ID列表
                upload_file_list = document_data["data_source"]["info_list"]['file_info_list']['file_ids']
                for file_id in upload_file_list:
                    # 查询文件信息
                    file = db.session.query(UploadFile).filter(
                        UploadFile.tenant_id == dataset.tenant_id,
                        UploadFile.id == file_id
                    ).first()

                    # raise error if file not found
                    if not file:
                        raise FileNotExistsError()
                    # 文件名和数据源信息
                    file_name = file.name
                    data_source_info = {
                        "upload_file_id": file_id,
                    }
                    # # 检查是否允许导入重复文档
                    if document_data.get('duplicate', False):
                        document = Document.query.filter_by(
                            dataset_id=dataset.id,
                            tenant_id=current_user.current_tenant_id,
                            data_source_type="upload_file",
                            enabled=True,
                            name=file_name,
                        ).first()
                        if document:
                            # 更新现有文档
                            document.dataset_process_rule_id = dataset_process_rule.id
                            document.updated_at = datetime.datetime.utcnow()
                            document.created_from = created_from
                            document.doc_form = document_data["doc_form"]
                            document.doc_language = document_data["doc_language"]
                            document.data_source_info = json.dumps(data_source_info)
                            document.batch = batch
                            document.indexing_status = "waiting"
                            db.session.add(document)
                            documents.append(document)
                            duplicate_document_ids.append(document.id)
                            continue
                    # 创建新文档
                    document = DocumentService.build_document(
                        dataset,
                        dataset_process_rule.id,
                        document_data["data_source"]["type"],
                        document_data["doc_form"],
                        document_data["doc_language"],
                        data_source_info,
                        created_from,
                        position,
                        account,
                        file_name,
                        batch,
                    )
                    db.session.add(document)
                    db.session.flush()
                    document_ids.append(document.id)
                    documents.append(document)
                    position += 1
            # 处理Notion导入数据源
            elif document_data["data_source"]["type"] == "notion_import":
                # 获取Notion信息列表
                notion_info_list = document_data["data_source"]['info_list']['notion_info_list']
                # 初始化已存在的Notion页面ID列表
                exist_page_ids = []
                # 初始化已存在的文档字典，键为Notion页面ID，值为文档ID
                exist_document = {}
                # 查询已存在的Notion导入类型的文档
                documents = Document.query.filter_by(
                    dataset_id=dataset.id,
                    tenant_id=current_user.current_tenant_id,
                    data_source_type="notion_import",
                    enabled=True,
                ).all()
                if documents:
                    for document in documents:
                        # 解析数据源信息
                        data_source_info = json.loads(document.data_source_info)
                        exist_page_ids.append(data_source_info['notion_page_id'])
                        exist_document[data_source_info['notion_page_id']] = document.id
                # 遍历Notion信息列表
                for notion_info in notion_info_list:
                    # 获取工作空间ID
                    workspace_id = notion_info['workspace_id']
                    # 查询数据源绑定信息
                    data_source_binding = DataSourceOauthBinding.query.filter(
                        db.and_(
                            DataSourceOauthBinding.tenant_id == current_user.current_tenant_id,
                            DataSourceOauthBinding.provider == "notion",
                            DataSourceOauthBinding.disabled == False,
                            DataSourceOauthBinding.source_info["workspace_id"] == f'"{workspace_id}"',
                        )
                    ).first()
                    if not data_source_binding:
                        # 如果数据源绑定不存在，抛出错误
                        raise ValueError("Data source binding not found.")
                    # 遍历Notion页面
                    for page in notion_info["pages"]:
                        # 如果页面ID不在已存在的页面ID列表中
                        if page["page_id"] not in exist_page_ids:
                            # 创建数据源信息字典
                            data_source_info = {
                                "notion_workspace_id": workspace_id,
                                "notion_page_id": page["page_id"],
                                "notion_page_icon": page["page_icon"],
                                "type": page["type"],
                            }
                            # 创建新文档
                            document = DocumentService.build_document(
                                dataset,
                                dataset_process_rule.id,
                                document_data["data_source"]["type"],
                                document_data["doc_form"],
                                document_data["doc_language"],
                                data_source_info,
                                created_from,
                                position,
                                account,
                                page["page_name"],
                                batch,
                            )
                            db.session.add(document)
                            db.session.flush()
                            document_ids.append(document.id)
                            documents.append(document)
                            position += 1
                        else:
                            exist_document.pop(page["page_id"])
                # # 删除未被选择的文档
                if len(exist_document) > 0:
                    clean_notion_document_task.delay(list(exist_document.values()), dataset.id)
            # 处理网站抓取数据源
            elif document_data["data_source"]["type"] == "website_crawl":
                # 获取网站信息
                website_info = document_data["data_source"]["info_list"]["website_info_list"]
                urls = website_info["urls"]
                # 遍历URL列表
                for url in urls:
                    data_source_info = {
                        "url": url,
                        "provider": website_info["provider"],
                        "job_id": website_info["job_id"],
                        "only_main_content": website_info.get("only_main_content", False),
                        "mode": "crawl",
                    }
                    if len(url) > 255:
                        document_name = url[:200] + "..."
                    else:
                        document_name = url
                    # 创建新文档
                    document = DocumentService.build_document(
                        dataset,
                        dataset_process_rule.id,
                        document_data["data_source"]["type"],
                        document_data["doc_form"],
                        document_data["doc_language"],
                        data_source_info,
                        created_from,
                        position,
                        account,
                        document_name,
                        batch,
                    )
                    db.session.add(document)
                    db.session.flush()
                    document_ids.append(document.id)
                    documents.append(document)

                    position += 1
            # 提交数据库会话，确保所有更改被持久化
            db.session.commit()

            #  # 触发异步任务，对新创建或更新的文档进行索引
            if document_ids:
                document_indexing_task.delay(dataset.id, document_ids)
            if duplicate_document_ids:
                duplicate_document_indexing_task.delay(dataset.id, duplicate_document_ids)

        return documents, batch

    @staticmethod
    def check_documents_upload_quota(count: int, features: FeatureModel):
        can_upload_size = features.documents_upload_quota.limit - features.documents_upload_quota.size
        if count > can_upload_size:
            raise ValueError(
                f"You have reached the limit of your subscription. Only {can_upload_size} documents can be uploaded."
            )

    @staticmethod
    def build_document(
        dataset: Dataset,
        process_rule_id: str,
        data_source_type: str,
        document_form: str,
        document_language: str,
        data_source_info: dict,
        created_from: str,
        position: int,
        account: Account,
        name: str,
        batch: str,
    ):
        document = Document(
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            position=position,
            data_source_type=data_source_type,
            data_source_info=json.dumps(data_source_info),
            dataset_process_rule_id=process_rule_id,
            batch=batch,
            name=name,
            created_from=created_from,
            created_by=account.id,
            doc_form=document_form,
            doc_language=document_language,
        )
        return document

    @staticmethod
    def get_tenant_documents_count():
        documents_count = Document.query.filter(
            Document.completed_at.isnot(None),
            Document.enabled == True,
            Document.archived == False,
            Document.tenant_id == current_user.current_tenant_id,
        ).count()
        return documents_count

    @staticmethod
    def update_document_with_dataset_id(
        dataset: Dataset,
        document_data: dict,
        account: Account,
        dataset_process_rule: Optional[DatasetProcessRule] = None,
        created_from: str = "web",
    ):
        DatasetService.check_dataset_model_setting(dataset)
        document = DocumentService.get_document(dataset.id, document_data["original_document_id"])
        if document.display_status != "available":
            raise ValueError("Document is not available")
        # update document name
        if document_data.get("name"):
            document.name = document_data["name"]
        # save process rule
        if document_data.get("process_rule"):
            process_rule = document_data["process_rule"]
            if process_rule["mode"] == "custom":
                dataset_process_rule = DatasetProcessRule(
                    dataset_id=dataset.id,
                    mode=process_rule["mode"],
                    rules=json.dumps(process_rule["rules"]),
                    created_by=account.id,
                )
            elif process_rule["mode"] == "automatic":
                dataset_process_rule = DatasetProcessRule(
                    dataset_id=dataset.id,
                    mode=process_rule["mode"],
                    rules=json.dumps(DatasetProcessRule.AUTOMATIC_RULES),
                    created_by=account.id,
                )
            db.session.add(dataset_process_rule)
            db.session.commit()
            document.dataset_process_rule_id = dataset_process_rule.id
        # update document data source
        if document_data.get("data_source"):
            file_name = ""
            data_source_info = {}
            if document_data["data_source"]["type"] == "upload_file":
                upload_file_list = document_data["data_source"]["info_list"]["file_info_list"]["file_ids"]
                for file_id in upload_file_list:
                    file = (
                        db.session.query(UploadFile)
                        .filter(UploadFile.tenant_id == dataset.tenant_id, UploadFile.id == file_id)
                        .first()
                    )

                    # raise error if file not found
                    if not file:
                        raise FileNotExistsError()

                    file_name = file.name
                    data_source_info = {
                        "upload_file_id": file_id,
                    }
            elif document_data["data_source"]["type"] == "notion_import":
                notion_info_list = document_data["data_source"]["info_list"]["notion_info_list"]
                for notion_info in notion_info_list:
                    workspace_id = notion_info["workspace_id"]
                    data_source_binding = DataSourceOauthBinding.query.filter(
                        db.and_(
                            DataSourceOauthBinding.tenant_id == current_user.current_tenant_id,
                            DataSourceOauthBinding.provider == "notion",
                            DataSourceOauthBinding.disabled == False,
                            DataSourceOauthBinding.source_info["workspace_id"] == f'"{workspace_id}"',
                        )
                    ).first()
                    if not data_source_binding:
                        raise ValueError("Data source binding not found.")
                    for page in notion_info["pages"]:
                        data_source_info = {
                            "notion_workspace_id": workspace_id,
                            "notion_page_id": page["page_id"],
                            "notion_page_icon": page["page_icon"],
                            "type": page["type"],
                        }
            elif document_data["data_source"]["type"] == "website_crawl":
                website_info = document_data["data_source"]["info_list"]["website_info_list"]
                urls = website_info["urls"]
                for url in urls:
                    data_source_info = {
                        "url": url,
                        "provider": website_info["provider"],
                        "job_id": website_info["job_id"],
                        "only_main_content": website_info.get("only_main_content", False),
                        "mode": "crawl",
                    }
            document.data_source_type = document_data["data_source"]["type"]
            document.data_source_info = json.dumps(data_source_info)
            document.name = file_name
        # update document to be waiting
        document.indexing_status = "waiting"
        document.completed_at = None
        document.processing_started_at = None
        document.parsing_completed_at = None
        document.cleaning_completed_at = None
        document.splitting_completed_at = None
        document.updated_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        document.created_from = created_from
        document.doc_form = document_data["doc_form"]
        db.session.add(document)
        db.session.commit()
        # update document segment
        update_params = {DocumentSegment.status: "re_segment"}
        DocumentSegment.query.filter_by(document_id=document.id).update(update_params)
        db.session.commit()
        # trigger async task
        document_indexing_update_task.delay(document.dataset_id, document.id)
        return document

    @staticmethod
    def save_document_without_dataset_id(tenant_id: str, document_data: dict, account: Account):
        # 获取租户的特性配置
        features = FeatureService.get_features(current_user.current_tenant_id)

        # 如果账单功能已启用，则检查文档上传限制
        if features.billing.enabled:
            count = 0
            if document_data["data_source"]["type"] == "upload_file":
                upload_file_list = document_data["data_source"]["info_list"]["file_info_list"]["file_ids"]
                count = len(upload_file_list)
            elif document_data["data_source"]["type"] == "notion_import":
                notion_info_list = document_data["data_source"]["info_list"]["notion_info_list"]
                for notion_info in notion_info_list:
                    count = count + len(notion_info["pages"])
            elif document_data["data_source"]["type"] == "website_crawl":
                website_info = document_data["data_source"]["info_list"]["website_info_list"]
                count = len(website_info["urls"])
            # 检查是否超过批量上传限制
            batch_upload_limit = int(dify_config.BATCH_UPLOAD_LIMIT)
            if count > batch_upload_limit:
                raise ValueError(f"You have reached the batch upload limit of {batch_upload_limit}.")

            # 检查文档上传配额
            DocumentService.check_documents_upload_quota(count, features)
        # 初始化嵌入模型和数据集绑定信息
        embedding_model = None
        dataset_collection_binding_id = None
        retrieval_model = None
        # 如果索引技术要求高质量，则获取默认的嵌入模型实例
        if document_data["indexing_technique"] == "high_quality":
            model_manager = ModelManager()
            embedding_model = model_manager.get_default_model_instance(
                tenant_id=current_user.current_tenant_id, model_type=ModelType.TEXT_EMBEDDING
            )
            # 获取与嵌入模型关联的数据集集合绑定
            dataset_collection_binding = DatasetCollectionBindingService.get_dataset_collection_binding(
                embedding_model.provider, embedding_model.model
            )
            dataset_collection_binding_id = dataset_collection_binding.id
            # 设置检索模型，优先使用文档数据中的模型，否则使用默认模型
            if document_data.get("retrieval_model"):
                retrieval_model = document_data["retrieval_model"]
            else:
                default_retrieval_model = {
                    "search_method": RetrievalMethod.SEMANTIC_SEARCH.value,
                    "reranking_enable": False,
                    "reranking_model": {"reranking_provider_name": "", "reranking_model_name": ""},
                    "top_k": 2,
                    "score_threshold_enabled": False,
                }
                retrieval_model = default_retrieval_model
        #  # 创建数据集实例
        dataset = Dataset(
            tenant_id=tenant_id,
            name="",
            data_source_type=document_data["data_source"]["type"],
            indexing_technique=document_data["indexing_technique"],
            created_by=account.id,
            embedding_model=embedding_model.model if embedding_model else None,
            embedding_model_provider=embedding_model.provider if embedding_model else None,
            collection_binding_id=dataset_collection_binding_id,
            retrieval_model=retrieval_model,
        )
        # 将数据集添加到数据库会话并刷新以获取ID
        db.session.add(dataset)
        db.session.flush()

        # 保存文档并获取文档列表、批次信息
        documents, batch = DocumentService.save_document_with_dataset_id(dataset, document_data, account)
        # 截断文档名称以适应显示
        cut_length = 18
        cut_name = documents[0].name[:cut_length]
        dataset.name = cut_name + "..."

        # 更新数据集描述
        dataset.description = "useful for when you want to answer queries about the " + documents[0].name
        db.session.commit()

        # 返回数据集、文档列表和批次信息
        return dataset, documents, batch

    @classmethod
    def document_create_args_validate(cls, args: dict):
        if "original_document_id" not in args or not args["original_document_id"]:
            DocumentService.data_source_args_validate(args)
            DocumentService.process_rule_args_validate(args)
        else:
            if ("data_source" not in args and not args["data_source"]) and (
                "process_rule" not in args and not args["process_rule"]
            ):
                raise ValueError("Data source or Process rule is required")
            else:
                if args.get("data_source"):
                    DocumentService.data_source_args_validate(args)
                if args.get("process_rule"):
                    DocumentService.process_rule_args_validate(args)

    @classmethod
    def data_source_args_validate(cls, args: dict):
        if "data_source" not in args or not args["data_source"]:
            raise ValueError("Data source is required")

        if not isinstance(args["data_source"], dict):
            raise ValueError("Data source is invalid")

        if "type" not in args["data_source"] or not args["data_source"]["type"]:
            raise ValueError("Data source type is required")

        if args["data_source"]["type"] not in Document.DATA_SOURCES:
            raise ValueError("Data source type is invalid")

        if "info_list" not in args["data_source"] or not args["data_source"]["info_list"]:
            raise ValueError("Data source info is required")

        if args["data_source"]["type"] == "upload_file":
            if (
                "file_info_list" not in args["data_source"]["info_list"]
                or not args["data_source"]["info_list"]["file_info_list"]
            ):
                raise ValueError("File source info is required")
        if args["data_source"]["type"] == "notion_import":
            if (
                "notion_info_list" not in args["data_source"]["info_list"]
                or not args["data_source"]["info_list"]["notion_info_list"]
            ):
                raise ValueError("Notion source info is required")
        if args["data_source"]["type"] == "website_crawl":
            if (
                "website_info_list" not in args["data_source"]["info_list"]
                or not args["data_source"]["info_list"]["website_info_list"]
            ):
                raise ValueError("Website source info is required")

    @classmethod
    def process_rule_args_validate(cls, args: dict):
        if "process_rule" not in args or not args["process_rule"]:
            raise ValueError("Process rule is required")

        if not isinstance(args["process_rule"], dict):
            raise ValueError("Process rule is invalid")

        if "mode" not in args["process_rule"] or not args["process_rule"]["mode"]:
            raise ValueError("Process rule mode is required")

        if args["process_rule"]["mode"] not in DatasetProcessRule.MODES:
            raise ValueError("Process rule mode is invalid")

        if args["process_rule"]["mode"] == "automatic":
            args["process_rule"]["rules"] = {}
        else:
            if "rules" not in args["process_rule"] or not args["process_rule"]["rules"]:
                raise ValueError("Process rule rules is required")

            if not isinstance(args["process_rule"]["rules"], dict):
                raise ValueError("Process rule rules is invalid")

            if (
                "pre_processing_rules" not in args["process_rule"]["rules"]
                or args["process_rule"]["rules"]["pre_processing_rules"] is None
            ):
                raise ValueError("Process rule pre_processing_rules is required")

            if not isinstance(args["process_rule"]["rules"]["pre_processing_rules"], list):
                raise ValueError("Process rule pre_processing_rules is invalid")

            unique_pre_processing_rule_dicts = {}
            for pre_processing_rule in args["process_rule"]["rules"]["pre_processing_rules"]:
                if "id" not in pre_processing_rule or not pre_processing_rule["id"]:
                    raise ValueError("Process rule pre_processing_rules id is required")

                if pre_processing_rule["id"] not in DatasetProcessRule.PRE_PROCESSING_RULES:
                    raise ValueError("Process rule pre_processing_rules id is invalid")

                if "enabled" not in pre_processing_rule or pre_processing_rule["enabled"] is None:
                    raise ValueError("Process rule pre_processing_rules enabled is required")

                if not isinstance(pre_processing_rule["enabled"], bool):
                    raise ValueError("Process rule pre_processing_rules enabled is invalid")

                unique_pre_processing_rule_dicts[pre_processing_rule["id"]] = pre_processing_rule

            args["process_rule"]["rules"]["pre_processing_rules"] = list(unique_pre_processing_rule_dicts.values())

            if (
                "segmentation" not in args["process_rule"]["rules"]
                or args["process_rule"]["rules"]["segmentation"] is None
            ):
                raise ValueError("Process rule segmentation is required")

            if not isinstance(args["process_rule"]["rules"]["segmentation"], dict):
                raise ValueError("Process rule segmentation is invalid")

            if (
                "separator" not in args["process_rule"]["rules"]["segmentation"]
                or not args["process_rule"]["rules"]["segmentation"]["separator"]
            ):
                raise ValueError("Process rule segmentation separator is required")

            if not isinstance(args["process_rule"]["rules"]["segmentation"]["separator"], str):
                raise ValueError("Process rule segmentation separator is invalid")

            if (
                "max_tokens" not in args["process_rule"]["rules"]["segmentation"]
                or not args["process_rule"]["rules"]["segmentation"]["max_tokens"]
            ):
                raise ValueError("Process rule segmentation max_tokens is required")

            if not isinstance(args["process_rule"]["rules"]["segmentation"]["max_tokens"], int):
                raise ValueError("Process rule segmentation max_tokens is invalid")

    @classmethod
    def estimate_args_validate(cls, args: dict):
        if "info_list" not in args or not args["info_list"]:
            raise ValueError("Data source info is required")

        if not isinstance(args["info_list"], dict):
            raise ValueError("Data info is invalid")

        if "process_rule" not in args or not args["process_rule"]:
            raise ValueError("Process rule is required")

        if not isinstance(args["process_rule"], dict):
            raise ValueError("Process rule is invalid")

        if "mode" not in args["process_rule"] or not args["process_rule"]["mode"]:
            raise ValueError("Process rule mode is required")

        if args["process_rule"]["mode"] not in DatasetProcessRule.MODES:
            raise ValueError("Process rule mode is invalid")

        if args["process_rule"]["mode"] == "automatic":
            args["process_rule"]["rules"] = {}
        else:
            if "rules" not in args["process_rule"] or not args["process_rule"]["rules"]:
                raise ValueError("Process rule rules is required")

            if not isinstance(args["process_rule"]["rules"], dict):
                raise ValueError("Process rule rules is invalid")

            if (
                "pre_processing_rules" not in args["process_rule"]["rules"]
                or args["process_rule"]["rules"]["pre_processing_rules"] is None
            ):
                raise ValueError("Process rule pre_processing_rules is required")

            if not isinstance(args["process_rule"]["rules"]["pre_processing_rules"], list):
                raise ValueError("Process rule pre_processing_rules is invalid")

            unique_pre_processing_rule_dicts = {}
            for pre_processing_rule in args["process_rule"]["rules"]["pre_processing_rules"]:
                if "id" not in pre_processing_rule or not pre_processing_rule["id"]:
                    raise ValueError("Process rule pre_processing_rules id is required")

                if pre_processing_rule["id"] not in DatasetProcessRule.PRE_PROCESSING_RULES:
                    raise ValueError("Process rule pre_processing_rules id is invalid")

                if "enabled" not in pre_processing_rule or pre_processing_rule["enabled"] is None:
                    raise ValueError("Process rule pre_processing_rules enabled is required")

                if not isinstance(pre_processing_rule["enabled"], bool):
                    raise ValueError("Process rule pre_processing_rules enabled is invalid")

                unique_pre_processing_rule_dicts[pre_processing_rule["id"]] = pre_processing_rule

            args["process_rule"]["rules"]["pre_processing_rules"] = list(unique_pre_processing_rule_dicts.values())

            if (
                "segmentation" not in args["process_rule"]["rules"]
                or args["process_rule"]["rules"]["segmentation"] is None
            ):
                raise ValueError("Process rule segmentation is required")

            if not isinstance(args["process_rule"]["rules"]["segmentation"], dict):
                raise ValueError("Process rule segmentation is invalid")

            if (
                "separator" not in args["process_rule"]["rules"]["segmentation"]
                or not args["process_rule"]["rules"]["segmentation"]["separator"]
            ):
                raise ValueError("Process rule segmentation separator is required")

            if not isinstance(args["process_rule"]["rules"]["segmentation"]["separator"], str):
                raise ValueError("Process rule segmentation separator is invalid")

            if (
                "max_tokens" not in args["process_rule"]["rules"]["segmentation"]
                or not args["process_rule"]["rules"]["segmentation"]["max_tokens"]
            ):
                raise ValueError("Process rule segmentation max_tokens is required")

            if not isinstance(args["process_rule"]["rules"]["segmentation"]["max_tokens"], int):
                raise ValueError("Process rule segmentation max_tokens is invalid")


class SegmentService:
    @classmethod
    def segment_create_args_validate(cls, args: dict, document: Document):
        if document.doc_form == "qa_model":
            if "answer" not in args or not args["answer"]:
                raise ValueError("Answer is required")
            if not args["answer"].strip():
                raise ValueError("Answer is empty")
        if "content" not in args or not args["content"] or not args["content"].strip():
            raise ValueError("Content is empty")

    @classmethod
    def create_segment(cls, args: dict, document: Document, dataset: Dataset):
        content = args["content"]
        doc_id = str(uuid.uuid4())
        segment_hash = helper.generate_text_hash(content)
        tokens = 0
        if dataset.indexing_technique == "high_quality":
            model_manager = ModelManager()
            embedding_model = model_manager.get_model_instance(
                tenant_id=current_user.current_tenant_id,
                provider=dataset.embedding_model_provider,
                model_type=ModelType.TEXT_EMBEDDING,
                model=dataset.embedding_model,
            )
            # calc embedding use tokens
            tokens = embedding_model.get_text_embedding_num_tokens(texts=[content])
        lock_name = "add_segment_lock_document_id_{}".format(document.id)
        with redis_client.lock(lock_name, timeout=600):
            max_position = (
                db.session.query(func.max(DocumentSegment.position))
                .filter(DocumentSegment.document_id == document.id)
                .scalar()
            )
            segment_document = DocumentSegment(
                tenant_id=current_user.current_tenant_id,
                dataset_id=document.dataset_id,
                document_id=document.id,
                index_node_id=doc_id,
                index_node_hash=segment_hash,
                position=max_position + 1 if max_position else 1,
                content=content,
                word_count=len(content),
                tokens=tokens,
                status="completed",
                indexing_at=datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None),
                completed_at=datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None),
                created_by=current_user.id,
            )
            if document.doc_form == "qa_model":
                segment_document.answer = args["answer"]

            db.session.add(segment_document)
            db.session.commit()

            # save vector index
            try:
                VectorService.create_segments_vector([args["keywords"]], [segment_document], dataset)
            except Exception as e:
                logging.exception("create segment index failed")
                segment_document.enabled = False
                segment_document.disabled_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
                segment_document.status = "error"
                segment_document.error = str(e)
                db.session.commit()
            segment = db.session.query(DocumentSegment).filter(DocumentSegment.id == segment_document.id).first()
            return segment

    @classmethod
    def multi_create_segment(cls, segments: list, document: Document, dataset: Dataset):
        lock_name = "multi_add_segment_lock_document_id_{}".format(document.id)
        with redis_client.lock(lock_name, timeout=600):
            embedding_model = None
            if dataset.indexing_technique == "high_quality":
                model_manager = ModelManager()
                embedding_model = model_manager.get_model_instance(
                    tenant_id=current_user.current_tenant_id,
                    provider=dataset.embedding_model_provider,
                    model_type=ModelType.TEXT_EMBEDDING,
                    model=dataset.embedding_model,
                )
            max_position = (
                db.session.query(func.max(DocumentSegment.position))
                .filter(DocumentSegment.document_id == document.id)
                .scalar()
            )
            pre_segment_data_list = []
            segment_data_list = []
            keywords_list = []
            for segment_item in segments:
                content = segment_item["content"]
                doc_id = str(uuid.uuid4())
                segment_hash = helper.generate_text_hash(content)
                tokens = 0
                if dataset.indexing_technique == "high_quality" and embedding_model:
                    # calc embedding use tokens
                    tokens = embedding_model.get_text_embedding_num_tokens(texts=[content])
                segment_document = DocumentSegment(
                    tenant_id=current_user.current_tenant_id,
                    dataset_id=document.dataset_id,
                    document_id=document.id,
                    index_node_id=doc_id,
                    index_node_hash=segment_hash,
                    position=max_position + 1 if max_position else 1,
                    content=content,
                    word_count=len(content),
                    tokens=tokens,
                    status="completed",
                    indexing_at=datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None),
                    completed_at=datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None),
                    created_by=current_user.id,
                )
                if document.doc_form == "qa_model":
                    segment_document.answer = segment_item["answer"]
                db.session.add(segment_document)
                segment_data_list.append(segment_document)

                pre_segment_data_list.append(segment_document)
                if "keywords" in segment_item:
                    keywords_list.append(segment_item["keywords"])
                else:
                    keywords_list.append(None)

            try:
                # save vector index
                VectorService.create_segments_vector(keywords_list, pre_segment_data_list, dataset)
            except Exception as e:
                logging.exception("create segment index failed")
                for segment_document in segment_data_list:
                    segment_document.enabled = False
                    segment_document.disabled_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
                    segment_document.status = "error"
                    segment_document.error = str(e)
            db.session.commit()
            return segment_data_list

    @classmethod
    def update_segment(cls, args: dict, segment: DocumentSegment, document: Document, dataset: Dataset):
        indexing_cache_key = "segment_{}_indexing".format(segment.id)
        cache_result = redis_client.get(indexing_cache_key)
        if cache_result is not None:
            raise ValueError("Segment is indexing, please try again later")
        if "enabled" in args and args["enabled"] is not None:
            action = args["enabled"]
            if segment.enabled != action:
                if not action:
                    segment.enabled = action
                    segment.disabled_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
                    segment.disabled_by = current_user.id
                    db.session.add(segment)
                    db.session.commit()
                    # Set cache to prevent indexing the same segment multiple times
                    redis_client.setex(indexing_cache_key, 600, 1)
                    disable_segment_from_index_task.delay(segment.id)
                    return segment
        if not segment.enabled:
            if "enabled" in args and args["enabled"] is not None:
                if not args["enabled"]:
                    raise ValueError("Can't update disabled segment")
            else:
                raise ValueError("Can't update disabled segment")
        try:
            content = args["content"]
            if segment.content == content:
                if document.doc_form == "qa_model":
                    segment.answer = args["answer"]
                if args.get("keywords"):
                    segment.keywords = args["keywords"]
                segment.enabled = True
                segment.disabled_at = None
                segment.disabled_by = None
                db.session.add(segment)
                db.session.commit()
                # update segment index task
                if "keywords" in args:
                    keyword = Keyword(dataset)
                    keyword.delete_by_ids([segment.index_node_id])
                    document = RAGDocument(
                        page_content=segment.content,
                        metadata={
                            "doc_id": segment.index_node_id,
                            "doc_hash": segment.index_node_hash,
                            "document_id": segment.document_id,
                            "dataset_id": segment.dataset_id,
                        },
                    )
                    keyword.add_texts([document], keywords_list=[args["keywords"]])
            else:
                segment_hash = helper.generate_text_hash(content)
                tokens = 0
                if dataset.indexing_technique == "high_quality":
                    model_manager = ModelManager()
                    embedding_model = model_manager.get_model_instance(
                        tenant_id=current_user.current_tenant_id,
                        provider=dataset.embedding_model_provider,
                        model_type=ModelType.TEXT_EMBEDDING,
                        model=dataset.embedding_model,
                    )

                    # calc embedding use tokens
                    tokens = embedding_model.get_text_embedding_num_tokens(texts=[content])
                segment.content = content
                segment.index_node_hash = segment_hash
                segment.word_count = len(content)
                segment.tokens = tokens
                segment.status = "completed"
                segment.indexing_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
                segment.completed_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
                segment.updated_by = current_user.id
                segment.updated_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
                segment.enabled = True
                segment.disabled_at = None
                segment.disabled_by = None
                if document.doc_form == "qa_model":
                    segment.answer = args["answer"]
                db.session.add(segment)
                db.session.commit()
                # update segment vector index
                VectorService.update_segment_vector(args["keywords"], segment, dataset)

        except Exception as e:
            logging.exception("update segment index failed")
            segment.enabled = False
            segment.disabled_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
            segment.status = "error"
            segment.error = str(e)
            db.session.commit()
        segment = db.session.query(DocumentSegment).filter(DocumentSegment.id == segment.id).first()
        return segment

    @classmethod
    def delete_segment(cls, segment: DocumentSegment, document: Document, dataset: Dataset):
        indexing_cache_key = "segment_{}_delete_indexing".format(segment.id)
        cache_result = redis_client.get(indexing_cache_key)
        if cache_result is not None:
            raise ValueError("Segment is deleting.")

        # enabled segment need to delete index
        if segment.enabled:
            # send delete segment index task
            redis_client.setex(indexing_cache_key, 600, 1)
            delete_segment_from_index_task.delay(segment.id, segment.index_node_id, dataset.id, document.id)
        db.session.delete(segment)
        db.session.commit()


class DatasetCollectionBindingService:
    @classmethod
    def get_dataset_collection_binding(
        cls, provider_name: str, model_name: str, collection_type: str = "dataset"
    ) -> DatasetCollectionBinding:
        dataset_collection_binding = (
            db.session.query(DatasetCollectionBinding)
            .filter(
                DatasetCollectionBinding.provider_name == provider_name,
                DatasetCollectionBinding.model_name == model_name,
                DatasetCollectionBinding.type == collection_type,
            )
            .order_by(DatasetCollectionBinding.created_at)
            .first()
        )

        if not dataset_collection_binding:
            dataset_collection_binding = DatasetCollectionBinding(
                provider_name=provider_name,
                model_name=model_name,
                collection_name=Dataset.gen_collection_name_by_id(str(uuid.uuid4())),
                type=collection_type,
            )
            db.session.add(dataset_collection_binding)
            db.session.commit()
        return dataset_collection_binding

    @classmethod
    def get_dataset_collection_binding_by_id_and_type(
        cls, collection_binding_id: str, collection_type: str = "dataset"
    ) -> DatasetCollectionBinding:
        dataset_collection_binding = (
            db.session.query(DatasetCollectionBinding)
            .filter(
                DatasetCollectionBinding.id == collection_binding_id, DatasetCollectionBinding.type == collection_type
            )
            .order_by(DatasetCollectionBinding.created_at)
            .first()
        )

        return dataset_collection_binding


class DatasetPermissionService:
    @classmethod
    def get_dataset_partial_member_list(cls, dataset_id):
        user_list_query = (
            db.session.query(
                DatasetPermission.account_id,
            )
            .filter(DatasetPermission.dataset_id == dataset_id)
            .all()
        )

        user_list = []
        for user in user_list_query:
            user_list.append(user.account_id)

        return user_list

    @classmethod
    def update_partial_member_list(cls, tenant_id, dataset_id, user_list):
        try:
            db.session.query(DatasetPermission).filter(DatasetPermission.dataset_id == dataset_id).delete()
            permissions = []
            for user in user_list:
                permission = DatasetPermission(
                    tenant_id=tenant_id,
                    dataset_id=dataset_id,
                    account_id=user["user_id"],
                )
                permissions.append(permission)

            db.session.add_all(permissions)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise e

    @classmethod
    def check_permission(cls, user, dataset, requested_permission, requested_partial_member_list):
        if not user.is_dataset_editor:
            raise NoPermissionError("User does not have permission to edit this dataset.")

        if user.is_dataset_operator and dataset.permission != requested_permission:
            raise NoPermissionError("Dataset operators cannot change the dataset permissions.")

        if user.is_dataset_operator and requested_permission == "partial_members":
            if not requested_partial_member_list:
                raise ValueError("Partial member list is required when setting to partial members.")

            local_member_list = cls.get_dataset_partial_member_list(dataset.id)
            request_member_list = [user["user_id"] for user in requested_partial_member_list]
            if set(local_member_list) != set(request_member_list):
                raise ValueError("Dataset operators cannot change the dataset permissions.")

    @classmethod
    def clear_partial_member_list(cls, dataset_id):
        try:
            db.session.query(DatasetPermission).filter(DatasetPermission.dataset_id == dataset_id).delete()
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise e
