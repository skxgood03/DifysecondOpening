import datetime
import logging
import time

import click
from celery import shared_task

from configs import dify_config
from core.indexing_runner import DocumentIsPausedException, IndexingRunner
from extensions.ext_database import db
from models.dataset import Dataset, Document
from services.feature_service import FeatureService


# 使用Celery装饰器定义一个异步任务，指定队列为'dataset'
@shared_task(queue='dataset')
def document_indexing_task(dataset_id: str, document_ids: list):
    """
    异步处理文档索引任务。
    :param dataset_id:数据集ID
    :param document_ids:需要处理的文档ID列表

    Usage: document_indexing_task.delay(dataset_id, document_id)
    主要功能：1.查询dataset的文章限制是否超出 入宫超出抛出异常，将所有document_ids状态改为 error 如果正常，则更新所有文章状态为 “解析中” parsing
    2.IndexingRunner.run()中包含了RAG索引的实现细节
    """


    # 初始化文档列表和开始时间
    documents = []
    start_at = time.perf_counter()
    # 从数据库中获取数据集信息
    dataset = db.session.query(Dataset).filter(Dataset.id == dataset_id).first()

    # 检查文档数量限制
    features = FeatureService.get_features(dataset.tenant_id)
    try:
        if features.billing.enabled:
            # 获取向量空间信息
            vector_space = features.vector_space
            count = len(document_ids)
            # 批量上传限制
            batch_upload_limit = int(dify_config.BATCH_UPLOAD_LIMIT)
            if count > batch_upload_limit:
                raise ValueError(f"You have reached the batch upload limit of {batch_upload_limit}.")

            # 检查是否超过订阅限制
            if 0 < vector_space.limit <= vector_space.size:
                raise ValueError("Your total number of documents plus the number of uploads have over the limit of "
                                 "your subscription.")
    except Exception as e:
        # 如果有异常，更新所有相关文档的状态为错误，并记录异常信息
        for document_id in document_ids:
            document = db.session.query(Document).filter(
                Document.id == document_id,
                Document.dataset_id == dataset_id
            ).first()
            if document:
                document.indexing_status = 'error'
                document.error = str(e)
                document.stopped_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
                db.session.add(document)
        db.session.commit()
        return
    # 更新文档状态为解析中，并添加到处理列表
    for document_id in document_ids:
        logging.info(click.style('Start process document: {}'.format(document_id), fg='green'))

        document = db.session.query(Document).filter(
            Document.id == document_id,
            Document.dataset_id == dataset_id
        ).first()

        if document:
            document.indexing_status = 'parsing'
            document.processing_started_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
            documents.append(document)
            db.session.add(document)
    db.session.commit()
    # 尝试运行索引处理
    try:
        # RAG索引的实现细节
        indexing_runner = IndexingRunner()
        indexing_runner.run(documents)
        end_at = time.perf_counter()
        logging.info(click.style('Processed dataset: {} latency: {}'.format(dataset_id, end_at - start_at), fg='green'))
    except DocumentIsPausedException as ex:
        # 如果文档被暂停，记录信息
        logging.info(click.style(str(ex), fg='yellow'))
    except Exception:
        pass
