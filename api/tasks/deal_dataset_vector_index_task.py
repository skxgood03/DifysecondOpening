import logging
import time

import click
from celery import shared_task

from core.rag.index_processor.index_processor_factory import IndexProcessorFactory
from core.rag.models.document import Document
from extensions.ext_database import db
from models.dataset import Dataset, DocumentSegment
from models.dataset import Document as DatasetDocument


@shared_task(queue='dataset')
def deal_dataset_vector_index_task(dataset_id: str, action: str):


    """
    异步处理数据集向量索引任务。

    :param dataset_id: 数据集ID
    :param action: 操作类型，可以是'remove'、'add'或'update'
    使用示例: deal_dataset_vector_index_task.delay(dataset_id, action)
    """
    logging.info(click.style('Start deal dataset vector index: {}'.format(dataset_id), fg='green'))
    start_at = time.perf_counter()

    try:
        # 从数据库查询指定ID的数据集
        dataset = Dataset.query.filter_by(
            id=dataset_id
        ).first()

        if not dataset:
            raise Exception('Dataset not found')
        index_type = dataset.doc_form
        index_processor = IndexProcessorFactory(index_type).init_index_processor()
        # 根据action执行不同操作
        if action == "remove":
            # 清除数据集的向量索引，不处理关键词
            index_processor.clean(dataset, None, with_keywords=False)
        elif action == "add":
            # 查询符合条件的文档列表
            dataset_documents = db.session.query(DatasetDocument).filter(
                DatasetDocument.dataset_id == dataset_id,
                DatasetDocument.indexing_status == 'completed',
                DatasetDocument.enabled == True,
                DatasetDocument.archived == False,
            ).all()
            # 如果有符合条件的文档，处理它们
            if dataset_documents:
                documents = []
                for dataset_document in dataset_documents:
                    #   # 查询文档的段落
                    segments = db.session.query(DocumentSegment).filter(
                        DocumentSegment.document_id == dataset_document.id,
                        DocumentSegment.enabled == True
                    ) .order_by(DocumentSegment.position.asc()).all()
                    # 构建Document对象并添加到列表
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

                # 加载数据到向量索引，不处理关键词
                index_processor.load(dataset, documents, with_keywords=False)
        elif action == 'update':
            # 清除数据集的向量索引，不处理关键词
            index_processor.clean(dataset, None, with_keywords=False)
            # 查询符合条件的文档列表
            dataset_documents = db.session.query(DatasetDocument).filter(
                DatasetDocument.dataset_id == dataset_id,
                DatasetDocument.indexing_status == 'completed',
                DatasetDocument.enabled == True,
                DatasetDocument.archived == False,
            ).all()
            # 如果有符合条件的文档，处理它们
            if dataset_documents:
                documents = []
                for dataset_document in dataset_documents:
                    # 查询文档的段落
                    segments = db.session.query(DocumentSegment).filter(
                        DocumentSegment.document_id == dataset_document.id,
                        DocumentSegment.enabled == True
                    ).order_by(DocumentSegment.position.asc()).all()
                    # 构建Document对象并添加到列表
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

                # 加载数据到向量索引，不处理关键词
                index_processor.load(dataset, documents, with_keywords=False)
        # 结束时间，计算并记录处理耗时
        end_at = time.perf_counter()
        logging.info(
            click.style('Deal dataset vector index: {} latency: {}'.format(dataset_id, end_at - start_at), fg='green'))
    except Exception:
        logging.exception("Deal dataset vector index failed")
