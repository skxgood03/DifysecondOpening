import base64
import logging
from typing import Optional, cast

import numpy as np
from sqlalchemy.exc import IntegrityError

from core.model_manager import ModelInstance
from core.model_runtime.entities.model_entities import ModelPropertyKey
from core.model_runtime.model_providers.__base.text_embedding_model import TextEmbeddingModel
from core.rag.datasource.entity.embedding import Embeddings
from extensions.ext_database import db
from extensions.ext_redis import redis_client
from libs import helper
from models.dataset import Embedding

logger = logging.getLogger(__name__)


class CacheEmbedding(Embeddings):
    def __init__(self, model_instance: ModelInstance, user: Optional[str] = None) -> None:
        self._model_instance = model_instance
        self._user = user

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入文本，每次处理10个文档。

            :param texts: 需要嵌入的文本列表
            :return: 文本对应的嵌入向量列表
        """
        # 初始化文本嵌入列表，长度与输入文本列表相同
        text_embeddings = [None for _ in range(len(texts))]
        # 存储需要嵌入的文本在原列表中的位置索引
        embedding_queue_indices = []
        # 遍历所有文本，检查是否已存在缓存的嵌入向量
        for i, text in enumerate(texts):
            # 为文本生成哈希值，用于数据库查询
            hash = helper.generate_text_hash(text)
            # 查询数据库中是否存在对应模型、提供商及哈希值的嵌入向量
            embedding = db.session.query(Embedding).filter_by(model_name=self._model_instance.model,
                                                              hash=hash,
                                                              provider_name=self._model_instance.provider).first()
            # 如果找到嵌入向量，则直接使用
            if embedding:

                text_embeddings[i] = embedding.get_embedding()
            else:
                # 否则，将文本索引添加到待处理队列
                embedding_queue_indices.append(i)
        # 如果存在未处理的文本，进行批量嵌入
        if embedding_queue_indices:
            # 提取待处理的文本
            embedding_queue_texts = [texts[i] for i in embedding_queue_indices]
            # 初始化嵌入结果队列
            embedding_queue_embeddings = []
            # 尝试获取模型实例的模型模式和最大块数属性
            try:
                model_type_instance = cast(TextEmbeddingModel, self._model_instance.model_type_instance)
                model_schema = model_type_instance.get_model_schema(self._model_instance.model,
                                                                    self._model_instance.credentials)
                # 获取模型的最大块数，用于控制批处理大小
                max_chunks = model_schema.model_properties[ModelPropertyKey.MAX_CHUNKS] \
                    if model_schema and ModelPropertyKey.MAX_CHUNKS in model_schema.model_properties else 1
                # 分批处理文本，每批不超过max_chunks个文本
                for i in range(0, len(embedding_queue_texts), max_chunks):
                    batch_texts = embedding_queue_texts[i:i + max_chunks]
                    # 调用模型进行文本嵌入
                    embedding_result = self._model_instance.invoke_text_embedding(
                        texts=batch_texts,
                        user=self._user
                    )
                    # 对每个嵌入向量进行归一化处理，并添加到结果队列
                    for vector in embedding_result.embeddings:
                        try:
                            normalized_embedding = (vector / np.linalg.norm(vector)).tolist()
                            embedding_queue_embeddings.append(normalized_embedding)
                        except IntegrityError:
                            db.session.rollback()
                        except Exception as e:
                            logging.exception('Failed transform embedding: ', e)
                cache_embeddings = []
                # 更新文本嵌入列表，并将新生成的嵌入向量缓存至数据库
                try:
                    for i, embedding in zip(embedding_queue_indices, embedding_queue_embeddings):
                        text_embeddings[i] = embedding
                        hash = helper.generate_text_hash(texts[i])
                        if hash not in cache_embeddings:
                            embedding_cache = Embedding(model_name=self._model_instance.model,
                                                        hash=hash,
                                                        provider_name=self._model_instance.provider)
                            embedding_cache.set_embedding(embedding)
                            db.session.add(embedding_cache)
                            cache_embeddings.append(hash)
                    db.session.commit()
                except IntegrityError:
                    db.session.rollback()
            except Exception as ex:
                db.session.rollback()
                logger.error('Failed to embed documents: ', ex)
                raise ex
        # 返回所有文本的嵌入向量列表
        return text_embeddings

    def embed_query(self, text: str) -> list[float]:
        """
        嵌入查询文本以生成向量表示。

        :param text: 要嵌入的文本字符串。
        :return: 文本的嵌入向量，表示为浮点数列表。
        """
        # 使用文档嵌入缓存或如果不存在则存储
        hash = helper.generate_text_hash(text) # 生成文本的哈希值
        embedding_cache_key = f'{self._model_instance.provider}_{self._model_instance.model}_{hash}'
        # 从Redis中获取嵌入向量
        embedding = redis_client.get(embedding_cache_key)
        if embedding: # 如果存在，则设置过期时间并返回解码后的嵌入向量
            redis_client.expire(embedding_cache_key, 600)
            return list(np.frombuffer(base64.b64decode(embedding), dtype="float"))
        try:
            # 调用模型实例生成文本嵌入
            embedding_result = self._model_instance.invoke_text_embedding(
                texts=[text],
                user=self._user
            )

            embedding_results = embedding_result.embeddings[0] # 获取第一个文本的嵌入结果
            embedding_results = (embedding_results / np.linalg.norm(embedding_results)).tolist() # 归一化并转换为列表
        except Exception as ex:
            raise ex

        try:
            # 将嵌入向量编码为Base64并存储到Redis中
            embedding_vector = np.array(embedding_results)
            vector_bytes = embedding_vector.tobytes()  # 转换为字节
            # Transform to Base64
            encoded_vector = base64.b64encode(vector_bytes)
            # Transform to string
            encoded_str = encoded_vector.decode("utf-8")
            redis_client.setex(embedding_cache_key, 600, encoded_str) # 存储到Redis并设置过期时间

        except IntegrityError:
            db.session.rollback()
        except:
            logging.exception('Failed to add embedding to redis')

        return embedding_results
