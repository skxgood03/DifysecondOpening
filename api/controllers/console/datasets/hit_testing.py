import logging

from flask_login import current_user
from flask_restful import Resource, marshal, reqparse
from werkzeug.exceptions import Forbidden, InternalServerError, NotFound

import services
from controllers.console import api
from controllers.console.app.error import (
    CompletionRequestError,
    ProviderModelCurrentlyNotSupportError,
    ProviderNotInitializeError,
    ProviderQuotaExceededError,
)
from controllers.console.datasets.error import DatasetNotInitializedError
from controllers.console.setup import setup_required
from controllers.console.wraps import account_initialization_required
from core.errors.error import (
    LLMBadRequestError,
    ModelCurrentlyNotSupportError,
    ProviderTokenNotInitError,
    QuotaExceededError,
)
from core.model_runtime.errors.invoke import InvokeError
from fields.hit_testing_fields import hit_testing_record_fields
from libs.login import login_required
from services.dataset_service import DatasetService
from services.hit_testing_service import HitTestingService


class HitTestingApi(Resource):

    @setup_required
    @login_required
    @account_initialization_required
    def post(self, dataset_id):
        dataset_id_str = str(dataset_id)
        # 从数据集中获取指定ID的数据集
        dataset = DatasetService.get_dataset(dataset_id_str)
        if dataset is None:
            raise NotFound("Dataset not found.")
        # 检查当前用户是否有访问该数据集的权限
        try:
            DatasetService.check_dataset_permission(dataset, current_user)
        except services.errors.account.NoPermissionError as e:
            raise Forbidden(str(e))
        # 解析请求参数
        parser = reqparse.RequestParser()
        parser.add_argument('query', type=str, location='json') #问题
        parser.add_argument('retrieval_model', type=dict, required=False, location='json') #模型
        args = parser.parse_args()
        # 验证命中测试所需的参数
        HitTestingService.hit_testing_args_check(args)

        try:# 执行命中测试
            response = HitTestingService.retrieve(
                dataset=dataset,
                query=args['query'],
                account=current_user,
                retrieval_model=args['retrieval_model'],
                limit=10
            )
            #构建响应数据
            return {"query": response['query'], 'records': marshal(response['records'], hit_testing_record_fields)}
        #异常处理
        except services.errors.index.IndexNotInitializedError:
            raise DatasetNotInitializedError()
        except ProviderTokenNotInitError as ex:
            raise ProviderNotInitializeError(ex.description)
        except QuotaExceededError:
            raise ProviderQuotaExceededError()
        except ModelCurrentlyNotSupportError:
            raise ProviderModelCurrentlyNotSupportError()
        except LLMBadRequestError:
            raise ProviderNotInitializeError(
                "No Embedding Model or Reranking Model available. Please configure a valid provider "
                "in the Settings -> Model Provider.")
        except InvokeError as e:
            raise CompletionRequestError(e.description)
        except ValueError as e:
            raise ValueError(str(e))
        except Exception as e:
            logging.exception("Hit testing failed.")
            raise InternalServerError(str(e))


api.add_resource(HitTestingApi, '/datasets/<uuid:dataset_id>/hit-testing')
