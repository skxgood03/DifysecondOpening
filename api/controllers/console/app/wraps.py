from collections.abc import Callable
from functools import wraps
from typing import Optional, Union

from controllers.console.app.error import AppNotFoundError
from extensions.ext_database import db
from libs.login import current_user
from models.model import App, AppMode


def get_app_model(view: Optional[Callable] = None, *, mode: Union[AppMode, list[AppMode]] = None):
    # 定义装饰器函数
    def decorator(view_func):
        # 定义装饰器内部函数
        @wraps(view_func)
        def decorated_view(*args, **kwargs):
            # 检查路径参数中是否包含 app_id
            if not kwargs.get("app_id"):
                raise ValueError("missing app_id in path parameters")
            # 从路径参数中获取 app_id
            app_id = kwargs.get("app_id")
            app_id = str(app_id)
            # 从路径参数中删除 app_id
            del kwargs["app_id"]
            # 从数据库中查询应用模型
            app_model = (
                db.session.query(App)
                .filter(App.id == app_id, App.tenant_id == current_user.current_tenant_id, App.status == "normal")
                .first()
            )
            # 如果未找到应用模型，则抛出异常
            if not app_model:
                raise AppNotFoundError()
            # 获取应用模式
            app_mode = AppMode.value_of(app_model.mode)
            # 如果应用模式为 CHANNEL，则抛出异常
            if app_mode == AppMode.CHANNEL:
                raise AppNotFoundError()
            # 如果提供了 mode 参数，则检查应用模式是否符合要求
            if mode is not None:
                if isinstance(mode, list):
                    modes = mode
                else:
                    modes = [mode]

                if app_mode not in modes:
                    mode_values = {m.value for m in modes}
                    raise AppNotFoundError(f"App mode is not in the supported list: {mode_values}")
            # 将应用模型添加到关键字参数中
            kwargs["app_model"] = app_model
            # 调用原始视图函数
            return view_func(*args, **kwargs)

        # 返回装饰器内部函数
        return decorated_view

    # 判断装饰器是否直接应用于视图函数
    if view is None:
        return decorator
    else: # 如果直接应用于视图函数，则立即应用装饰器并返回结果
        return decorator(view)
