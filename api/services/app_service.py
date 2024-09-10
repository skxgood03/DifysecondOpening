import json
import logging
from datetime import datetime, timezone
from typing import cast

from flask_login import current_user
from flask_sqlalchemy.pagination import Pagination

from configs import dify_config
from constants.model_template import default_app_templates
from core.agent.entities import AgentToolEntity
from core.app.features.rate_limiting import RateLimit
from core.errors.error import LLMBadRequestError, ProviderTokenNotInitError
from core.model_manager import ModelManager
from core.model_runtime.entities.model_entities import ModelPropertyKey, ModelType
from core.model_runtime.model_providers.__base.large_language_model import LargeLanguageModel
from core.tools.tool_manager import ToolManager
from core.tools.utils.configuration import ToolParameterConfigurationManager
from events.app_event import app_was_created
from extensions.ext_database import db
from models.account import Account
from models.model import App, AppMode, AppModelConfig
from models.tools import ApiToolProvider
from services.tag_service import TagService
from tasks.remove_app_and_related_data_task import remove_app_and_related_data_task


class AppService:
    def get_paginate_apps(self, tenant_id: str, args: dict) -> Pagination | None:
        """
        Get app list with pagination
        :param tenant_id: tenant id
        :param args: request args
        :return:
        """
        filters = [App.tenant_id == tenant_id, App.is_universal == False]

        if args["mode"] == "workflow":
            filters.append(App.mode.in_([AppMode.WORKFLOW.value, AppMode.COMPLETION.value]))
        elif args["mode"] == "chat":
            filters.append(App.mode.in_([AppMode.CHAT.value, AppMode.ADVANCED_CHAT.value]))
        elif args["mode"] == "agent-chat":
            filters.append(App.mode == AppMode.AGENT_CHAT.value)
        elif args["mode"] == "channel":
            filters.append(App.mode == AppMode.CHANNEL.value)

        if args.get("name"):
            name = args["name"][:30]
            filters.append(App.name.ilike(f"%{name}%"))
        if args.get("tag_ids"):
            target_ids = TagService.get_target_ids_by_tag_ids("app", tenant_id, args["tag_ids"])
            if target_ids:
                filters.append(App.id.in_(target_ids))
            else:
                return None

        app_models = db.paginate(
            db.select(App).where(*filters).order_by(App.created_at.desc()),
            page=args["page"],
            per_page=args["limit"],
            error_out=False,
        )

        return app_models

    def create_app(self, tenant_id: str, args: dict, account: Account) -> App:
        """
        Create app
        :param tenant_id: tenant id
        :param args: request args
        :param account: Account instance
        """
        # 获取应用模式
        app_mode = AppMode.value_of(args["mode"])
        # 根据应用模式获取默认的应用模板
        app_template = default_app_templates[app_mode]

        # 获取模型配置
        default_model_config = app_template.get("model_config")
        # 如果存在默认模型配置，则复制一份以避免修改原始数据
        default_model_config = default_model_config.copy() if default_model_config else None
        # 如果存在默认模型配置并且配置中包含 "model" 字段
        if default_model_config and "model" in default_model_config:
            # 初始化模型管理器
            model_manager = ModelManager()

            # 尝试获取默认的模型实例
            try:
                model_instance = model_manager.get_default_model_instance(
                    tenant_id=account.current_tenant_id, model_type=ModelType.LLM
                )
            except (ProviderTokenNotInitError, LLMBadRequestError):
                # 如果出现特定错误，则设置模型实例为 None
                model_instance = None
            except Exception as e:
                # 记录其他异常
                logging.exception(e)
                model_instance = None
            # 如果找到了模型实例
            if model_instance:
                # 检查模型实例是否与默认配置中的模型和提供商匹配
                if (
                        model_instance.model == default_model_config["model"]["name"]
                        and model_instance.provider == default_model_config["model"]["provider"]
                ):  # 如果匹配，则直接使用默认配置中的模型字典
                    default_model_dict = default_model_config["model"]
                else:
                    # 如果不匹配，根据模型实例创建新的模型字典
                    llm_model = cast(LargeLanguageModel, model_instance.model_type_instance)
                    model_schema = llm_model.get_model_schema(model_instance.model, model_instance.credentials)

                    default_model_dict = {
                        "provider": model_instance.provider,
                        "name": model_instance.model,
                        "mode": model_schema.model_properties.get(ModelPropertyKey.MODE),
                        "completion_params": {},
                    }
            else:
                # 如果没有找到模型实例，则尝试获取默认的提供商和模型名称
                provider, model = model_manager.get_default_provider_model_name(
                    tenant_id=account.current_tenant_id, model_type=ModelType.LLM
                )
                default_model_config["model"]["provider"] = provider
                default_model_config["model"]["name"] = model
                default_model_dict = default_model_config["model"]
            # 将模型字典转换为 JSON 格式并存储在默认模型配置中
            default_model_config["model"] = json.dumps(default_model_dict)
        # 创建应用实例
        app = App(**app_template["app"])
        app.name = args["name"]
        app.description = args.get("description", "")
        app.mode = args["mode"]
        app.icon_type = args.get("icon_type", "emoji")
        app.icon = args["icon"]
        app.icon_background = args["icon_background"]
        app.tenant_id = tenant_id
        app.api_rph = args.get("api_rph", 0)
        app.api_rpm = args.get("api_rpm", 0)
        app.created_by = account.id
        app.updated_by = account.id

        db.session.add(app)
        # 提交事务以便获取应用 ID
        db.session.flush()
        # 如果存在默认模型配置
        if default_model_config:
            # 创建应用模型配置实例
            app_model_config = AppModelConfig(**default_model_config)
            # 设置应用模型配置关联的应用 ID
            app_model_config.app_id = app.id
            app_model_config.created_by = account.id
            app_model_config.updated_by = account.id
            db.session.add(app_model_config)
            # 提交事务以便获取应用模型配置 ID
            db.session.flush()
            # 设置应用关联的应用模型配置 ID
            app.app_model_config_id = app_model_config.id
        # 提交所有更改到数据库
        db.session.commit()
        # 发送应用创建信号
        app_was_created.send(app, account=account)

        return app

    def get_app(self, app: App) -> App:
        """
        Get App
        """
        # 获取原始的应用模型配置
        if app.mode == AppMode.AGENT_CHAT.value or app.is_agent:
            model_config: AppModelConfig = app.app_model_config
            agent_mode = model_config.agent_mode_dict
            # decrypt agent tool parameters if it's secret-input
            # 遍历代理模式下的工具列表
            for tool in agent_mode.get("tools") or []:
                # 如果工具不是字典类型或键的数量小于等于3，则跳过
                if not isinstance(tool, dict) or len(tool.keys()) <= 3:
                    continue
                # 将工具信息转换为 AgentToolEntity 对象
                agent_tool_entity = AgentToolEntity(**tool)
                # 尝试获取工具运行时实例
                try:
                    tool_runtime = ToolManager.get_agent_tool_runtime(
                        tenant_id=current_user.current_tenant_id,
                        app_id=app.id,
                        agent_tool=agent_tool_entity,
                    )
                    # 创建工具参数配置管理器实例
                    manager = ToolParameterConfigurationManager(
                        tenant_id=current_user.current_tenant_id,
                        tool_runtime=tool_runtime,
                        provider_name=agent_tool_entity.provider_id,
                        provider_type=agent_tool_entity.provider_type,
                        identity_id=f"AGENT.{app.id}",
                    )

                    # 获取解密后的工具参数
                    if agent_tool_entity.tool_parameters:
                        parameters = manager.decrypt_tool_parameters(agent_tool_entity.tool_parameters or {})
                        masked_parameter = manager.mask_tool_parameters(parameters or {})
                    else:
                        masked_parameter = {}

                    # 替换工具参数
                    tool["tool_parameters"] = masked_parameter
                except Exception as e:  # 如果出现异常，则忽略该工具
                    pass

            # 将解密后的代理模式转换为 JSON 格式并覆盖原有值
            model_config.agent_mode = json.dumps(agent_mode)

            # 定义一个继承自 App 的新类，用于覆盖模型配置
            class ModifiedApp(App):
                """
                Modified App class
                """

                def __init__(self, app):
                    # 更新新类的属性为原始 App 的属性
                    self.__dict__.update(app.__dict__)

                @property
                def app_model_config(self):
                    # 返回修改后的模型配置
                    return model_config

            # 创建 ModifiedApp 类的实例并返回
            app = ModifiedApp(app)
        # 返回最终的应用实例
        return app

    def update_app(self, app: App, args: dict) -> App:
        """
        Update app
        :param app: App instance
        :param args: request args
        :return: App instance
        """
        app.name = args.get("name")
        app.description = args.get("description", "")
        app.max_active_requests = args.get("max_active_requests")
        app.icon_type = args.get("icon_type", "emoji")
        app.icon = args.get("icon")
        app.icon_background = args.get("icon_background")
        app.use_icon_as_answer_icon = args.get("use_icon_as_answer_icon", False)
        app.updated_by = current_user.id
        app.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        db.session.commit()

        if app.max_active_requests is not None:
            rate_limit = RateLimit(app.id, app.max_active_requests)
            rate_limit.flush_cache(use_local_value=True)
        return app

    def update_app_name(self, app: App, name: str) -> App:
        """
        Update app name
        :param app: App instance
        :param name: new name
        :return: App instance
        """
        app.name = name
        app.updated_by = current_user.id
        app.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        db.session.commit()

        return app

    def update_app_icon(self, app: App, icon: str, icon_background: str) -> App:
        """
        Update app icon
        :param app: App instance
        :param icon: new icon
        :param icon_background: new icon_background
        :return: App instance
        """
        app.icon = icon
        app.icon_background = icon_background
        app.updated_by = current_user.id
        app.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        db.session.commit()

        return app

    def update_app_site_status(self, app: App, enable_site: bool) -> App:
        """
        Update app site status
        :param app: App instance
        :param enable_site: enable site status
        :return: App instance
        """
        if enable_site == app.enable_site:
            return app

        app.enable_site = enable_site
        app.updated_by = current_user.id
        app.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        db.session.commit()

        return app

    def update_app_api_status(self, app: App, enable_api: bool) -> App:
        """
        Update app api status
        :param app: App instance
        :param enable_api: enable api status
        :return: App instance
        """
        if enable_api == app.enable_api:
            return app

        app.enable_api = enable_api
        app.updated_by = current_user.id
        app.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        db.session.commit()

        return app

    def delete_app(self, app: App) -> None:
        """
        Delete app
        :param app: App instance
        """
        db.session.delete(app)
        db.session.commit()

        # Trigger asynchronous deletion of app and related data
        remove_app_and_related_data_task.delay(tenant_id=app.tenant_id, app_id=app.id)

    def get_app_meta(self, app_model: App) -> dict:
        """
        Get app meta info
        :param app_model: app model
        :return:
        """
        app_mode = AppMode.value_of(app_model.mode)

        meta = {"tool_icons": {}}

        if app_mode in [AppMode.ADVANCED_CHAT, AppMode.WORKFLOW]:
            workflow = app_model.workflow
            if workflow is None:
                return meta

            graph = workflow.graph_dict
            nodes = graph.get("nodes", [])
            tools = []
            for node in nodes:
                if node.get("data", {}).get("type") == "tool":
                    node_data = node.get("data", {})
                    tools.append(
                        {
                            "provider_type": node_data.get("provider_type"),
                            "provider_id": node_data.get("provider_id"),
                            "tool_name": node_data.get("tool_name"),
                            "tool_parameters": {},
                        }
                    )
        else:
            app_model_config: AppModelConfig = app_model.app_model_config

            if not app_model_config:
                return meta

            agent_config = app_model_config.agent_mode_dict or {}

            # get all tools
            tools = agent_config.get("tools", [])

        url_prefix = dify_config.CONSOLE_API_URL + "/console/api/workspaces/current/tool-provider/builtin/"

        for tool in tools:
            keys = list(tool.keys())
            if len(keys) >= 4:
                # current tool standard
                provider_type = tool.get("provider_type")
                provider_id = tool.get("provider_id")
                tool_name = tool.get("tool_name")
                if provider_type == "builtin":
                    meta["tool_icons"][tool_name] = url_prefix + provider_id + "/icon"
                elif provider_type == "api":
                    try:
                        provider: ApiToolProvider = (
                            db.session.query(ApiToolProvider).filter(ApiToolProvider.id == provider_id).first()
                        )
                        meta["tool_icons"][tool_name] = json.loads(provider.icon)
                    except:
                        meta["tool_icons"][tool_name] = {"background": "#252525", "content": "\ud83d\ude01"}

        return meta
