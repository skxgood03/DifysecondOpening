import json

from flask import request
from flask_login import current_user
from flask_restful import Resource

from controllers.console import api
from controllers.console.app.wraps import get_app_model
from controllers.console.setup import setup_required
from controllers.console.wraps import account_initialization_required
from core.agent.entities import AgentToolEntity
from core.tools.tool_manager import ToolManager
from core.tools.utils.configuration import ToolParameterConfigurationManager
from events.app_event import app_model_config_was_updated
from extensions.ext_database import db
from libs.login import login_required
from models.model import AppMode, AppModelConfig
from services.app_model_config_service import AppModelConfigService


class ModelConfigResource(Resource):




    @setup_required
    @login_required
    @account_initialization_required
    @get_app_model(mode=[AppMode.AGENT_CHAT, AppMode.CHAT, AppMode.COMPLETION])
    def post(self, app_model):
        """Modify app model config"""
        # 验证配置
        model_configuration = AppModelConfigService.validate_configuration(
            tenant_id=current_user.current_tenant_id, config=request.json, app_mode=AppMode.value_of(app_model.mode)
        )
        # 创建一个新的应用程序模型配置对象
        new_app_model_config = AppModelConfig(
            app_id=app_model.id, # 设置应用程序ID
            created_by=current_user.id,
            updated_by=current_user.id,
        )
        new_app_model_config = new_app_model_config.from_model_config_dict(model_configuration) # 使用验证后的配置填充新对象
        # 如果是代理聊天或代理模式 如果是agent才会走下面
        if app_model.mode == AppMode.AGENT_CHAT.value or app_model.is_agent:
            # 获取原始的应用程序模型配置
            original_app_model_config: AppModelConfig = ( # 查询数据库获取原始配置
                db.session.query(AppModelConfig).filter(AppModelConfig.id == app_model.app_model_config_id).first()
            )
            agent_mode = original_app_model_config.agent_mode_dict
            # 解密代理工具参数（如果为秘密输入）
            parameter_map = {} # 存储解密后的参数
            masked_parameter_map = {} # 存储被遮盖的参数
            tool_map = {}  # 存储工具运行时实例
            for tool in agent_mode.get("tools") or []:  # 遍历代理模式下的所有工具
                if not isinstance(tool, dict) or len(tool.keys()) <= 3:  # 检查工具是否有效
                    continue
                # 创建代理工具实体
                agent_tool_entity = AgentToolEntity(**tool)
                # 获取工具
                try:
                    tool_runtime = ToolManager.get_agent_tool_runtime(
                        tenant_id=current_user.current_tenant_id,
                        app_id=app_model.id,
                        agent_tool=agent_tool_entity,
                    )
                    manager = ToolParameterConfigurationManager(
                        tenant_id=current_user.current_tenant_id,
                        tool_runtime=tool_runtime,
                        provider_name=agent_tool_entity.provider_id,
                        provider_type=agent_tool_entity.provider_type,
                        identity_id=f"AGENT.{app_model.id}",
                    )
                except Exception as e:
                    continue

                # 获取解密后的参数
                if agent_tool_entity.tool_parameters:
                    parameters = manager.decrypt_tool_parameters(agent_tool_entity.tool_parameters or {})
                    masked_parameter = manager.mask_tool_parameters(parameters or {})
                else:
                    parameters = {}
                    masked_parameter = {}

                key = f"{agent_tool_entity.provider_id}.{agent_tool_entity.provider_type}.{agent_tool_entity.tool_name}"
                masked_parameter_map[key] = masked_parameter # 存储遮盖参数
                parameter_map[key] = parameters # 存储解密参数
                tool_map[key] = tool_runtime # 存储工具运行时

            # 加密代理工具参数（如果为秘密输入）
            agent_mode = new_app_model_config.agent_mode_dict  # 获取新配置中的代理模式
            for tool in agent_mode.get("tools") or []:  # 遍历新配置中的工具
                agent_tool_entity = AgentToolEntity(**tool) # 创建代理工具实体

                # 获取工具
                key = f"{agent_tool_entity.provider_id}.{agent_tool_entity.provider_type}.{agent_tool_entity.tool_name}"
                if key in tool_map:
                    tool_runtime = tool_map[key]
                else:
                    try:
                        tool_runtime = ToolManager.get_agent_tool_runtime(
                            tenant_id=current_user.current_tenant_id,
                            app_id=app_model.id,
                            agent_tool=agent_tool_entity,
                        )
                    except Exception as e:
                        continue
                # 创建工具参数管理器
                manager = ToolParameterConfigurationManager(
                    tenant_id=current_user.current_tenant_id,
                    tool_runtime=tool_runtime,
                    provider_name=agent_tool_entity.provider_id,
                    provider_type=agent_tool_entity.provider_type,
                    identity_id=f"AGENT.{app_model.id}",
                )
                manager.delete_tool_parameters_cache()

                # 如果参数与遮盖参数相同则替换
                if agent_tool_entity.tool_parameters:
                    if key not in masked_parameter_map:
                        continue

                    for masked_key, masked_value in masked_parameter_map[key].items():
                        if (
                                masked_key in agent_tool_entity.tool_parameters
                                and agent_tool_entity.tool_parameters[masked_key] == masked_value
                        ):
                            agent_tool_entity.tool_parameters[masked_key] = parameter_map[key].get(masked_key)

                # 加密参数
                if agent_tool_entity.tool_parameters:
                    tool["tool_parameters"] = manager.encrypt_tool_parameters(agent_tool_entity.tool_parameters or {})

            # 更新应用程序模型配置
            new_app_model_config.agent_mode = json.dumps(agent_mode)  # 将代理模式序列化为JSON字符串
        # 添加新的应用程序模型配置到数据库会话
        db.session.add(new_app_model_config)
        db.session.flush() # 刷新会话以获取新记录的ID

        app_model.app_model_config_id = new_app_model_config.id # 更新应用程序模型的配置ID
        db.session.commit()  # 提交事务

        app_model_config_was_updated.send(app_model, app_model_config=new_app_model_config)  # 发送配置更新信号

        return {"result": "success"}


api.add_resource(ModelConfigResource, "/apps/<uuid:app_id>/model-config")
