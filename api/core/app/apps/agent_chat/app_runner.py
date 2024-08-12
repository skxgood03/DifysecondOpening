import logging
from typing import cast

from core.agent.cot_chat_agent_runner import CotChatAgentRunner
from core.agent.cot_completion_agent_runner import CotCompletionAgentRunner
from core.agent.entities import AgentEntity
from core.agent.fc_agent_runner import FunctionCallAgentRunner
from core.app.apps.agent_chat.app_config_manager import AgentChatAppConfig
from core.app.apps.base_app_queue_manager import AppQueueManager, PublishFrom
from core.app.apps.base_app_runner import AppRunner
from core.app.entities.app_invoke_entities import AgentChatAppGenerateEntity, ModelConfigWithCredentialsEntity
from core.app.entities.queue_entities import QueueAnnotationReplyEvent
from core.memory.token_buffer_memory import TokenBufferMemory
from core.model_manager import ModelInstance
from core.model_runtime.entities.llm_entities import LLMMode, LLMUsage
from core.model_runtime.entities.model_entities import ModelFeature, ModelPropertyKey
from core.model_runtime.model_providers.__base.large_language_model import LargeLanguageModel
from core.moderation.base import ModerationException
from core.tools.entities.tool_entities import ToolRuntimeVariablePool
from extensions.ext_database import db
from models.model import App, Conversation, Message, MessageAgentThought
from models.tools import ToolConversationVariables

logger = logging.getLogger(__name__)


class AgentChatAppRunner(AppRunner):
    """
    代理应用程序运行器
    """

    def run(
        self, application_generate_entity: AgentChatAppGenerateEntity,
        queue_manager: AppQueueManager,
        conversation: Conversation,
        message: Message,
    ) -> None:
        """
        运行助手应用程序

        :param application_generate_entity: 应用生成实体
        :param queue_manager: 应用队列管理器
        :param conversation: 会话
        :param message: 消息
        """

        # 获取应用配置
        app_config = application_generate_entity.app_config
        app_config = cast(AgentChatAppConfig, app_config)
        # 查询数据库获取应用记录
        app_record = db.session.query(App).filter(App.id == app_config.app_id).first()
        if not app_record:
            raise ValueError("App not found")
        # 获取输入、查询和文件
        inputs = application_generate_entity.inputs
        query = application_generate_entity.query
        files = application_generate_entity.files

        # Pre-calculate the number of tokens of the prompt messages,
        # and return the rest number of tokens by model context token size limit and max token size limit.
        # If the rest number of tokens is not enough, raise exception.
        # Include: prompt template, inputs, query(optional), files(optional)
        # Not Include: memory, external data, dataset context
        # 预计算提示消息的令牌数量，并确保剩余令牌数量足够
        self.get_pre_calculate_rest_tokens(
            app_record=app_record,
            model_config=application_generate_entity.model_conf,
            prompt_template_entity=app_config.prompt_template,
            inputs=inputs,
            files=files,
            query=query
        )
        # 初始化记忆对象
        memory = None
        if application_generate_entity.conversation_id:
            # get memory of conversation (read-only)
            model_instance = ModelInstance(
                provider_model_bundle=application_generate_entity.model_conf.provider_model_bundle,
                model=application_generate_entity.model_conf.model
            )

            memory = TokenBufferMemory(
                conversation=conversation,
                model_instance=model_instance
            )

        # organize all inputs and template to prompt messages
        # Include: prompt template, inputs, query(optional), files(optional)
        #          memory(optional)
        # 组织所有输入和模板到提示消息
        prompt_messages, _ = self.organize_prompt_messages(
            app_record=app_record,
            model_config=application_generate_entity.model_conf,
            prompt_template_entity=app_config.prompt_template,
            inputs=inputs,
            files=files,
            query=query,
            memory=memory
        )

        # 执行敏感词过滤
        try:
            # process sensitive_word_avoidance
            _, inputs, query = self.moderation_for_inputs(
                app_id=app_record.id,
                tenant_id=app_config.tenant_id,
                app_generate_entity=application_generate_entity,
                inputs=inputs,
                query=query,
                message_id=message.id
            )
        except ModerationException as e:
            # 直接输出异常信息
            self.direct_output(
                queue_manager=queue_manager,
                app_generate_entity=application_generate_entity,
                prompt_messages=prompt_messages,
                text=str(e),
                stream=application_generate_entity.stream
            )
            return
        # 检查是否有注解回复
        if query:
            # annotation reply
            annotation_reply = self.query_app_annotations_to_reply(
                app_record=app_record,
                message=message,
                query=query,
                user_id=application_generate_entity.user_id,
                invoke_from=application_generate_entity.invoke_from
            )
            # 发布注解回复事件
            if annotation_reply:
                queue_manager.publish(
                    QueueAnnotationReplyEvent(message_annotation_id=annotation_reply.id),
                    PublishFrom.APPLICATION_MANAGER
                )
                # 直接输出注解回复内容
                self.direct_output(
                    queue_manager=queue_manager,
                    app_generate_entity=application_generate_entity,
                    prompt_messages=prompt_messages,
                    text=annotation_reply.content,
                    stream=application_generate_entity.stream
                )
                return

        # 从外部数据工具填充输入变量
        external_data_tools = app_config.external_data_variables
        if external_data_tools:
            inputs = self.fill_in_inputs_from_external_data_tools(
                tenant_id=app_record.tenant_id,
                app_id=app_record.id,
                external_data_tools=external_data_tools,
                inputs=inputs,
                query=query
            )

        # reorganize all inputs and template to prompt messages
        # Include: prompt template, inputs, query(optional), files(optional)
        #          memory(optional), external data, dataset context(optional)
        # 重新组织所有输入和模板到提示消息
        prompt_messages, _ = self.organize_prompt_messages(
            app_record=app_record,
            model_config=application_generate_entity.model_conf,
            prompt_template_entity=app_config.prompt_template,
            inputs=inputs,
            files=files,
            query=query,
            memory=memory
        )

        # 检查托管审核
        hosting_moderation_result = self.check_hosting_moderation(
            application_generate_entity=application_generate_entity,
            queue_manager=queue_manager,
            prompt_messages=prompt_messages
        )

        if hosting_moderation_result:
            return

        agent_entity = app_config.agent

        # # 加载工具变量
        tool_conversation_variables = self._load_tool_variables(conversation_id=conversation.id,
                                                   user_id=application_generate_entity.user_id,
                                                   tenant_id=app_config.tenant_id)

        # 将数据库变量转换为工具变量
        tool_variables = self._convert_db_variables_to_tool_variables(tool_conversation_variables)

        # 初始化模型实例
        model_instance = ModelInstance(
            provider_model_bundle=application_generate_entity.model_conf.provider_model_bundle,
            model=application_generate_entity.model_conf.model
        )
        prompt_message, _ = self.organize_prompt_messages(
            app_record=app_record,
            model_config=application_generate_entity.model_conf,
            prompt_template_entity=app_config.prompt_template,
            inputs=inputs,
            files=files,
            query=query,
            memory=memory,
        )

        # 确定函数调用策略
        llm_model = cast(LargeLanguageModel, model_instance.model_type_instance)
        model_schema = llm_model.get_model_schema(model_instance.model, model_instance.credentials)

        if {ModelFeature.MULTI_TOOL_CALL, ModelFeature.TOOL_CALL}.intersection(model_schema.features or []):
            agent_entity.strategy = AgentEntity.Strategy.FUNCTION_CALLING
        # 重新加载会话和消息
        conversation = db.session.query(Conversation).filter(Conversation.id == conversation.id).first()
        message = db.session.query(Message).filter(Message.id == message.id).first()
        db.session.close()

        # 根据策略选择运行器
        if agent_entity.strategy == AgentEntity.Strategy.CHAIN_OF_THOUGHT:
            # check LLM mode
            if model_schema.model_properties.get(ModelPropertyKey.MODE) == LLMMode.CHAT.value:
                runner_cls = CotChatAgentRunner
            elif model_schema.model_properties.get(ModelPropertyKey.MODE) == LLMMode.COMPLETION.value:
                runner_cls = CotCompletionAgentRunner
            else:
                raise ValueError(f"Invalid LLM mode: {model_schema.model_properties.get(ModelPropertyKey.MODE)}")
        elif agent_entity.strategy == AgentEntity.Strategy.FUNCTION_CALLING:
            runner_cls = FunctionCallAgentRunner
        else:
            raise ValueError(f"Invalid agent strategy: {agent_entity.strategy}")
        # 初始化运行器
        runner = runner_cls(
            tenant_id=app_config.tenant_id,
            application_generate_entity=application_generate_entity,
            conversation=conversation,
            app_config=app_config,
            model_config=application_generate_entity.model_conf,
            config=agent_entity,
            queue_manager=queue_manager,
            message=message,
            user_id=application_generate_entity.user_id,
            memory=memory,
            prompt_messages=prompt_message,
            variables_pool=tool_variables,
            db_variables=tool_conversation_variables,
            model_instance=model_instance
        )
        # 运行代理
        invoke_result = runner.run(
            message=message,
            query=query,
            inputs=inputs,
        )

        # 处理调用结果
        self._handle_invoke_result(
            invoke_result=invoke_result,
            queue_manager=queue_manager,
            stream=application_generate_entity.stream,
            agent=True
        )

    def _load_tool_variables(self, conversation_id: str, user_id: str, tenant_id: str) -> ToolConversationVariables:
        """
        load tool variables from database
        """
        tool_variables: ToolConversationVariables = db.session.query(ToolConversationVariables).filter(
            ToolConversationVariables.conversation_id == conversation_id,
            ToolConversationVariables.tenant_id == tenant_id
        ).first()

        if tool_variables:
            # save tool variables to session, so that we can update it later
            db.session.add(tool_variables)
        else:
            # create new tool variables
            tool_variables = ToolConversationVariables(
                conversation_id=conversation_id,
                user_id=user_id,
                tenant_id=tenant_id,
                variables_str='[]',
            )
            db.session.add(tool_variables)
            db.session.commit()

        return tool_variables
    
    def _convert_db_variables_to_tool_variables(self, db_variables: ToolConversationVariables) -> ToolRuntimeVariablePool:
        """
        convert db variables to tool variables
        """
        return ToolRuntimeVariablePool(**{
            'conversation_id': db_variables.conversation_id,
            'user_id': db_variables.user_id,
            'tenant_id': db_variables.tenant_id,
            'pool': db_variables.variables
        })

    def _get_usage_of_all_agent_thoughts(self, model_config: ModelConfigWithCredentialsEntity,
                                         message: Message) -> LLMUsage:
        """
        Get usage of all agent thoughts
        :param model_config: model config
        :param message: message
        :return:
        """
        agent_thoughts = (db.session.query(MessageAgentThought)
                          .filter(MessageAgentThought.message_id == message.id).all())

        all_message_tokens = 0
        all_answer_tokens = 0
        for agent_thought in agent_thoughts:
            all_message_tokens += agent_thought.message_tokens
            all_answer_tokens += agent_thought.answer_tokens

        model_type_instance = model_config.provider_model_bundle.model_type_instance
        model_type_instance = cast(LargeLanguageModel, model_type_instance)

        return model_type_instance._calc_response_usage(
            model_config.model,
            model_config.credentials,
            all_message_tokens,
            all_answer_tokens
        )
