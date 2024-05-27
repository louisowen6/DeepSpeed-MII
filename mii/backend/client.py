# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import asyncio
import grpc
import requests
from typing import Dict, Any, Callable, List, Union

from mii.batching.data_classes import Response
from mii.config import MIIConfig
from mii.constants import GRPC_MAX_MSG_SIZE
from mii.grpc_related.proto import modelresponse_pb2, modelresponse_pb2_grpc
from mii.grpc_related.task_methods import TASK_METHODS_DICT
from transformers import AutoTokenizer


def create_channel(host, port):
    return grpc.aio.insecure_channel(
        f"{host}:{port}",
        options=[
            ("grpc.max_send_message_length",
             GRPC_MAX_MSG_SIZE),
            ("grpc.max_receive_message_length",
             GRPC_MAX_MSG_SIZE),
        ],
    )


class MIIClient:
    """
    Client for sending generation requests to a persistent deployment created
    with :func:`mii.serve`. Use :func:`mii.client` to create an instance of this
    class.

    :param mii_config: MII config for the persistent deployment to connect with.
    :param host: hostname where the persistent deployment is running.
    """
    def __init__(self, mii_config: MIIConfig, host: str = "localhost") -> None:
        self.mii_config = mii_config
        self.task = mii_config.model_config.task
        self.port = mii_config.port_number
        self.asyncio_loop = asyncio.get_event_loop()
        channel = create_channel(host, self.port)
        self.stub = modelresponse_pb2_grpc.ModelResponseStub(channel)
        self.tokenizer = AutoTokenizer.from_pretrained(mii_config.model_config.model_name_or_path)
        self.max_input_token_length = mii_config.model_config.max_input_length

    def __call__(self, *args, **kwargs) -> List[Response]:
        """
        All args and kwargs get passed directly to
        :meth:`~mii.backend.client.MIIClient.generate`.

        :return: A list of :class:`Response` objects containing the generated
            text for all prompts.
        """
        return self.generate(*args, **kwargs)

    async def _request_async_response(self, prompts, **query_kwargs):
        task_methods = TASK_METHODS_DICT[self.task]
        proto_request = task_methods.pack_request_to_proto(prompts, **query_kwargs)
        proto_response = await getattr(self.stub, task_methods.method)(proto_request)
        return task_methods.unpack_response_from_proto(proto_response)

    async def _request_async_response_stream(self, prompts, **query_kwargs):
        task_methods = TASK_METHODS_DICT[self.task]
        proto_request = task_methods.pack_request_to_proto(prompts, **query_kwargs)
        assert hasattr(task_methods, "method_stream_out"), f"{self.task} does not support streaming response"
        async for response in getattr(self.stub,
                                      task_methods.method_stream_out)(proto_request):
            yield task_methods.unpack_response_from_proto(response)

    def generate(self,
                 prompts: dict,
                 streaming_fn: Callable = None,
                 **generate_kwargs: Dict) -> List[Response]:
        """
        Generates text for the given prompts.

        :param prompts: The string or list of strings used as prompts for generation.
        :param streaming_fn: Streaming support is currently a WIP.
        :param \\*\\*generate_kwargs: Generation keywords. A full list can be found here.

        :return: A list of :class:`Response` objects containing the generated
            text for all prompts.
        """ # noqa: W605
        task = prompts["task"]
        inp = prompts["input"]
        use_empathy,use_short,use_instruction = prompts.get("use_empathy",False), prompts.get("use_short",False), prompts.get("use_instruction",False)
        if task == "doc_qna":
            knowledge_list = inp['knowledge_list']
            knowledge = get_knowledge(knowledge_list)
            while get_token_length(self.tokenizer,knowledge) > self.max_input_token_length:
                knowledge_list = knowledge_list[:-1]
                knowledge = get_knowledge(knowledge_list)
    
            question = inp['query']
            formatted_input = f"Knowledge: {knowledge}\n\nQuestion: {question}"
        elif task in ['contextual_user_query_rephrasing', 'contextual_user_query_rephrasing_negative']:
            formatted_input = inp['query']
        else: #conversation_summarization
            formatted_input = inp

        prompts = [f"<sys>{get_prompt(task,use_empathy,use_short,use_instruction)}{formatted_input}\n<bot>"]

        return self.asyncio_loop.run_until_complete(
            self._request_async_response(prompts,
                                         **generate_kwargs))

    def _generate_stream(self,
                         callback,
                         prompts: List[str],
                         **query_kwargs: Dict[str,
                                              Any]) -> None:
        async def put_result():
            response_stream = self._request_async_response_stream(
                prompts,
                **query_kwargs)

            while True:
                try:
                    response = await response_stream.__anext__()
                    callback(response)
                except StopAsyncIteration:
                    break

        self.asyncio_loop.run_until_complete(put_result())

    async def terminate_async(self) -> None:
        await self.stub.Terminate(
            modelresponse_pb2.google_dot_protobuf_dot_empty__pb2.Empty())

    def terminate_server(self) -> None:
        """
        Terminates the persistent deployment server. This can be called from any
        client.
        """
        self.asyncio_loop.run_until_complete(self.terminate_async())
        if self.mii_config.enable_restful_api:
            requests.get(
                f"http://localhost:{self.mii_config.restful_api_port}/terminate")


def get_prompt(task: str,use_empathy: bool,use_short: bool, use_instruction:bool):
    if task == 'conv_summarization':
        return "Read the given dialogue and summarize it.\n\n"
    elif task in ['contextual_user_query_rephrasing', 'contextual_user_query_rephrasing_negative']:
        return 'Rephrase the given last user message based on the given conversation history.\n\n'
    elif task == 'doc_qna':
        instruction = 'Answer the given question based only on the given knowledge.\n\n'
        if use_empathy:          
            instruction += "If needed, answer with an empathetic sentence (e.g. Im sorry to hear, I apologize, I empathize, etc).\n\n"
        if use_short:          
            instruction += "Always answer in concise and short form (<40 words).\n\n"
        
        if use_instruction:
            instruction += "If needed, answer with steps in every new line (e.g. 'Step 1: <content> \n', 'Step 2: <content> \n', etc).\n\n"
            
        return instruction
    else:
        print(f"task {task} is not supported yet")
        return

def get_knowledge(knwl_lst):
    out = ''
    for i,knowledge in enumerate(knwl_lst):
        out += '\n'
        out += f'<{i+1}> {knowledge}'
    
    return out


def get_token_length(tokenizer,text):
    encoded = tokenizer(text, 
                        return_tensors = "pt",
                        add_special_tokens=False)
    return encoded['input_ids'].shape[1]
