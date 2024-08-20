from threading import Thread, Lock
from typing import List, Optional, Sequence, Union, cast

import torch.cuda
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.inputs.data import PromptInputs
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.guided_decoding import get_local_guided_decoding_logits_processor
from vllm.model_executor.guided_decoding.guided_fields import GuidedDecodingRequest, LLMGuidedOptions
from vllm.outputs import EmbeddingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Counter

logger = init_logger(__name__)

class LLM:
    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: int = 4,
        cpu_offload_gb: float = 0,
        enforce_eager: Optional[bool] = None,
        max_context_len_to_capture: Optional[int] = None,
        max_seq_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        **kwargs,
    ) -> None:
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        removed_vision_keys = ("image_token_id", "image_feature_size",
                               "image_input_shape", "image_input_type")
        if any(k in kwargs for k in removed_vision_keys):
            raise TypeError(
                "There is no need to pass vision-related arguments anymore.")
        engine_args = EngineArgs(
            model=model,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            cpu_offload_gb=cpu_offload_gb,
            enforce_eager=enforce_eager,
            max_context_len_to_capture=max_context_len_to_capture,
            max_seq_len_to_capture=max_seq_len_to_capture,
            disable_custom_all_reduce=disable_custom_all_reduce,
            **kwargs,
        )
        self.streams = [torch.cuda.Stream() for _ in range(3)]
        self.llm_engines = []
        for stream in self.streams:
            with torch.cuda.stream(stream):
                self.llm_engines.append(LLMEngine.from_engine_args(
                    engine_args, usage_context=UsageContext.LLM_CLASS))
        self.request_counter = Counter()

    def get_tokenizer(
            self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        return self.llm_engines[0].tokenizer.tokenizer

    def generate(
        self,
        prompts: Union[Union[PromptInputs, Sequence[PromptInputs]],
                       Optional[Union[str, List[str]]]] = None,
        sampling_params: Optional[Union[SamplingParams,
                                        Sequence[SamplingParams]]] = None,
        prompt_token_ids: Optional[Union[List[int], List[List[int]]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        guided_options_request: Optional[Union[LLMGuidedOptions,
                                               GuidedDecodingRequest]] = None
    ) -> List[RequestOutput]:
        if self.llm_engines[0].model_config.embedding_mode:
            raise ValueError(
                "LLM.generate() is only supported for (conditional) generation "
                "models (XForCausalLM, XForConditionalGeneration).")

        assert prompt_token_ids is None
        inputs = cast(Union[PromptInputs, Sequence[PromptInputs]], prompts)

        if isinstance(guided_options_request, dict):
            if len(guided_options_request) > 1:
                raise ValueError(
                    "You can only use one guided decoding but multiple is "
                    f"specified: {guided_options_request}")
            guided_options_request = GuidedDecodingRequest(
                **guided_options_request)

        if sampling_params is None:
            # Use default sampling params.
            sampling_params = SamplingParams()

        self._validate_and_add_requests(
            inputs=inputs,
            params=sampling_params,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
            guided_options=guided_options_request)

        outputs = self._run_engine(use_tqdm=use_tqdm)
        return LLMEngine.validate_outputs(outputs, RequestOutput)

    def _validate_and_add_requests(
        self,
        inputs: Union[PromptInputs, Sequence[PromptInputs]],
        params: Union[SamplingParams, Sequence[SamplingParams], PoolingParams,
                      Sequence[PoolingParams]],
        lora_request: Optional[Union[Sequence[LoRARequest], LoRARequest]],
        prompt_adapter_request: Optional[PromptAdapterRequest],
        guided_options: Optional[GuidedDecodingRequest] = None,
    ) -> None:
        if isinstance(inputs, (str, dict)):
            # Convert a single prompt to a list.
            inputs = [inputs]

        num_requests = len(inputs)

        if isinstance(params, list) and len(params) != num_requests:
            raise ValueError("The lengths of prompts and params "
                             "must be the same.")
        if isinstance(lora_request,
                      list) and len(lora_request) != num_requests:
            raise ValueError("The lengths of prompts and lora_request "
                             "must be the same.")

        if isinstance(params, list):
            params = [
                self._add_guided_processor(param, guided_options)
                if isinstance(param, SamplingParams) else param
                for param in params
            ]
        elif isinstance(params, SamplingParams):
            params = self._add_guided_processor(params, guided_options)

        # Add requests to the engine.
        for i, request_inputs in enumerate(inputs):
            self._add_request(
                request_inputs,
                params[i] if isinstance(params, Sequence) else params,
                lora_request=lora_request[i] if isinstance(
                    lora_request, Sequence) else lora_request,
                prompt_adapter_request=prompt_adapter_request)

    def _add_request(
            self,
            inputs: PromptInputs,
            params: Union[SamplingParams, PoolingParams],
            lora_request: Optional[Union[List[LoRARequest],
                                         LoRARequest]] = None,
            prompt_adapter_request: Optional[PromptAdapterRequest] = None
    ) -> None:
        request_id = next(self.request_counter)
        engine_id = request_id % len(self.llm_engines)
        self.llm_engines[engine_id].add_request(
            str(request_id),
            inputs,
            params,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request)

    def _add_guided_processor(
            self,
            params: SamplingParams,
            guided_options: Optional[GuidedDecodingRequest] = None):
        if guided_options:
            if guided_options.guided_decoding_backend is None:
                decoding_config = self.llm_engines[0].get_decoding_config()
                guided_options.guided_decoding_backend = (
                    decoding_config.guided_decoding_backend)
            guided_logits_processor = get_local_guided_decoding_logits_processor(  #noqa
                guided_options.guided_decoding_backend, guided_options,
                self.get_tokenizer())
            if guided_logits_processor:
                if params.logits_processors is None:
                    params.logits_processors = []
                params.logits_processors.append(guided_logits_processor)
        return params

    def _run_engine(
            self, *, use_tqdm: bool
    ) -> List[Union[RequestOutput, EmbeddingRequestOutput]]:
        # Initialize tqdm.
        if use_tqdm:
            num_requests = sum(
                llm_engine.get_num_unfinished_requests() for llm_engine in
                self.llm_engines)
            pbar = tqdm(
                total=num_requests,
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix=(f"est. speed input: {0:.2f} toks/s, "
                         f"output: {0:.2f} toks/s"),
            )
        else:
            pbar = None
        # Run the engine.
        outputs: List[Union[RequestOutput, EmbeddingRequestOutput]] = []
        total_in_toks = 0
        total_out_toks = 0
        lock = Lock()

        def update_pbar(output):
            nonlocal total_in_toks, total_out_toks

            with lock:
                if isinstance(output, RequestOutput):
                    # Calculate tokens only for RequestOutput
                    total_in_toks += len(output.prompt_token_ids)
                    in_spd = total_in_toks / pbar.format_dict["elapsed"]
                    total_out_toks += sum(
                        len(stp.token_ids) for stp in output.outputs)
                    out_spd = total_out_toks / pbar.format_dict["elapsed"]
                    pbar.postfix = (
                        f"est. speed input: {in_spd:.2f} toks/s, "
                        f"output: {out_spd:.2f} toks/s")
                pbar.update(1)

        def run_engine(llm_engine: LLMEngine, stream: torch.cuda.Stream):
            with torch.cuda.stream(stream):
                while llm_engine.has_unfinished_requests():
                    step_outputs = llm_engine.step()
                    for output in step_outputs:
                        if output.finished:
                            outputs.append(output)
                            if pbar is not None:
                                    update_pbar(output)

        threads = [Thread(target=run_engine, args=(llm_engine, stream)) for
                   llm_engine, stream in zip(self.llm_engines, self.streams)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if pbar is not None:
            pbar.close()
        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        return sorted(outputs, key=lambda x: int(x.request_id))