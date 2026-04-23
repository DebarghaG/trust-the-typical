# TODO: when min batch size for prediction is not met and requests are finishing, they may not be predicted due to being freed by the processor.
from vllm import LLM, SamplingParams
import torch
from typing import Optional, Union

from forte_guardrails.forte_text_api import ForteTextOODDetector
from forte_guardrails.data_loader import load_safe_instructions, load_orbench_prompts_for_testing

class ForteGuardrails:
    def __init__(self):
        self.detector = None
    
    def init(self):
        torch.cuda.nvtx.range_push("guardrail_init")
        # Initialize llm guard
        self.detector = ForteTextOODDetector(
            embedding_dir='/tmp/forte_guardrails/embeddings',
            nearest_k=5,
            method="gmm",  # or "kde", "ocsvm"
            device="cuda:0",  # or "cpu", "mps"
            model_names=[
                ("qwen3", "Qwen/Qwen3-Embedding-0.6B"),
                ("bge-m3", "BAAI/bge-m3"),
                ("e5", "intfloat/e5-large-v2")
            ]
        )
        dataset_name = 'alpaca' ## change to hhh bench
        safe_train_texts, _ = load_safe_instructions(
            max_samples=1000, dataset_name=dataset_name
        )
        self.detector.fit(safe_train_texts)
        torch.cuda.nvtx.range_pop()
    
    def predict(self, input_text):
        torch.cuda.nvtx.range_push("guardrail_predict")
        if not isinstance(input_text, list):
            input_text = [input_text]
        
        string_list = input_text.copy()
        length = len(string_list)

        labels = self.detector.predict(string_list)[:length]
        torch.cuda.nvtx.range_pop()
        return labels


def init_guardrails(check_interval=20, min_batch_size=10):
    """Initialize guardrails and setup vLLM override."""
    try:
        guardrails = ForteGuardrails()
        guardrails.init()
        
        from vllm.v1.engine.output_processor import OutputProcessor, OutputProcessorOutput
        from vllm.v1.engine import EngineCoreOutput, FinishReason
        from vllm.v1.metrics.stats import IterationStats
        from vllm.outputs import RequestOutput, PoolingRequestOutput
        
        # Store the original process_outputs method
        _original_process_outputs = OutputProcessor.process_outputs
        
        def process_outputs_with_guardrails(
            self,
            engine_core_outputs: "list[EngineCoreOutput]",
            engine_core_timestamp: Optional[float] = None,
            iteration_stats: Optional["IterationStats"] = None,
        ) -> "OutputProcessorOutput":
            
            if not hasattr(self, 'all_requests'):
                self.all_requests = {} 
            
            candidates_for_prediction = []
            
            for engine_core_output in engine_core_outputs:
                req_id = engine_core_output.request_id
                req_state = self.request_states.get(req_id)
                if req_state is None:
                    continue
                
                # Update stats
                self._update_stats_from_output(req_state, engine_core_output,
                                               engine_core_timestamp, iteration_stats)
                
                new_token_ids = engine_core_output.new_token_ids
                pooling_output = engine_core_output.pooling_output
                finish_reason = engine_core_output.finish_reason
                stop_reason = engine_core_output.stop_reason
                kv_transfer_params = engine_core_output.kv_transfer_params
                req_state.num_cached_tokens = engine_core_output.num_cached_tokens
                req_state.is_prefilling = False
                
                # Process non-pooling outputs (text generation)
                if pooling_output is None:
                    assert req_state.detokenizer is not None
                    assert req_state.logprobs_processor is not None
                    
                    # Detokenize and get current text
                    stop_string = req_state.detokenizer.update(
                        new_token_ids, finish_reason == FinishReason.STOP)
                    if stop_string:
                        finish_reason = FinishReason.STOP
                        stop_reason = stop_string
                    
                    # Update logprobs
                    req_state.logprobs_processor.update_from_output(engine_core_output)
                    
                    current_text = req_state.detokenizer.output_text
                    word_count = len(current_text.split())
                    
                    if req_id not in self.all_requests:
                        self.all_requests[req_id] = {
                            'text': current_text,
                            'word_count': word_count,
                            'last_predicted_at': 0,    
                            'finish_reason': finish_reason,
                        }
                    else:
                        self.all_requests[req_id].update({
                            'text': current_text,
                            'word_count': word_count, 
                            'finish_reason': finish_reason,
                        })
                    
                    
                    last_predicted = self.all_requests[req_id]['last_predicted_at']
                    words_since_prediction = word_count - last_predicted
                    
                    if words_since_prediction >= check_interval or finish_reason is not None:
                        candidates_for_prediction.append(req_id)
            
            texts_for_guardrails = []
            req_ids_for_guardrails = []
            
            print(f"Candidates for prediction: {len(candidates_for_prediction)}")
            
            if candidates_for_prediction:
                print(f"Found {len(candidates_for_prediction)} candidates for prediction: {candidates_for_prediction}")
                
                # Add all candidates that need immediate prediction
                for req_id in candidates_for_prediction:
                    texts_for_guardrails.append(self.all_requests[req_id]['text'])
                    req_ids_for_guardrails.append(req_id)

                # If we don't have enough for samples, fill with requests close to their interval
                if len(texts_for_guardrails) < min_batch_size:
                    close_to_interval = []
                    for req_id, data in self.all_requests.items():
                        if data['finish_reason'] is None and req_id not in candidates_for_prediction:
                            words_since_prediction = data['word_count'] - data['last_predicted_at']
                            if words_since_prediction >= check_interval * 0.6: 
                                close_to_interval.append(req_id)
                    
                    for req_id in close_to_interval:
                        if len(texts_for_guardrails) >= min_batch_size:
                            break
                        texts_for_guardrails.append(self.all_requests[req_id]['text'])
                        req_ids_for_guardrails.append(req_id)
                        print(f"Added close-to-interval request {req_id} (words since prediction: {self.all_requests[req_id]['word_count'] - self.all_requests[req_id]['last_predicted_at']})")
                
                # Only if still not enough AND we have finishing requests, use historical requests
                if len(texts_for_guardrails) < min_batch_size:
                    if finished_requests := [
                        req_id for req_id, data in self.all_requests.items()
                        if data['finish_reason'] is not None and req_id not in req_ids_for_guardrails
                    ]:
                        for req_id in finished_requests:
                            if len(texts_for_guardrails) >= min_batch_size:
                                break
                            texts_for_guardrails.append(self.all_requests[req_id]['text'])
                            req_ids_for_guardrails.append(req_id)
                            print(f"Added finished request {req_id} to ensure {min_batch_size} texts for finishing requests")
                    else:
                        print(f"Insufficient samples ({len(texts_for_guardrails)}/{min_batch_size}) and no finished requests available - skipping prediction")
                
                print(f"Assembled batch: {len(texts_for_guardrails)} texts (min required: {min_batch_size})")

            if len(texts_for_guardrails) >= min_batch_size:
                try:
                    torch.cuda.nvtx.range_push("batch_guardrails_predict")
                    print(f"Running guardrails on {len(texts_for_guardrails)} texts")
                    
                    labels = guardrails.detector.predict(texts_for_guardrails)
                    
                    # Map results back to request IDs and update prediction state
                    for i, req_id in enumerate(req_ids_for_guardrails):
                        self.all_requests[req_id]['last_predicted_at'] = self.all_requests[req_id]['word_count']
                        
                        if labels[i] < 1:  # Toxic content detected
                            for engine_core_output in engine_core_outputs:
                                if engine_core_output.request_id == req_id:
                                    engine_core_output.finish_reason = FinishReason.STOP
                                    engine_core_output.stop_reason = 'forte_guardrails_stop'
                                    break
                            
                            print(f"Guardrails blocked request {req_id}")
                    
                    torch.cuda.nvtx.range_pop()
                except Exception as e:
                    print(f"Guardrails prediction failed: {e}")
                    torch.cuda.nvtx.range_pop()  
            else:
                if len(texts_for_guardrails) > 0:
                    print(f"Respecting intervals: only {len(texts_for_guardrails)} texts available, need {min_batch_size} for PRDC. Waiting for more requests to reach their {check_interval}-word intervals.")
                else:
                    print(f"No candidates for prediction - all requests still within their {check_interval}-word intervals")
            
            request_outputs: "Union[list[RequestOutput], list[PoolingRequestOutput]]" = []
            reqs_to_abort: list[str] = []
            
            for engine_core_output in engine_core_outputs:
                req_id = engine_core_output.request_id
                req_state = self.request_states.get(req_id)
                if req_state is None:
                    continue
                
                new_token_ids = engine_core_output.new_token_ids
                pooling_output = engine_core_output.pooling_output
                finish_reason = engine_core_output.finish_reason
                stop_reason = engine_core_output.stop_reason
                kv_transfer_params = engine_core_output.kv_transfer_params
                
                # Create and handle RequestOutput objects
                if request_output := req_state.make_request_output(
                        new_token_ids, pooling_output, finish_reason, stop_reason,
                        kv_transfer_params):
                    if req_state.queue is not None:
                        # AsyncLLM: put into queue for handling by generate()
                        req_state.queue.put(request_output)
                    else:
                        # LLMEngine: return list of RequestOutputs
                        request_outputs.append(request_output)
                
                # Free completed requests
                if finish_reason is not None:
                    # Mark as inactive in our tracking
                    if req_id in self.all_requests:
                        self.all_requests[req_id]['finish_reason'] = finish_reason
                    
                    # Clean up vLLM state
                    self.request_states.pop(req_id)
                    # Remove parent request if applicable
                    parent_req = req_state.parent_req
                    if parent_req and not parent_req.child_requests:
                        self.parent_requests.pop(parent_req.request_id, None)
                    if not engine_core_output.finished:
                        # If req not finished in EngineCore, but Detokenizer
                        # detected stop string, abort needed in EngineCore.
                        reqs_to_abort.append(req_id)
                    
                    # Track per-request stats
                    self._update_stats_from_finished(req_state, finish_reason, iteration_stats)
                    if self.tracer:
                        self.do_tracing(engine_core_output, req_state, iteration_stats)
            
            self.lora_states.update_iteration_stats(iteration_stats)
            
            return OutputProcessorOutput(
                request_outputs=request_outputs,
                reqs_to_abort=reqs_to_abort,
            )
        
        OutputProcessor.process_outputs = process_outputs_with_guardrails
        
    except Exception as e:
        print(f"Guardrails initialization failed: {e}")


def run_vllm():   
    prompts_pool = load_orbench_prompts_for_testing(max_samples=500, include_toxic=True)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)
    llm = LLM(model="facebook/opt-125m")
    outputs = llm.generate(prompts_pool, sampling_params)
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print(f'finish_reason:    {output.outputs[0].finish_reason}')
        print(f'stop_reason:    {output.outputs[0].stop_reason}')
        print("-" * 60)


if __name__ == "__main__":
    init_guardrails(check_interval=20, min_batch_size=10)
    run_vllm()