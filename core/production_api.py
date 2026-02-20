#!/usr/bin/env python3
"""
VRAMancer API — Production Server
OpenAI-compatible REST API: health, monitoring, inference (generate/completions/chat).
"""
import os
import sys
import time
import uuid
import json
import threading
import queue
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, jsonify, request, Response
from core.logger import get_logger

# Import extracted sub-modules
from core.api.registry import PipelineRegistry
from core.api.validation import validate_generation_params, count_tokens

# Logger
logger = get_logger('api.production')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_HOST = os.environ.get('VRM_API_HOST', '0.0.0.0')
API_PORT = int(os.environ.get('VRM_API_PORT', '5030'))
API_DEBUG = os.environ.get('VRM_API_DEBUG', '0') in {'1', 'true', 'TRUE'}
os.environ.setdefault('VRM_API_BASE', f'http://localhost:{API_PORT}')

# Inference queue settings
_INFERENCE_TIMEOUT = int(os.environ.get('VRM_INFERENCE_TIMEOUT', '120'))
_MAX_QUEUE_SIZE = int(os.environ.get('VRM_MAX_QUEUE_SIZE', '32'))

# ---------------------------------------------------------------------------
# Pipeline registry (extracted to core.api.registry)
# PipelineRegistry class is imported from core.api.registry
# ---------------------------------------------------------------------------

# Module-level registry instance
_registry = PipelineRegistry()


# ---------------------------------------------------------------------------
# Input validation (extracted to core.api.validation)
# ---------------------------------------------------------------------------

# Backward-compatible aliases
_validate_generation_params = validate_generation_params
_count_tokens = count_tokens


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_app(model_name: Optional[str] = None,
               backend: str = "auto",
               num_gpus: Optional[int] = None) -> Flask:
    """Flask app factory with optional model pre-loading.

    Parameters
    ----------
    model_name : str, optional
        If provided, load the model at startup.
    backend : str
        Backend to use (auto, huggingface, vllm, ollama).
    num_gpus : int, optional
        Number of GPUs (auto-detect if None).

    Returns
    -------
    Flask
        Configured Flask application.
    """
    application = Flask(__name__)
    application.config['JSON_SORT_KEYS'] = False
    application.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

    # Security middleware
    try:
        from core.security import install_security
        install_security(application)
        
        # Enforce zero-trust startup checks
        try:
            from core.security.zero_trust import enforce_zero_trust_startup
            enforce_zero_trust_startup()
        except ImportError:
            pass
    except ImportError:
        logger.warning("Security module not available — running without auth")

    # Circuit-breaker for inference protection
    try:
        from core.api.circuit_breaker import CircuitBreaker, CircuitOpenError
        _circuit_breaker = CircuitBreaker(
            failure_threshold=int(os.environ.get('VRM_CB_FAILURE_THRESHOLD', '5')),
            recovery_timeout=float(os.environ.get('VRM_CB_RECOVERY_TIMEOUT', '30')),
            name="inference",
        )
    except ImportError:
        _circuit_breaker = None
        CircuitOpenError = Exception  # noqa: N818

    # Inference executor (tied to app lifecycle)
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=int(os.environ.get('VRM_MAX_CONCURRENT', '4')),
        thread_name_prefix="vrm-infer",
    )
    # Store executor on app for graceful shutdown
    application.vrm_executor = executor
    queue_depth = [0]  # mutable for closure
    queue_lock = threading.Lock()

    def _run_with_timeout(fn, timeout_s=None):
        """Run fn with timeout, queue backpressure and circuit-breaker.

        Returns (result, error_tuple_or_None).
        """
        timeout = timeout_s or _INFERENCE_TIMEOUT

        # Circuit-breaker check
        if _circuit_breaker and not _circuit_breaker.allow_request():
            return None, (
                "Service temporarily unavailable — circuit breaker open "
                "(too many recent failures)", 503
            )

        with queue_lock:
            if queue_depth[0] >= _MAX_QUEUE_SIZE:
                return None, ("Queue full — server overloaded, try again later", 503)
            queue_depth[0] += 1

        try:
            future = executor.submit(fn)
            try:
                result = future.result(timeout=timeout)
                if _circuit_breaker:
                    _circuit_breaker.record_success()
                return result, None
            except concurrent.futures.TimeoutError:
                future.cancel()
                if _circuit_breaker:
                    _circuit_breaker.record_failure()
                return None, ("Inference timeout — request took too long", 504)
            except Exception:
                if _circuit_breaker:
                    _circuit_breaker.record_failure()
                raise
        finally:
            with queue_lock:
                queue_depth[0] -= 1

    # Register all routes
    _register_routes(application, _run_with_timeout, queue_depth, queue_lock,
                      _circuit_breaker)

    # Pre-load model if requested
    if model_name:
        try:
            _registry.load(model_name, backend=backend,
                           num_gpus=num_gpus, verbose=True)
            logger.info("Model pre-loaded: %s", model_name)
        except Exception as e:
            logger.error("Failed to pre-load model: %s", e)

    return application


# ============================================================================
# Route registration
# ============================================================================

def _register_routes(application: Flask, _run_with_timeout, queue_depth, queue_lock,
                      _circuit_breaker=None):
    """Register all routes on a Flask app instance."""

    @application.before_request
    def log_request():
        if API_DEBUG:
            logger.debug(
                "Request: %s %s", request.method, request.path,
                extra={'context': {
                    'method': request.method,
                    'path': request.path,
                    'remote_addr': request.remote_addr,
                    'user_agent': request.headers.get('User-Agent', 'Unknown'),
                }}
            )

    @application.after_request
    def log_response(response):
        if API_DEBUG:
            logger.debug(
                "Response: %s for %s", response.status_code, request.path,
                extra={'context': {
                    'status_code': response.status_code,
                    'path': request.path,
                }}
            )
        return response

    # ====================================================================
    # SSE guard: circuit-breaker + queue + error handling for streaming
    # ====================================================================

    def _guarded_sse(gen_fn, path_label):
        """Wrap an SSE generator with circuit-breaker and queue protection.

        Checks the circuit-breaker before streaming begins, counts
        in-flight SSE requests in queue_depth, and records
        success/failure when the generator completes or errors.
        """
        # 1. Circuit-breaker check
        if _circuit_breaker and not _circuit_breaker.allow_request():
            def _cb_error():
                yield ('data: {"error": {"message": '
                       '"Service temporarily unavailable — circuit breaker open", '
                       '"type": "server_error"}}\n\n')
            return Response(_cb_error(), mimetype='text/event-stream',
                            status=503,
                            headers={'Cache-Control': 'no-cache'})

        # 2. Queue backpressure
        with queue_lock:
            if queue_depth[0] >= _MAX_QUEUE_SIZE:
                def _q_error():
                    yield ('data: {"error": {"message": '
                           '"Queue full — server overloaded, try again later", '
                           '"type": "server_error"}}\n\n')
                return Response(_q_error(), mimetype='text/event-stream',
                                status=503,
                                headers={'Cache-Control': 'no-cache'})
            queue_depth[0] += 1

        # 3. Wrapped generator with cleanup
        def _wrapped():
            try:
                _start = time.perf_counter()
                yield from gen_fn()
                if _circuit_breaker:
                    _circuit_breaker.record_success()
                _elapsed = time.perf_counter() - _start
                try:
                    from core.metrics import API_LATENCY
                    API_LATENCY.labels(
                        path=path_label, method='POST', status='200'
                    ).observe(_elapsed)
                except Exception:
                    pass
            except Exception as exc:
                if _circuit_breaker:
                    _circuit_breaker.record_failure()
                yield (f'data: {json.dumps({"error": {"message": str(exc), "type": "server_error"}})}\n\n')
            finally:
                with queue_lock:
                    queue_depth[0] -= 1

        return Response(
            _wrapped(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
                'Connection': 'keep-alive',
            },
        )

    @application.errorhandler(404)
    def not_found(error):
        logger.warning("404 Not Found: %s", request.path)
        return jsonify({
            'error': 'Not Found',
            'message': f'Endpoint {request.path} does not exist',
            'status': 404,
        }), 404

    @application.errorhandler(500)
    def internal_error(error):
        logger.error("500 Internal Error: %s", error, exc_info=True)
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred',
            'status': 500,
        }), 500

    # ====================================================================
    # Inference endpoints
    # ====================================================================

    def _ensure_model(model_name: Optional[str]) -> Optional[Tuple[dict, int]]:
        """Ensure a model is loaded. Returns error response or None."""
        if _registry.is_loaded():
            if model_name and model_name != _registry.model_name:
                _registry.load(model_name)
            return None
        if not model_name:
            return jsonify({
                'error': 'No model loaded. '
                         'Send {"model": "gpt2", "prompt": "..."} '
                         'or pre-load via serve --model.',
            }), 400
        _registry.load(model_name)
        return None

    @application.route('/v1/completions', methods=['POST'])
    @application.route('/api/generate', methods=['POST'])
    def generate():
        """OpenAI-compatible completion endpoint.

        Request body (JSON):
            prompt: str — input text
            model: str — model name (optional if already loaded)
            max_tokens: int — max tokens to generate (default 128, 1-4096)
            temperature: float — sampling temperature (default 1.0, 0.0-2.0)
            top_p: float — nucleus sampling (default 1.0, 0.0-1.0)
            stream: bool — SSE streaming (default false)
        """
        data = request.get_json(silent=True) or {}
        prompt = data.get('prompt', '')
        model_name = data.get('model')
        stream = data.get('stream', False)

        if not prompt:
            return jsonify({'error': 'Missing "prompt" field'}), 400

        params, val_err = _validate_generation_params(data)
        if val_err:
            return jsonify({'error': val_err[0]}), val_err[1]

        try:
            err_resp = _ensure_model(model_name)
            if err_resp:
                return err_resp

            # Metrics
            try:
                from core.metrics import INFER_REQUESTS, INFER_LATENCY, API_LATENCY
                INFER_REQUESTS.inc()
            except Exception:
                pass

            req_id = "vrm-" + uuid.uuid4().hex[:12]

            # --- Streaming SSE (protected by circuit-breaker + queue) ---
            if stream:
                def _sse_generate():
                    """Stream tokens as Server-Sent Events."""
                    for token_text in _registry.generate_stream(
                        prompt,
                        max_new_tokens=params['max_tokens'],
                        temperature=params['temperature'],
                        top_p=params['top_p'],
                        top_k=params['top_k'],
                    ):
                        chunk = {
                            "id": req_id,
                            "object": "text_completion",
                            "created": int(time.time()),
                            "model": _registry.model_name or model_name,
                            "choices": [{
                                "text": token_text,
                                "index": 0,
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                    final_chunk = {
                        "id": req_id,
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": _registry.model_name or model_name,
                        "choices": [{
                            "text": "",
                            "index": 0,
                            "finish_reason": "stop",
                        }],
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

                return _guarded_sse(_sse_generate, '/v1/completions')

            # --- Non-streaming response ---
            start = time.perf_counter()

            def _do_generate():
                return _registry.generate(
                    prompt,
                    max_new_tokens=params['max_tokens'],
                    temperature=params['temperature'],
                    top_p=params['top_p'],
                    top_k=params['top_k'],
                )

            text, queue_err = _run_with_timeout(_do_generate)
            if queue_err:
                return jsonify({'error': queue_err[0]}), queue_err[1]

            elapsed = time.perf_counter() - start

            tokenizer = _registry.get_tokenizer()
            prompt_tokens = _count_tokens(prompt, tokenizer)
            completion_tokens = _count_tokens(text, tokenizer)

            try:
                API_LATENCY.labels(path='/v1/completions', method='POST', status='200').observe(elapsed)
            except Exception:
                pass

            return jsonify({
                'id': req_id,
                'object': 'text_completion',
                'created': int(time.time()),
                'model': _registry.model_name or model_name,
                'choices': [{
                    'text': text,
                    'index': 0,
                    'finish_reason': 'stop',
                }],
                'usage': {
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': prompt_tokens + completion_tokens,
                },
                'timing': {
                    'total_seconds': round(elapsed, 4),
                },
            })

        except Exception as e:
            logger.error("Generate failed: %s", e, exc_info=True)
            try:
                from core.metrics import INFER_ERRORS
                INFER_ERRORS.inc()
            except Exception:
                pass
            return jsonify({'error': str(e)}), 500

    @application.route('/v1/chat/completions', methods=['POST'])
    def chat_completions():
        """OpenAI-compatible chat completions endpoint.

        Request body (JSON):
            model: str — model name
            messages: list[{role, content}] — conversation messages
            max_tokens: int — max tokens to generate (1-4096)
            temperature: float — sampling temperature (0.0-2.0)
            top_p: float — nucleus sampling (0.0-1.0)
            stream: bool — SSE streaming
        """
        data = request.get_json(silent=True) or {}
        messages = data.get('messages', [])
        model_name = data.get('model')
        stream = data.get('stream', False)

        if not messages:
            return jsonify({'error': 'Missing "messages" field'}), 400

        params, val_err = _validate_generation_params(data)
        if val_err:
            return jsonify({'error': val_err[0]}), val_err[1]

        # Convert messages to prompt
        prompt_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
            else:
                prompt_parts.append(f"User: {content}")
        prompt_parts.append("Assistant:")
        prompt = "\n".join(prompt_parts)

        try:
            err_resp = _ensure_model(model_name)
            if err_resp:
                return err_resp

            # Metrics
            try:
                from core.metrics import INFER_REQUESTS, API_LATENCY
                INFER_REQUESTS.inc()
            except Exception:
                pass

            req_id = "chatcmpl-" + uuid.uuid4().hex[:12]

            if stream:
                def _sse_chat():
                    for token_text in _registry.generate_stream(
                        prompt,
                        max_new_tokens=params['max_tokens'],
                        temperature=params['temperature'],
                        top_p=params['top_p'],
                    ):
                        chunk = {
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": _registry.model_name or model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": token_text},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                    final = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": _registry.model_name or model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }],
                    }
                    yield f"data: {json.dumps(final)}\n\n"
                    yield "data: [DONE]\n\n"

                return _guarded_sse(_sse_chat, '/v1/chat/completions')

            # Non-streaming with timeout
            start = time.perf_counter()

            def _do_chat():
                return _registry.generate(
                    prompt,
                    max_new_tokens=params['max_tokens'],
                    temperature=params['temperature'],
                    top_p=params['top_p'],
                )

            text, queue_err = _run_with_timeout(_do_chat)
            if queue_err:
                return jsonify({'error': queue_err[0]}), queue_err[1]

            elapsed = time.perf_counter() - start

            tokenizer = _registry.get_tokenizer()
            prompt_tokens = _count_tokens(prompt, tokenizer)
            completion_tokens = _count_tokens(text, tokenizer)

            try:
                API_LATENCY.labels(path='/v1/chat/completions', method='POST', status='200').observe(elapsed)
            except Exception:
                pass

            return jsonify({
                'id': req_id,
                'object': 'chat.completion',
                'created': int(time.time()),
                'model': _registry.model_name or model_name,
                'choices': [{
                    'index': 0,
                    'message': {'role': 'assistant', 'content': text},
                    'finish_reason': 'stop',
                }],
                'usage': {
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': prompt_tokens + completion_tokens,
                },
                'timing': {
                    'total_seconds': round(elapsed, 4),
                },
            })

        except Exception as e:
            logger.error("Chat completion failed: %s", e, exc_info=True)
            try:
                from core.metrics import INFER_ERRORS
                INFER_ERRORS.inc()
            except Exception:
                pass
            return jsonify({'error': str(e)}), 500

    @application.route('/api/infer', methods=['POST'])
    def infer():
        """Raw tensor inference endpoint (for programmatic use).

        Request body (JSON):
            input_ids: list[int] — token IDs
            model: str — model name (optional if already loaded)
        """
        data = request.get_json(silent=True) or {}
        input_ids_raw = data.get('input_ids', [])
        model_name = data.get('model')

        if not input_ids_raw:
            return jsonify({'error': 'Missing "input_ids" field'}), 400

        try:
            err_resp = _ensure_model(model_name)
            if err_resp:
                return err_resp

            import torch as _torch
            input_tensor = _torch.tensor([input_ids_raw], dtype=_torch.long)
            if _torch.cuda.is_available():
                input_tensor = input_tensor.cuda()

            def _do_infer():
                return _registry.infer(input_tensor)

            output, queue_err = _run_with_timeout(_do_infer)
            if queue_err:
                return jsonify({'error': queue_err[0]}), queue_err[1]

            if hasattr(output, 'tolist'):
                output_list = output.detach().cpu().tolist()
                shape = list(output.shape)
            else:
                output_list = output
                shape = []

            return jsonify({
                'logits': output_list,
                'shape': shape,
                'model': _registry.model_name,
            })

        except Exception as e:
            logger.error("Infer failed: %s", e, exc_info=True)
            return jsonify({'error': str(e)}), 500

    @application.route('/api/models', methods=['GET'])
    def list_models():
        """List loaded model info."""
        if _registry.is_loaded():
            backend = _registry.backend
            return jsonify({
                'models': [{
                    'id': _registry.model_name,
                    'backend': type(backend).__name__ if backend else 'unknown',
                    'num_gpus': _registry.num_gpus,
                    'num_blocks': len(_registry.blocks),
                    'status': 'loaded',
                }]
            })
        return jsonify({'models': [], 'message': 'No model loaded'})

    # ====================================================================
    # Batch inference endpoint
    # ====================================================================

    @application.route('/v1/batch/completions', methods=['POST'])
    def batch_completions():
        """Batch completion endpoint — submit multiple prompts at once.

        Request body (JSON):
            prompts: list[str] — list of input prompts (max 32)
            model: str — model name (optional if already loaded)
            max_tokens: int — max tokens per prompt (default 128, 1-4096)
            temperature: float — sampling temperature (default 1.0)
            top_p: float — nucleus sampling (default 1.0)

        Response:
            results: list[{text, index, finish_reason}]
            usage: {prompt_tokens, completion_tokens, total_tokens}
            timing: {total_seconds, avg_per_prompt}
        """
        data = request.get_json(silent=True) or {}
        prompts = data.get('prompts', [])
        model_name = data.get('model')

        if not prompts or not isinstance(prompts, list):
            return jsonify({'error': 'Missing or invalid "prompts" field (must be a list)'}), 400

        max_batch = int(os.environ.get('VRM_MAX_BATCH_SIZE', '32'))
        if len(prompts) > max_batch:
            return jsonify({
                'error': f'Batch too large: {len(prompts)} prompts (max {max_batch})',
            }), 400

        params, val_err = _validate_generation_params(data)
        if val_err:
            return jsonify({'error': val_err[0]}), val_err[1]

        try:
            err_resp = _ensure_model(model_name)
            if err_resp:
                return err_resp

            try:
                from core.metrics import INFER_REQUESTS, API_LATENCY
                INFER_REQUESTS.inc(len(prompts))
            except Exception:
                pass

            req_id = "vrm-batch-" + uuid.uuid4().hex[:12]
            start = time.perf_counter()

            results = []
            total_prompt_tokens = 0
            total_completion_tokens = 0
            tokenizer = _registry.get_tokenizer()

            def _do_batch():
                batch_results = []
                for i, prompt in enumerate(prompts):
                    text = _registry.generate(
                        prompt,
                        max_new_tokens=params['max_tokens'],
                        temperature=params['temperature'],
                        top_p=params['top_p'],
                        top_k=params['top_k'],
                    )
                    batch_results.append((i, prompt, text))
                return batch_results

            batch_results, queue_err = _run_with_timeout(
                _do_batch,
                timeout_s=_INFERENCE_TIMEOUT * len(prompts),
            )
            if queue_err:
                return jsonify({'error': queue_err[0]}), queue_err[1]

            for i, prompt, text in batch_results:
                p_tokens = _count_tokens(prompt, tokenizer)
                c_tokens = _count_tokens(text, tokenizer)
                total_prompt_tokens += p_tokens
                total_completion_tokens += c_tokens
                results.append({
                    'text': text,
                    'index': i,
                    'finish_reason': 'stop',
                })

            elapsed = time.perf_counter() - start

            try:
                API_LATENCY.labels(
                    path='/v1/batch/completions', method='POST', status='200'
                ).observe(elapsed)
            except Exception:
                pass

            return jsonify({
                'id': req_id,
                'object': 'text_completion_batch',
                'created': int(time.time()),
                'model': _registry.model_name or model_name,
                'results': results,
                'usage': {
                    'prompt_tokens': total_prompt_tokens,
                    'completion_tokens': total_completion_tokens,
                    'total_tokens': total_prompt_tokens + total_completion_tokens,
                },
                'timing': {
                    'total_seconds': round(elapsed, 4),
                    'avg_per_prompt': round(elapsed / max(1, len(prompts)), 4),
                    'batch_size': len(prompts),
                },
            })

        except Exception as e:
            logger.error("Batch completions failed: %s", e, exc_info=True)
            try:
                from core.metrics import INFER_ERRORS
                INFER_ERRORS.inc()
            except Exception:
                pass
            return jsonify({'error': str(e)}), 500

    # ====================================================================
    # Continuous batching stats
    # ====================================================================

    @application.route('/api/batcher/stats', methods=['GET'])
    def batcher_stats():
        """Return continuous batcher statistics.

        Shows throughput (tok/s), queue depth, active batch size,
        and request counts for the continuous batching engine.
        """
        pipeline = _registry.get()
        if pipeline and hasattr(pipeline, 'batcher_stats'):
            stats = pipeline.batcher_stats()
            if stats:
                return jsonify(stats)
        return jsonify({
            'status': 'unavailable',
            'message': 'Continuous batcher not initialized',
        })

    # ====================================================================
    # Benchmark endpoint
    # ====================================================================

    @application.route('/api/benchmark', methods=['POST'])
    def run_benchmark():
        """Run inference benchmark and return performance metrics.

        Request body (JSON):
            model: str — model to benchmark (uses loaded model if omitted)
            num_prompts: int — number of prompts (default 10)
            max_tokens: int — tokens per prompt (default 128)
            mode: str — "sequential" | "concurrent" | "continuous" (default sequential)

        Returns tok/s, latency percentiles, VRAM usage.
        """
        data = request.get_json(silent=True) or {}

        try:
            from core.benchmark import BenchmarkRunner
            runner = BenchmarkRunner(verbose=True)

            model_name = data.get('model', _registry.model_name or 'gpt2')
            num_prompts = min(data.get('num_prompts', 10), 50)
            max_tokens = min(data.get('max_tokens', 128), 512)
            mode = data.get('mode', 'sequential')

            result = runner.run(
                model_name=model_name,
                max_new_tokens=max_tokens,
                num_concurrent=4 if mode == 'concurrent' else 1,
                use_continuous_batching=(mode == 'continuous'),
            )

            return jsonify({
                'status': 'completed',
                'result': result.to_dict(),
            })
        except Exception as e:
            logger.error("Benchmark failed: %s", e, exc_info=True)
            return jsonify({'error': str(e)}), 500

    @application.route('/api/models/load', methods=['POST'])
    def load_model():
        """Load a model into the pipeline.

        Request body: {"model": "gpt2", "num_gpus": 2, "backend": "auto"}
        """
        data = request.get_json(silent=True) or {}
        model_name = data.get('model')
        if not model_name:
            return jsonify({'error': 'Missing "model" field'}), 400

        num_gpus = data.get('num_gpus')
        backend = data.get('backend', 'auto')

        try:
            _registry.load(model_name, backend=backend, num_gpus=num_gpus)
            return jsonify({
                'status': 'loaded',
                'model': model_name,
                'num_gpus': _registry.num_gpus,
                'num_blocks': len(_registry.blocks),
            })
        except Exception as e:
            logger.error("Model load failed: %s", e, exc_info=True)
            return jsonify({'error': str(e)}), 500

    @application.route('/api/pipeline/status', methods=['GET'])
    def pipeline_status():
        """Pipeline status with subsystem details."""
        return jsonify(_registry.status())

    @application.route('/api/queue/status', methods=['GET'])
    def queue_status():
        """Inference queue status and backpressure info."""
        resp = {
            'queue_depth': queue_depth[0],
            'max_queue_size': _MAX_QUEUE_SIZE,
            'max_concurrent': int(os.environ.get('VRM_MAX_CONCURRENT', '4')),
            'inference_timeout_s': _INFERENCE_TIMEOUT,
            'utilization_pct': round(
                (queue_depth[0] / max(1, _MAX_QUEUE_SIZE)) * 100, 1
            ),
        }
        if _circuit_breaker:
            resp['circuit_breaker'] = _circuit_breaker.status()
        return jsonify(resp)

    # ====================================================================
    # Health / readiness / liveness
    # ====================================================================

    @application.route('/health')
    def health():
        """Health check (Kubernetes/Docker ready)."""
        checks = {'process': True}
        overall = 'healthy'
        try:
            from core.monitor import GPUMonitor
            checks['monitor'] = True
        except Exception:
            checks['monitor'] = False
        try:
            from core.config import get_config
            checks['config'] = True
        except Exception:
            checks['config'] = False
        try:
            import psutil
            mem = psutil.virtual_memory()
            checks['memory_available_mb'] = int(mem.available / 1024 / 1024)
            if mem.available < 500 * 1024 * 1024:
                overall = 'degraded'
                checks['memory_warning'] = True
        except ImportError:
            checks['memory_available_mb'] = -1
        try:
            from core import __version__
            version = __version__
        except Exception:
            version = '0.2.4'
        checks['pipeline_loaded'] = _registry.is_loaded()
        if not all(checks.get(k, False) for k in ('monitor', 'config')):
            overall = 'degraded'
        return jsonify({
            'status': overall, 'service': 'vramancer-api',
            'version': version, 'checks': checks,
        }), 200

    @application.route('/ready')
    def ready():
        """Readiness check."""
        try:
            from core.utils import detect_backend, enumerate_devices
            backend = detect_backend()
            devices = enumerate_devices()
            has_gpu = any(d['backend'] in ('cuda', 'rocm', 'mps') for d in devices)
            return jsonify({
                'status': 'ready', 'backend': backend,
                'devices': len(devices), 'has_gpu': has_gpu,
                'model_loaded': _registry.is_loaded(),
                'service': 'vramancer-api',
            })
        except Exception as e:
            logger.error("Readiness check failed: %s", e)
            return jsonify({'status': 'not_ready', 'error': str(e)}), 503

    @application.route('/live')
    def live():
        """Liveness probe."""
        return jsonify({'status': 'alive'}), 200

    @application.route('/metrics')
    def metrics_endpoint():
        """Expose Prometheus metrics on the main API port.

        Proxies the prometheus_client registry so that a single port
        serves both the REST API and the /metrics scrape target.
        """
        try:
            from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
            return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)
        except ImportError:
            return Response("# prometheus_client not installed\n",
                            mimetype="text/plain"), 501

    @application.route('/api/status')
    def api_status():
        """Detailed API status with all endpoints."""
        try:
            from core import __version__
            version = __version__
        except Exception:
            version = '0.2.4'
        return jsonify({
            'backend': 'running', 'version': version,
            'api_base': os.environ.get('VRM_API_BASE', f'http://localhost:{API_PORT}'),
            'endpoints': {
                'generate': 'POST /v1/completions',
                'generate_alt': 'POST /api/generate',
                'chat': 'POST /v1/chat/completions',
                'batch': 'POST /v1/batch/completions',
                'infer': 'POST /api/infer',
                'models': 'GET /api/models',
                'load_model': 'POST /api/models/load',
                'pipeline': 'GET /api/pipeline/status',
                'queue': 'GET /api/queue/status',
                'health': 'GET /health',
                'ready': 'GET /ready',
                'live': 'GET /live',
                'metrics': 'GET /metrics',
                'gpu': 'GET /api/gpu',
                'system': 'GET /api/system',
                'nodes': 'GET /api/nodes',

            },
        })

    @application.route('/api/gpu/info')
    @application.route('/api/gpu')
    def gpu_info():
        """GPU information."""
        try:
            import torch
            if not torch.cuda.is_available():
                return jsonify({'cuda_available': False, 'devices': [],
                                'message': 'CUDA not available'})
            devices = []
            for i in range(torch.cuda.device_count()):
                try:
                    props = torch.cuda.get_device_properties(i)
                    alloc = torch.cuda.memory_allocated(i)
                    total = props.total_memory
                    devices.append({
                        'id': i, 'name': props.name,
                        'memory_used': alloc, 'memory_total': total,
                        'memory_free': total - alloc,
                        'memory_usage_percent': round(
                            (alloc / total) * 100, 2
                        ) if total else 0,
                        'compute_capability': f"{props.major}.{props.minor}",
                    })
                except Exception as e:
                    devices.append({'id': i, 'error': str(e)})
            return jsonify({
                'cuda_available': True, 'device_count': len(devices),
                'devices': devices, 'cuda_version': torch.version.cuda,
            })
        except ImportError:
            return jsonify({'cuda_available': False, 'devices': [],
                            'message': 'PyTorch not installed'}), 503
        except Exception as e:
            return jsonify({'error': str(e), 'cuda_available': False}), 500

    @application.route('/api/system')
    def system_info():
        """System information."""
        try:
            import psutil, platform
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            freq = psutil.cpu_freq()
            return jsonify({
                'platform': platform.system(),
                'architecture': platform.machine(),
                'cpu': {
                    'count_logical': psutil.cpu_count(logical=True),
                    'count_physical': psutil.cpu_count(logical=False),
                    'percent': round(psutil.cpu_percent(interval=0.1), 2),
                    'frequency_mhz': round(freq.current) if freq else None,
                },
                'memory': {
                    'total_gb': round(mem.total / (1024**3), 2),
                    'available_gb': round(mem.available / (1024**3), 2),
                    'percent': round(mem.percent, 2),
                },
                'disk': {
                    'total_gb': round(disk.total / (1024**3), 2),
                    'free_gb': round(disk.free / (1024**3), 2),
                    'percent': round(disk.percent, 2),
                },
            })
        except ImportError:
            return jsonify({'error': 'psutil not installed'}), 503
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @application.route('/api/nodes')
    def nodes_info():
        """Cluster node information."""
        try:
            import psutil, platform
            mem = psutil.virtual_memory()
            cpu_pct = psutil.cpu_percent(interval=0.1)
            uptime_s = time.time() - psutil.boot_time()
            gpu_data = {'name': 'No GPU', 'vram': 0, 'count': 0}
            try:
                import torch
                if torch.cuda.is_available():
                    props = torch.cuda.get_device_properties(0)
                    gpu_data = {
                        'name': props.name,
                        'vram': round(props.total_memory / (1024**2)),
                        'count': torch.cuda.device_count(),
                    }
            except Exception:
                pass
            discovered = []
            try:
                nodes = _registry.get_nodes()
                for hostname, info in nodes.items():
                    discovered.append({'hostname': hostname, **info})
            except Exception:
                pass
            local_node = {
                'host': 'localhost',
                'name': f'VRAMancer-{platform.node()}',
                'status': 'active', 'role': 'master',
                'os': f"{platform.system()} {platform.release()}",
                'cpu': psutil.cpu_count(logical=True),
                'memory': round(mem.total / (1024**3), 1),
                'vram': gpu_data['vram'], 'gpu_name': gpu_data['name'],
                'gpu_count': gpu_data['count'],
                'ip': '127.0.0.1', 'port': API_PORT,
                'uptime': f"{int(uptime_s / 3600)}h {int((uptime_s % 3600) / 60)}m",
                'load': round(cpu_pct, 1),
                'memory_percent': round(mem.percent, 1),
            }
            all_nodes = [local_node] + discovered
            return jsonify({'nodes': all_nodes, 'count': len(all_nodes),
                            'timestamp': time.time()})
        except Exception as e:
            return jsonify({
                'nodes': [{'host': 'localhost', 'status': 'error', 'error': str(e)}],
                'count': 1,
            }), 500



# Backward-compatible module-level app for imports (e.g. conftest fixtures)
# Created via factory — single Flask instance, no duplicate.
app = create_app()


# ============================================================================
# Main (CLI entry point)
# ============================================================================

def main():
    """Main entry point with graceful shutdown."""
    import signal
    import atexit
    import argparse

    parser = argparse.ArgumentParser(description="VRAMancer API Server")
    parser.add_argument("--model", type=str, default=None,
                        help="Pre-load a model at startup")
    parser.add_argument("--backend", type=str, default="auto",
                        help="LLM backend (auto, huggingface, vllm, ollama)")
    parser.add_argument("--gpus", type=int, default=None,
                        help="Number of GPUs to use")
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    args, _ = parser.parse_known_args()

    host = args.host or API_HOST
    port = args.port or API_PORT

    def _cleanup():
        logger.info("Shutting down gracefully...")
        # Drain inference executor
        try:
            if hasattr(app, 'vrm_executor') and app.vrm_executor:
                logger.info("Draining inference executor...")
                app.vrm_executor.shutdown(wait=True, cancel_futures=True)
                logger.info("Executor drained")
        except Exception as e:
            logger.warning("Executor shutdown error: %s", e)
        _registry.shutdown()
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
        logger.info("Cleanup complete")

    atexit.register(_cleanup)

    def _signal_handler(signum, frame):
        signame = signal.Signals(signum).name
        logger.info("Received %s, initiating shutdown...", signame)
        _cleanup()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # Pre-load model if specified via CLI
    if args.model:
        try:
            _registry.load(args.model, backend=args.backend,
                           num_gpus=args.gpus, verbose=True)
            logger.info("Model pre-loaded: %s", args.model)
        except Exception as e:
            logger.error("Failed to pre-load model: %s", e)

    logger.info("=" * 60)
    logger.info("VRAMancer API — Production Mode")
    logger.info("=" * 60)
    logger.info("  Host: %s:%s", host, port)
    logger.info("  POST /v1/completions  — Text generation (OpenAI-compatible)")
    logger.info("  POST /v1/chat/completions — Chat completions")
    logger.info("  POST /api/generate    — Text generation (alias)")
    logger.info("  POST /api/infer       — Raw tensor inference")
    logger.info("  POST /api/models/load — Load a model")
    logger.info("  GET  /api/models      — List loaded models")
    logger.info("  GET  /health          — Health check")
    logger.info("  GET  /api/gpu         — GPU info")
    logger.info("  GET  /api/nodes       — Cluster nodes")
    logger.info("=" * 60)

    # Production: use gunicorn if available, fallback to Werkzeug
    try:
        from gunicorn.app.base import BaseApplication

        class VRAMancerGunicorn(BaseApplication):
            """Gunicorn application wrapper for VRAMancer API."""

            def __init__(self, flask_app, options=None):
                self.options = options or {}
                self.application = flask_app
                super().__init__()

            def load_config(self):
                for key, value in self.options.items():
                    if key in self.cfg.settings and value is not None:
                        self.cfg.set(key.lower(), value)

            def load(self):
                return self.application

        workers = int(os.environ.get('VRM_WORKERS', '1'))
        threads = int(os.environ.get('VRM_THREADS', '4'))
        gunicorn_opts = {
            'bind': f'{host}:{port}',
            'workers': workers,
            'threads': threads,
            'timeout': int(os.environ.get('VRM_GUNICORN_TIMEOUT', '120')),
            'graceful_timeout': 30,
            'keep_alive': 5,
            'accesslog': '-' if API_DEBUG else None,
            'errorlog': '-',
            'loglevel': 'info',
            'preload_app': True,
        }
        logger.info("Starting gunicorn (%d worker(s), %d thread(s))", workers, threads)
        VRAMancerGunicorn(app, gunicorn_opts).run()

    except ImportError:
        logger.warning(
            "gunicorn not installed — falling back to Werkzeug dev server. "
            "Install gunicorn for production: pip install gunicorn"
        )
        try:
            app.run(host=host, port=port, debug=False,
                    use_reloader=False, threaded=True)
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        except Exception as e:
            logger.critical("Fatal error: %s", e, exc_info=True)
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.critical("Fatal error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
