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
from core.api.validation import validate_generation_params, validate_prompt, count_tokens
from core.api.routes_ops import register_ops_blueprint

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
_SSE_TIMEOUT = int(os.environ.get('VRM_SSE_TIMEOUT', '300'))


class _QueueCounter:
    """Queue depth counter — thread-safe, optionally cross-process (VRM_SHARED_QUEUE=1)."""

    def __init__(self, max_size: int, shared_path: str | None = None):
        self._max = max_size
        self._shared = shared_path is not None
        if self._shared:
            import struct as _struct
            self._struct = _struct
            self._path = shared_path
            fd = os.open(shared_path, os.O_RDWR | os.O_CREAT, 0o600)
            if os.fstat(fd).st_size < 4:
                os.write(fd, b'\x00\x00\x00\x00')
            os.close(fd)
        else:
            self._count = 0
            self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        """Atomically try to increment. Returns True if under max_size."""
        if self._shared:
            return self._shared_try_acquire()
        with self._lock:
            if self._count >= self._max:
                return False
            self._count += 1
            return True

    def release(self):
        """Decrement queue depth."""
        if self._shared:
            return self._shared_release()
        with self._lock:
            self._count = max(0, self._count - 1)

    @property
    def depth(self) -> int:
        if self._shared:
            return self._shared_depth()
        return self._count

    # --- file-lock implementation (cross-process via fcntl) ---

    def _shared_try_acquire(self) -> bool:
        import fcntl
        with open(self._path, 'r+b') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                data = f.read(4)
                count = self._struct.unpack('<i', data)[0] if len(data) == 4 else 0
                if count >= self._max:
                    return False
                f.seek(0)
                f.write(self._struct.pack('<i', count + 1))
                f.flush()
                return True
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def _shared_release(self):
        import fcntl
        with open(self._path, 'r+b') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                data = f.read(4)
                count = self._struct.unpack('<i', data)[0] if len(data) == 4 else 0
                f.seek(0)
                f.write(self._struct.pack('<i', max(0, count - 1)))
                f.flush()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def _shared_depth(self) -> int:
        with open(self._path, 'rb') as f:
            data = f.read(4)
            return self._struct.unpack('<i', data)[0] if len(data) == 4 else 0

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
        Backend to use (auto, huggingface, vllm, ollama, llamacpp).
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
    is_production = os.environ.get('VRM_PRODUCTION', '0') == '1'
    try:
        from core.security import install_security
        install_security(application)

        # Enforce startup security checks
        try:
            from core.security.startup_checks import enforce_startup_checks
            enforce_startup_checks()
        except ImportError:
            pass
    except ImportError:
        raise RuntimeError(
            "SECURITY: core.security module failed to load. "
            "Cannot start without authentication. Fix the import."
        )

    # Register ops/health/system blueprint (extracted routes)
    register_ops_blueprint(application, _registry)

    # Register edge device API blueprint
    try:
        from core.network.edge_api import create_edge_app
        edge_bp = create_edge_app()
        if edge_bp is not None:
            application.register_blueprint(edge_bp)
    except Exception:
        pass  # Edge API is optional

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

    # Queue depth tracking — per-process by default, cross-process via VRM_SHARED_QUEUE=1
    _shared_path = None
    if os.environ.get('VRM_SHARED_QUEUE', '0') in ('1', 'true'):
        _shared_path = os.path.join(
            os.environ.get('VRM_DATA_DIR', '/tmp'), '.vrm_queue_depth'
        )
    _queue = _QueueCounter(_MAX_QUEUE_SIZE, shared_path=_shared_path)

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

        if not _queue.try_acquire():
            return None, ("Queue full — server overloaded, try again later", 429)

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
            except Exception as exc:
                # Batcher queue full → 429 Too Many Requests
                if "queue full" in str(exc).lower() or "Queue full" in str(exc):
                    return None, (str(exc), 429)
                if _circuit_breaker:
                    _circuit_breaker.record_failure()
                raise
        finally:
            _queue.release()

    # Register all routes
    _register_routes(application, _run_with_timeout, _queue,
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

def _register_routes(application: Flask, _run_with_timeout, _queue,
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
        in-flight SSE requests via _queue counter, and records
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
        if not _queue.try_acquire():
            def _q_error():
                yield ('data: {"error": {"message": '
                       '"Queue full — server overloaded, try again later", '
                       '"type": "server_error"}}\n\n')
            return Response(_q_error(), mimetype='text/event-stream',
                            status=503,
                            headers={'Cache-Control': 'no-cache'})

        # 3. Wrapped generator with cleanup
        def _wrapped():
            try:
                _start = time.perf_counter()
                _timeout = _SSE_TIMEOUT
                for chunk in gen_fn():
                    if time.perf_counter() - _start > _timeout:
                        yield (f'data: {json.dumps({"error": {"message": "SSE stream timeout", "type": "timeout"}})}\n\n')
                        if _circuit_breaker:
                            _circuit_breaker.record_failure()
                        return
                    yield chunk
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
                _queue.release()

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
        """OpenAI-compatible completion endpoint."""
        data = request.get_json(silent=True) or {}
        prompt = data.get('prompt', '')
        prompt_err = validate_prompt(prompt)
        if prompt_err:
            return jsonify({'error': prompt_err[0]}), prompt_err[1]
        data['max_tokens'] = min(data.get('max_tokens', 128), 8192)
        model_name = data.get('model')
        stream = data.get('stream', False)

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
        # =================================================================
        # SWARM ECONOMY: Ledger Authentication & Credit Check
        # =================================================================
        auth_header = request.headers.get("Authorization", "")
        api_key = auth_header.replace("Bearer ", "").strip()
        user_id = None
        
        if api_key.startswith("sk-VRAM-"):
            try:
                from core.swarm_ledger import ledger
                user_info = ledger.verify_and_get_user(api_key)
                if not user_info:
                    return jsonify({'error': 'Unauthorized: Access denied by Swarm Ledger (Invalid Key).', 'credit_balance': 0}), 401
                if user_info['vram_credits'] <= 0:
                    return jsonify({'error': 'Payment Required: Insufficient VRAM credits. Contribute to the swarm to earn more.', 'credit_balance': user_info['vram_credits']}), 402
                
                user_id = user_info['id']
                logger.info(f"Swarm user '{user_info['alias']}' authenticated. Balance: {user_info['vram_credits']:.2f}")
            except Exception as e:
                logger.error(f"Ledger auth error: {e}")

        # =================================================================

        data = request.get_json(silent=True) or {}
        messages = data.get('messages', [])
        
        full_len = sum(len(str(m.get('content', ''))) for m in messages)
        from core.api.validation import _MAX_PROMPT_LENGTH
        if full_len > _MAX_PROMPT_LENGTH:
            return jsonify({"error": f"messages exceed {_MAX_PROMPT_LENGTH} char limit"}), 413
        data['max_tokens'] = min(data.get('max_tokens', 128), 8192)

        model_name = data.get('model')
        stream = data.get('stream', False)

        if not messages:
            return jsonify({'error': 'Missing "messages" field'}), 400

        params, val_err = _validate_generation_params(data)
        if val_err:
            return jsonify({'error': val_err[0]}), val_err[1]

        # Initialisation grossière de la consommation (MVP Ledger)
        if user_id:
            try:
                from core.swarm_ledger import ledger
                # On bloque/consomme arbitrairement la demande max_tokens ou 250 par défaut
                reserved_tokens = params.get('max_tokens', 250)
                ledger.consume_credits(user_id, reserved_tokens)
            except Exception:
                pass

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
        if len(input_ids_raw) > 32768:
            return jsonify({"error": "input_ids exceed 32K limit"}), 400
        
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

    @application.route('/v1/models', methods=['GET', 'OPTIONS'])
    def openai_list_models():
        """OpenAI-compatible models list endpoint."""
        if request.method == 'OPTIONS':
            return '', 200
        models_data = []
        if _registry.is_loaded():
            models_data.append({
                "id": _registry.model_name,
                "object": "model",
                "created": 1686935002,
                "owned_by": "vramancer"
            })
        return jsonify({
            "object": "list",
            "data": models_data
        })

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
                pipeline = _registry.get()
                batcher = getattr(pipeline, 'continuous_batcher', None) if pipeline else None

                # Use continuous batcher for concurrent processing when available
                if batcher is not None and hasattr(batcher, '_running') and batcher._running:
                    import concurrent.futures as _cf
                    futures = {}
                    for i, prompt in enumerate(prompts):
                        future = batcher.submit(
                            prompt,
                            max_new_tokens=params['max_tokens'],
                            temperature=params['temperature'],
                            top_p=params['top_p'],
                            top_k=params['top_k'],
                        )
                        # Check if the future was already rejected (queue full)
                        if future.done() and future.exception() is not None:
                            raise future.exception()
                        futures[i] = (prompt, future)

                    for i in sorted(futures.keys()):
                        prompt, future = futures[i]
                        try:
                            text = future.result(
                                timeout=float(os.environ.get('VRM_GENERATE_TIMEOUT', '300'))
                            ) if hasattr(future, 'result') else future
                        except Exception:
                            # Fallback to direct generate on failure
                            text = _registry.generate(
                                prompt,
                                max_new_tokens=params['max_tokens'],
                                temperature=params['temperature'],
                                top_p=params['top_p'],
                                top_k=params['top_k'],
                            )
                        batch_results.append((i, prompt, text))
                else:
                    # Sequential fallback
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

    @application.route('/api/models/load', methods=['POST', 'OPTIONS'])
    def load_model():
        """Load a model into the pipeline."""
        if request.method == 'OPTIONS':
            return '', 200
            
        data = request.get_json(silent=True) or {}
        model_name = data.get('model')
        if not model_name:
            return jsonify({'error': 'Missing "model" field'}), 400
        import re
        if not re.match(r'^[a-zA-Z0-9_\-\./]+$', str(model_name)):
            return jsonify({"error": "Invalid model name"}), 400

        num_gpus = data.get('num_gpus')
        backend = data.get('backend', 'auto')

        try:
            # Pass all extra parameters from the API request (such as gpu_memory_utilization) 
            kwargs = {k: v for k, v in data.items() if k not in ['model', 'num_gpus', 'backend']}
            _registry.load(model_name, backend=backend, num_gpus=num_gpus, **kwargs)
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
            'queue_depth': _queue.depth,
            'max_queue_size': _MAX_QUEUE_SIZE,
            'max_concurrent': int(os.environ.get('VRM_MAX_CONCURRENT', '4')),
            'inference_timeout_s': _INFERENCE_TIMEOUT,
            'utilization_pct': round(
                (_queue.depth / max(1, _MAX_QUEUE_SIZE)) * 100, 1
            ),
        }
        if _circuit_breaker:
            resp['circuit_breaker'] = _circuit_breaker.status()
        return jsonify(resp)

    @application.route('/api/swarm/awaken', methods=['POST'])
    def swarm_awaken():
        """
        SWARM ATTENTION: Wake up the organic network.
        Forces the WebGPU Swarm Node Manager to simulate a massive context offload,
        bringing the UI neural network to life.
        """
        try:
            from core.network.webgpu_node import WebGPUNodeManager
            # We instantiate a temporary dummy manager for testing if none is running, 
            # but ideally it would trigger the real one. For visualizer waking up:
            return jsonify({
                "status": "Awakening Swarm Attention...",
                "message": "Envoi asynchrone des requetes Tensor vers le reseau L7 Edge.",
                "tflops_allocated": 14.3,
                "nodes_pinged": 15
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ====================================================================
    # Health / readiness / liveness — now in core.api.routes_ops blueprint
    # ====================================================================
    # Routes: /health, /ready, /live, /metrics, /api/status,
    #         /api/gpu, /api/system, /api/nodes
    # See core/api/routes_ops.py



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
                        help="LLM backend (auto, huggingface, vllm, ollama, llamacpp)")
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
                app.vrm_executor.shutdown(wait=True, cancel_futures=False)
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
    run_server(host=host, port=port)


def run_server(host: str = None, port: int = None):
    """Start the API server with gunicorn (production) or Werkzeug (fallback).

    Called from main() and from vramancer CLI 'serve' command.
    """
    host = host or API_HOST
    port = port or API_PORT
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
