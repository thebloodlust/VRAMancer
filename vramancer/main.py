#!/usr/bin/env python3
"""VRAMancer - Point d'entree principal unifie.
"""

import argparse
import os
import sys
from pathlib import Path

# S'assurer que le repertoire racine est dans le path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main(argv=None):
    """Point d'entree CLI principal.

    Parameters
    ----------
    argv : list[str] | None
        Arguments CLI. Si None, utilise sys.argv[1:].
    """
    parser = argparse.ArgumentParser(
        prog="vramancer",
        description="VRAMancer - Multi-GPU LLM Inference Orchestrator",
    )
    sub = parser.add_subparsers(dest="command", help="Commande")

    # ---- serve ----
    p_serve = sub.add_parser("serve", help="Lancer le serveur API REST")
    p_serve.add_argument("--model", type=str, default=None,
                         help="Modele LLM a pre-charger (ex: gpt2)")
    p_serve.add_argument("--backend", type=str, default="auto",
                         help="Backend : auto, huggingface, vllm, ollama")
    p_serve.add_argument("--gpus", type=int, default=None,
                         help="Nombre de GPUs (auto si non specifie)")
    p_serve.add_argument("--host", type=str, default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=5030)

    # ---- generate ----
    p_gen = sub.add_parser("generate", help="Generer du texte (one-shot)")
    p_gen.add_argument("--model", type=str, required=True, help="Modele LLM")
    p_gen.add_argument("--prompt", type=str, required=True, help="Texte d'entree")
    p_gen.add_argument("--max-tokens", type=int, default=128)
    p_gen.add_argument("--temperature", type=float, default=1.0)
    p_gen.add_argument("--backend", type=str, default="auto")
    p_gen.add_argument("--gpus", type=int, default=None)

    # ---- status ----
    sub.add_parser("status", help="Afficher l'etat du systeme")

    # ---- benchmark ----
    p_bench = sub.add_parser("benchmark", help="Benchmark GPU")
    p_bench.add_argument("--size", type=int, default=4096, help="Taille matrice (NxN)")

    # ---- discover ----
    p_disc = sub.add_parser("discover", help="Decouverte reseau (noeuds)")
    p_disc.add_argument("--timeout", type=int, default=5, help="Duree d'ecoute (secondes)")

    # ---- split ----
    p_split = sub.add_parser("split", help="Profiler et splitter un modele")
    p_split.add_argument("model", help="Nom du modele HuggingFace")
    p_split.add_argument("--gpus", type=int, default=2, help="Nombre de GPUs")
    p_split.add_argument("--profile", action="store_true", default=True,
                         help="Utiliser le profiler (defaut)")
    p_split.add_argument("--no-profile", dest="profile", action="store_false",
                         help="Split VRAM-proportionnel simple")

    # ---- hub ----
    p_hub = sub.add_parser("hub", help="Explorer le catalogue de modeles HuggingFace")
    p_hub.add_argument("model", help="Identifiant du modele sur HF (ex: HuggingFaceH4/zephyr-7b-beta)")


    # ---- health ----
    sub.add_parser("health", help="Verifier la sante du systeme")

    # ---- auth ----
    p_auth = sub.add_parser("auth", help="Générer une identité Swarm Ledger (sk-VRAM-...)")

    # ---- version ----
    sub.add_parser("version", help="Afficher la version")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return

    if args.command == "version":
        _cmd_version()
    elif args.command == "auth":
        try:
            from vramancer.cli.swarm_cli import ui_auth_generate
            ui_auth_generate()
        except ImportError as e:
            print(f"Erreur UI (rich manquant?): {e}")
    elif args.command == "status":
        _cmd_status()
    elif args.command == "serve":
        _cmd_serve(args)
    elif args.command == "generate":
        _cmd_generate(args)
    elif args.command == "benchmark":
        _cmd_benchmark(args)
    elif args.command == "discover":
        _cmd_discover(args)
    elif args.command == "hub":
        _cmd_hub(args)
    elif args.command == "split":
        _cmd_split(args)
    elif args.command == "health":
        _cmd_health()
    else:
        parser.print_help()


# ====================================================================
# Commandes
# ====================================================================

def _cmd_version():
    try:
        from core import __version__
        print(f"VRAMancer v{__version__}")
    except Exception:
        print("VRAMancer v0.2.4")


def _cmd_status():
    """Affiche l'etat du systeme (GPUs, memoire, backend)."""
    print("=" * 50)
    print("VRAMancer - Etat du systeme")
    print("=" * 50)
    try:
        from core.utils import detect_backend, enumerate_devices
        backend = detect_backend()
        devices = enumerate_devices()
        print(f"Backend: {backend}")
        print(f"Devices: {len(devices)}")
        for d in devices:
            print(f"  [{d.get('index', '?')}] {d.get('name', '?')} "
                  f"({d.get('backend', '?')}) "
                  f"{d.get('memory_total_mb', '?')} MB")
    except Exception as e:
        print(f"Backend: unavailable ({e})")
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"\nRAM: {mem.used / (1024**3):.1f} / {mem.total / (1024**3):.1f} GB "
              f"({mem.percent}%)")
    except ImportError:
        pass
    print("=" * 50)


def _cmd_serve(args):
    """Lancer le serveur API avec pipeline complet."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        console = Console()
    except ImportError:
        console = None

    from core.production_api import app

    os.environ.setdefault('VRM_API_HOST', args.host)
    os.environ.setdefault('VRM_API_PORT', str(args.port))

    if console:
        table = Table(show_header=False, box=None)
        table.add_column("Param", style="cyan")
        table.add_column("Value", style="yellow")
        table.add_row("Hôte/Port", f"{args.host}:{args.port}")
        if args.model:
            table.add_row("Modèle", args.model)
        table.add_row("Backend", args.backend)
        table.add_row("GPUs", str(args.gpus or 'auto'))

        endpoints = (
            "[bold]Endpoints:[/bold]\n"
            "  POST [green]/v1/completions[/green]  - Text generation (OpenAI-compatible)\n"
            "  POST [green]/api/models/load[/green] - Load a model dynamic\n"
            "  GET  [cyan]/health[/cyan]          - Health check\n"
            "  GET  [cyan]/api/nodes[/cyan]       - Cluster nodes\n"
        )
        
        console.print(Panel.fit(table, title="[bold blue]VRAMancer API Server[/bold blue]", subtitle="🚀 En cours de démarrage..."))
        console.print(endpoints)
    else:
        print("=" * 60)
        print("VRAMancer API Server")
        print("=" * 60)
        print(f"  Host: {args.host}:{args.port}")
        if args.model:
            print(f"  Model: {args.model}")
        print(f"  Backend: {args.backend}")
        print(f"  GPUs: {args.gpus or 'auto'}")
        print()
        print("  POST /v1/completions  - Text generation (OpenAI-compatible)")
        print("  POST /api/generate    - Text generation (alias)")
        print("  POST /api/infer       - Raw tensor inference")
        print("  POST /api/models/load - Load a model")
        print("  GET  /health          - Health check")
        print("  GET  /api/gpu         - GPU info")
        print("  GET  /api/nodes       - Cluster nodes")
        print("=" * 60)

    # Pre-load model if specified
    if args.model:
        try:
            from core.inference_pipeline import InferencePipeline
            import core.production_api as api_mod
            pipeline = InferencePipeline(
                backend_name=args.backend,
                verbose=True,
                enable_metrics=True,
            )
            
            gpus_to_use = args.gpus
            if gpus_to_use is None:
                try:
                    import torch
                    gpus_to_use = torch.cuda.device_count()
                except ImportError:
                    gpus_to_use = 1
                    
            load_kwargs = {}
            if args.backend == "vllm":
                if gpus_to_use > 1:
                    load_kwargs["tensor_parallel_size"] = gpus_to_use
                load_kwargs["max_model_len"] = 8192
                load_kwargs["gpu_memory_utilization"] = 0.65  # Lower utilization to avoid OOM on smaller heterogeneous GPUs
                
            pipeline.load(args.model, num_gpus=gpus_to_use, **load_kwargs)
            api_mod._registry._pipeline = pipeline
            print(f"\n  Model loaded: {args.model}")
            print(f"  Blocks: {len(pipeline.blocks)}")
            print(f"  GPUs: {pipeline.num_gpus}")
            print()
        except Exception as e:
            print(f"\n  WARNING: Model pre-load failed: {e}")
            print("  Send POST /api/models/load to load later.\n")

    try:
        app.run(host=args.host, port=args.port, debug=False,
                use_reloader=False, threaded=True)
    except KeyboardInterrupt:
        print("\nShutdown requested.")


def _cmd_generate(args):
    """Executer une generation de texte one-shot via le pipeline."""
    print(f"Loading model: {args.model}...")
    try:
        from core.inference_pipeline import InferencePipeline
        pipeline = InferencePipeline(
            backend_name=args.backend,
            verbose=False,
            enable_metrics=False,
        )
        
        # Auto-detect GPUs if not specified
        gpus_to_use = args.gpus
        if gpus_to_use is None:
            try:
                import torch
                gpus_to_use = torch.cuda.device_count()
            except ImportError:
                gpus_to_use = 1

        # Passer parameters specifiques au backend
        load_kwargs = {}
        if args.backend == "vllm":
            if gpus_to_use > 1:
                load_kwargs["tensor_parallel_size"] = gpus_to_use
            # Prevent OOM by reducing max_model_len if it's very large
            load_kwargs["max_model_len"] = 8192
            load_kwargs["gpu_memory_utilization"] = 0.65
        
        pipeline.load(args.model, num_gpus=gpus_to_use, **load_kwargs)
        print(f"Model loaded ({pipeline.num_gpus} GPU(s), "
              f"{len(pipeline.blocks)} blocks)")
        print()
        result = pipeline.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(result)
        pipeline.shutdown()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def _cmd_benchmark(args):
    """Benchmark GPU avec matmul."""
    print("GPU Benchmark")
    print("-" * 40)
    try:
        from core.layer_profiler import LayerProfiler
        profiler = LayerProfiler()
        results = profiler.benchmark_gpu(matrix_size=args.size)
        for key, val in results.items():
            if isinstance(val, float):
                print(f"  {key}: {val:.2f}")
            else:
                print(f"  {key}: {val}")
    except Exception as e:
        print(f"Benchmark failed: {e}")
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    print(f"  GPU {i}: {props.name} "
                          f"({props.total_memory / (1024**3):.1f} GB)")
            else:
                print("  No CUDA GPU available")
        except ImportError:
            print("  PyTorch not installed")


def _cmd_discover(args):
    """Decouverte des noeuds reseau."""
    print("Network Node Discovery")
    print("-" * 40)
    try:
        from core.network.cluster_discovery import ClusterDiscovery
        import time
        disc = ClusterDiscovery(heartbeat_interval=2)
        disc.start()
        print(f"Listening for {args.timeout}s...")
        time.sleep(args.timeout)
        nodes = disc.get_nodes()
        disc.stop()
        if nodes:
            for hostname, info in nodes.items():
                print(f"  [{hostname}] {info}")
        else:
            print("  No nodes discovered (try a longer timeout or check network)")
    except Exception as e:
        print(f"Discovery failed: {e}")


def _cmd_hub(args):
    """Explorer et afficher les quantizations HuggingFace d'un modele."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        console = Console()
    except ImportError:
        console = None

    if console:
        console.print(f"[bold blue]VRAMancer Hub[/bold blue] - Recherche de [yellow]{args.model}[/yellow]...")
    else:
        print(f"VRAMancer Hub - Recherche de {args.model}...")

    from core.model_hub import search_huggingface_model
    info = search_huggingface_model(args.model)

    if console:
        if "error" in info:
            console.print(f"[bold red]Erreur:[/bold red] {info['error']}")
            return
            
        table = Table(title=f"Informations - {info['id']}", show_header=False)
        table.add_column("Propriété", style="cyan")
        table.add_column("Valeur", style="green")
        
        table.add_row("ID du modèle", info.get("id"))
        table.add_row("Task (Pipeline)", info.get("pipeline_tag", "N/A"))
        table.add_row("Téléchargements", str(info.get("downloads", 0)))
        
        console.print(table)
        
        formats_str = "\n".join(f"• {fmt}" for fmt in info.get("formats", []))
        console.print(Panel(formats_str, title="Précisions et Formats détectés (ex: NVFP4, AWQ, FP16)", expand=False))
        
    else:
        if "error" in info:
            print(f"Erreur: {info['error']}")
            return
            
        print(f"\nModele : {info.get('id')}")
        print(f"Task   : {info.get('pipeline_tag')}")
        print(f"DLs    : {info.get('downloads')}")
        print("\nFormats disponibles :")
        for fmt in info.get("formats", []):
            print(f" - {fmt}")


def _cmd_split(args):
    """Profiler et splitter un modele sur plusieurs GPUs."""
    print(f"VRAMancer Model Splitter")
    print(f"  Model: {args.model}")
    print(f"  GPUs: {args.gpus}")
    print(f"  Strategy: {'profiler' if args.profile else 'vram-proportional'}")
    print()
    try:
        from core.model_splitter import split_model_into_blocks
        blocks = split_model_into_blocks(
            model_name=args.model,
            num_gpus=args.gpus,
            use_profiler=args.profile,
        )
        print(f"Split into {len(blocks)} blocks:")
        for i, block in enumerate(blocks):
            n_layers = len(list(block.children())) if hasattr(block, "children") else "?"
            params = sum(p.numel() for p in block.parameters()) if hasattr(block, "parameters") else 0
            print(f"  Block {i}: {n_layers} layers, {params / 1e6:.1f}M parameters")
    except ImportError as e:
        print(f"Import error: {e}")
        print("  Install: pip install torch transformers")
        sys.exit(1)
    except Exception as e:
        print(f"Split error: {e}")
        sys.exit(1)


def _cmd_health():
    """Verifier la sante du systeme."""
    print("VRAMancer Health Check")
    print("=" * 50)
    checks = {}
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    checks["python"] = {"status": "ok", "version": py_ver}

    for module_name in ["core.config", "core.monitor", "core.scheduler",
                        "core.block_router", "core.backends", "core.production_api"]:
        try:
            __import__(module_name)
            checks[module_name] = {"status": "ok"}
        except Exception as e:
            checks[module_name] = {"status": "error", "error": str(e)}

    try:
        import torch
        checks["torch"] = {"status": "ok", "version": torch.__version__,
                           "cuda": torch.cuda.is_available()}
    except ImportError:
        checks["torch"] = {"status": "missing"}

    all_ok = True
    for name, result in checks.items():
        status = result["status"]
        icon = "OK" if status == "ok" else "FAIL"
        extra = ""
        if "version" in result:
            extra = f" v{result['version']}"
        if "cuda" in result:
            extra += f" (CUDA: {result['cuda']})"
        if "error" in result:
            extra = f" — {result['error']}"
        print(f"  [{icon:>4}] {name}{extra}")
        if status != "ok":
            all_ok = False

    print()
    if all_ok:
        print("All checks passed!")
    else:
        print("Some checks failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()

