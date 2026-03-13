import argparse
import sys
import os

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

def ui_auth_generate():
    """Génère et affiche une clé dans le terminal proprement"""
    try:
        from core.swarm_ledger import ledger
        console = Console()
        console.print(Panel("[cyan]VRAMancer Swarm - Génération d'Identité P2P[/cyan]"))
        alias = Prompt.ask("Entrez votre pseudo/alias de nœud ")
        user_id, api_key = ledger.create_user(alias)
        
        console.print(f"\n[green]Identité créée pour {alias} ![/green]")
        console.print(f"Votre ID public : [yellow]{user_id}[/yellow]")
        console.print(f"Votre Clé Secrète : [red bold]{api_key}[/red bold]")
        console.print("\n[bold]⚠️ GARDEZ CETTE CLÉ UNIQUE PRÉCIEUSEMENT. ELLE NE SERA PLUS AFFICHÉE.[/bold]")
        console.print("Utilisez-la comme --token lors du lancement de votre node.")
    except Exception as e:
        print(f"Erreur d'initialisation du ledger: {e}")
