"""Client CLI pour décoder un flux binaire de télémétrie.

Usage:
    python -m cli.telemetry_cli --url http://localhost:5010/api/telemetry.bin
"""
from __future__ import annotations
import argparse, requests, sys
from core.telemetry import decode_stream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://localhost:5010/api/telemetry.bin')
    args = ap.parse_args()
    try:
        r = requests.get(args.url, timeout=3)
        r.raise_for_status()
    except Exception as e:
        print('Erreur requête:', e, file=sys.stderr)
        sys.exit(1)
    blob = r.content
    for entry in decode_stream(blob):
        print(f"{entry['id']}: load={entry['cpu_load_pct']:.2f}% free={entry['free_cores']} vram={entry['vram_used_mb']}/{entry['vram_total_mb']}MB")

if __name__ == '__main__':
    main()
