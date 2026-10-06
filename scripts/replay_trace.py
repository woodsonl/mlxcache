#!/usr/bin/env python3
"""Replay a daemon trace (MLXCACHE_TRACE JSONL) and score reuse honestly.

Step 5 harness (success criteria 3-4). A trace record embeds the EXACT
request payload (messages array) plus the verdict the request got at capture
time. This harness:

  1. Scores the captured run itself (ground truth: capture-time verdicts,
     ALL traffic included — several requests means several records, warming
     included, because every served request wrote a record).
  2. Optionally re-sends each captured request to a live daemon and reports
     what the daemon did on replay (--twice for a second, warm pass).

The headline number is the captured hit_rate: prefill_from.sum / n_tokens.sum
— identical in definition to the daemon's /stats hit_rate, so a capture and
its /stats line must always agree.

Usage:
  uv run python scripts/replay_trace.py --trace /tmp/run.jsonl --summary-only
  uv run python scripts/replay_trace.py --trace /tmp/run.jsonl \
      --daemon http://127.0.0.1:8420 --twice [--report out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request


def read_trace(path: str) -> list[dict]:
    records = []
    with open(path, "rb") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: bad JSON line: {exc}") from exc
    if not records:
        raise SystemExit(f"{path}: no records")
    missing = [r for r in records if "messages" not in r]
    if missing:
        raise SystemExit(
            f"{path}: {len(missing)} record(s) have no embedded payload — "
            "captured by an older daemon; re-capture with MLXCACHE_TRACE on a "
            "current build (payload embedding is required for faithful replay)"
        )
    return records


def score_records(records: list[dict], label: str) -> dict:
    total = sum(int(r["n_tokens"]) for r in records)
    reused = sum(int(r["prefill_from"]) for r in records)
    verdicts: dict[str, int] = {}
    for r in records:
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
    return {
        "label": label,
        "requests": len(records),
        "tokens_total": total,
        "tokens_reused": reused,
        "hit_rate": round(reused / total, 4) if total else 0.0,
        "verdicts": verdicts,
    }


def replay(records: list[dict], base: str) -> dict:
    """Re-send every captured request verbatim; report the daemon's own tally."""
    url = f"{base.rstrip('/')}/v1/chat/completions"
    walls: list[float] = []
    for i, rec in enumerate(records):
        body = json.dumps(
            {
                "model": rec.get("model") or rec["model_hash"] and str(rec["model_hash"]),
                "messages": rec["messages"],
                "stream": False,
            }
        ).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=900) as resp:
                payload = json.loads(resp.read())
        except Exception as exc:  # noqa: BLE001 — a failed replay leg is the finding
            return {"error": f"replay request {i + 1}/{len(records)} failed: {exc}"}
        walls.append(round((time.perf_counter() - t0) * 1000, 1))
        verdict = payload.get("mlxcache", {}).get("verdict")
        expected = rec["verdict"]
        if verdict != expected and expected == "hit":
            # A replayed hit that comes back miss/partial is a fidelity
            # failure worth flagging loudly (deterministic tokenizers must
            # reproduce the capture's routing).
            print(
                f"warning: request {i + 1} captured {expected}, replayed {verdict}",
                file=sys.stderr,
            )
    time.sleep(0.2)
    stats = json.loads(urllib.request.urlopen(f"{base.rstrip('/')}/stats", timeout=60).read())
    return {
        "requests_sent": len(records),
        "wall_ms": walls,
        "daemon_stats": {
            "requests": stats["requests"],
            "hits": stats["hits"],
            "misses": stats["misses"],
            "partials": stats["partials"],
            "tokens_total": stats["tokens_total"],
            "tokens_cached": stats["tokens_cached"],
            "hit_rate": stats["hit_rate"],
            "checkpoints_published": stats.get("checkpoints_published", 0),
            "evictions": stats.get("evictions", 0),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", required=True, help="MLXCACHE_TRACE JSONL file")
    ap.add_argument("--daemon", help="daemon base URL to replay against")
    ap.add_argument(
        "--twice", action="store_true", help="replay two passes (cold then warm behavior)"
    )
    ap.add_argument(
        "--summary-only", action="store_true", help="skip replay; score the capture only"
    )
    ap.add_argument("--report", help="write the JSON result to this path too")
    args = ap.parse_args()

    records = read_trace(args.trace)
    result: dict = {
        "trace": args.trace,
        "record_count": len(records),
        "captured_run": score_records(records, "captured"),
    }
    print(json.dumps(result["captured_run"], indent=2))

    if not args.summary_only:
        if not args.daemon:
            raise SystemExit("--daemon is required unless --summary-only")
        base = args.daemon.rstrip("/")
        print(f"\nreplaying {len(records)} requests against {base} (pass 1)...")
        result["replay_pass1"] = replay(records, base)
        print(json.dumps(result["replay_pass1"], indent=2))
        if args.twice:
            print("\nreplaying again (pass 2: everything persisted is now reused)...")
            result["replay_pass2"] = replay(records, base)
            print(json.dumps(result["replay_pass2"], indent=2))

    if args.report:
        with open(args.report, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nreport written to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
