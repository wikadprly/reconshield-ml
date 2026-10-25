"""Publish the latest metrics.json into PostgreSQL.

After training, the web dashboard can read metrics from the database instead of
hitting this service (`security-service.ts` tries database -> API -> mock).

    python -m training.sync_metrics_db
    python -m training.sync_metrics_db --prune   # delete older evaluations too

Re-running for the same modelName+version updates the existing row in place.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from typing import Any

import psycopg2

import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync metrics.json into PostgreSQL")
    parser.add_argument(
        "--metrics",
        default=None,
        help="Path to metrics.json (default: data/processed/metrics.json)",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Delete evaluations other than the one being synced",
    )
    return parser.parse_args()


def validate(metrics: dict[str, Any]) -> None:
    required = (
        "modelName",
        "version",
        "accuracy",
        "precision",
        "recall",
        "f1Score",
        "confusionMatrix",
        "featureImportance",
        "evaluatedAt",
    )
    missing = [key for key in required if key not in metrics]
    if missing:
        raise SystemExit(f"metrics.json is missing required fields: {missing}")

    matrix = metrics["confusionMatrix"]
    matrix_keys = ("trueNormal", "falseThreat", "missedThreat", "trueThreat")
    missing_matrix = [key for key in matrix_keys if key not in matrix]
    if missing_matrix:
        raise SystemExit(f"confusionMatrix is missing keys: {missing_matrix}")


def main() -> int:
    args = parse_args()

    if not config.DATABASE_URL:
        print("[sync] DATABASE_URL is not set — add it to .env", file=sys.stderr)
        return 1

    metrics_path = args.metrics or config.METRICS_PATH
    if not metrics_path.exists():
        print(f"[sync] {metrics_path} not found — train a model first", file=sys.stderr)
        return 1

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    validate(metrics)

    matrix = metrics["confusionMatrix"]
    conn = psycopg2.connect(config.DATABASE_URL)
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id FROM model_evaluations
                WHERE "modelName" = %s AND version = %s
                ORDER BY "evaluatedAt" DESC
                LIMIT 1
                """,
                (metrics["modelName"], metrics["version"]),
            )
            row = cur.fetchone()
            evaluation_id = row[0] if row else f"eval-{uuid.uuid4().hex[:20]}"

            values = (
                metrics["modelName"],
                metrics["version"],
                float(metrics["accuracy"]),
                float(metrics["precision"]),
                float(metrics["recall"]),
                float(metrics["f1Score"]),
                int(matrix["trueNormal"]),
                int(matrix["falseThreat"]),
                int(matrix["missedThreat"]),
                int(matrix["trueThreat"]),
                metrics.get("dataset"),
                int(metrics["samples"]) if metrics.get("samples") is not None else None,
                metrics["evaluatedAt"],
            )

            if row:
                cur.execute(
                    """
                    UPDATE model_evaluations SET
                        accuracy = %s, precision = %s, recall = %s, "f1Score" = %s,
                        "trueNormal" = %s, "falseThreat" = %s, "missedThreat" = %s,
                        "trueThreat" = %s, dataset = %s, samples = %s, "evaluatedAt" = %s
                    WHERE id = %s
                    """,
                    values[2:] + (evaluation_id,),
                )
                print(f"[sync] updated evaluation {evaluation_id}")
            else:
                cur.execute(
                    """
                    INSERT INTO model_evaluations
                        (id, "modelName", version, accuracy, precision, recall, "f1Score",
                         "trueNormal", "falseThreat", "missedThreat", "trueThreat",
                         dataset, samples, "evaluatedAt")
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (evaluation_id,) + values,
                )
                print(f"[sync] inserted evaluation {evaluation_id}")

            # Replace feature importances wholesale so stale rows cannot linger.
            cur.execute(
                'DELETE FROM feature_importances WHERE "evaluationId" = %s',
                (evaluation_id,),
            )
            for item in metrics["featureImportance"]:
                cur.execute(
                    "INSERT INTO feature_importances (id, feature, importance, \"evaluationId\") VALUES (%s, %s, %s, %s)",
                    (f"fi-{uuid.uuid4().hex[:20]}", item["feature"], float(item["importance"]), evaluation_id),
                )

            if args.prune:
                cur.execute("DELETE FROM model_evaluations WHERE id <> %s", (evaluation_id,))
                print("[sync] pruned all other evaluations")

            count = len(metrics["featureImportance"])
            print(f"[sync] wrote {count} feature importance rows for {metrics['version']}")

        conn.commit()
    except Exception as exc:
        conn.rollback()
        print(f"[sync] failed, rolled back: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()

    print(f"[sync] done — reconshield-web now reads these metrics with source=database")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
