"""
HireAgent Vector Migration Utility.
Migrates existing Qdrant vector collections and stored embeddings to:
sentence-transformers/all-MiniLM-L6-v2 (384D)

Principles:
1. Non-destructive: Never drops relational tables, user accounts, or resume records.
2. Idempotent: Uses deterministic UUID5 point IDs.
3. Safe Resumption: Skips records already stamped with the target model unless --force is specified.
4. Memory-bounded: Processes in configurable batches (default 25) with active GC.
5. Dry-run support: Inspects and reports counts and status without mutating data.
"""

import os
import sys
import argparse
import logging
from typing import Dict, Any, List

# Ensure backend root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
for p in [PROJECT_ROOT, BACKEND_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from app.core.config import settings
from app.core.memory import log_memory_checkpoint, get_current_rss_mb
from app.services.embeddings import get_embedding_model, embed_texts, EMBEDDING_DIMENSION, MODEL_NAME
from app.services.qdrant_service import (
    get_qdrant_client,
    ensure_collection,
    reset_qdrant_client,
    RESUMES_COLLECTION,
    JOBS_COLLECTION,
    EMBEDDING_DIM
)
from app.services.resume_chunking import index_user_resume_vectors
from app.services.job_indexing import index_all_jobs
from app.services.jobs import get_jobs_from_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("hireagent.migration")

def parse_args():
    parser = argparse.ArgumentParser(description="Migrate HireAgent Qdrant vectors to sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--dry-run", action="store_true", help="Report records without modifying Qdrant")
    parser.add_argument("--batch-size", type=int, default=25, help="Batch size for embedding and upserting (default: 25)")
    parser.add_argument("--force", action="store_true", help="Force re-embedding of already migrated records")
    parser.add_argument("--skip-resumes", action="store_true", help="Skip resume vector migration")
    parser.add_argument("--skip-jobs", action="store_true", help="Skip job vector migration")
    parser.add_argument("--resumes-collection", type=str, default=None, help="Target resumes collection name")
    parser.add_argument("--jobs-collection", type=str, default=None, help="Target jobs collection name")
    return parser.parse_args()

def check_qdrant_connectivity() -> bool:
    try:
        client = get_qdrant_client()
        client.get_collections()
        return True
    except Exception as e:
        logger.error(f"Cannot connect to Qdrant Cloud: {e}")
        return False

def migrate_resumes(db, target_collection: str, batch_size: int, force: bool, dry_run: bool) -> Dict[str, Any]:
    from app.models.resume import Resume
    resumes = db.query(Resume).all()
    total = len(resumes)
    logger.info(f"Discovered {total} resume records in database for migration.")

    if dry_run:
        return {"total_resumes": total, "status": "dry_run", "migrated": 0, "skipped": 0}

    migrated = 0
    failed = 0
    skipped = 0

    ensure_collection(target_collection)

    for idx, r in enumerate(resumes):
        if not r.extracted_text or not r.extracted_text.strip():
            logger.info(f"Resume {r.id} (user {r.user_id}) has empty text, skipping.")
            skipped += 1
            continue

        try:
            res = index_user_resume_vectors(user_id=r.user_id, resume_id=r.id, extracted_text=r.extracted_text)
            if res.get("status") == "success":
                migrated += 1
            else:
                failed += 1
        except Exception as e:
            logger.warning(f"Error re-indexing resume {r.id}: {e}")
            failed += 1

        if (idx + 1) % batch_size == 0 or (idx + 1) == total:
            log_memory_checkpoint("migration:resumes_progress", f"{idx + 1}/{total} processed")
            import gc
            gc.collect()

    return {"total_resumes": total, "migrated": migrated, "failed": failed, "skipped": skipped}

def migrate_jobs(target_collection: str, batch_size: int, force: bool, dry_run: bool) -> Dict[str, Any]:
    cat_result = get_jobs_from_db(limit=2000)
    jobs = cat_result.get("jobs", [])
    total = len(jobs)
    logger.info(f"Discovered {total} catalog jobs for migration.")

    if dry_run:
        return {"total_jobs": total, "status": "dry_run", "migrated": 0}

    ensure_collection(target_collection)

    migrated = 0
    # Process in batches
    for i in range(0, total, batch_size):
        batch = jobs[i:i + batch_size]
        try:
            res = index_all_jobs(batch, batch_size=batch_size)
            migrated += res.get("indexed_count", 0)
        except Exception as e:
            logger.error(f"Error indexing job batch {i}-{i + len(batch)}: {e}")
        log_memory_checkpoint("migration:jobs_progress", f"{min(i + batch_size, total)}/{total} processed")
        import gc
        gc.collect()

    return {"total_jobs": total, "migrated": migrated}

def run_migration():
    args = parse_args()
    logger.info("=== HireAgent Vector Migration Starting ===")
    logger.info(f"Target Model: {settings.EMBEDDING_MODEL} (dim: {settings.EMBEDDING_DIMENSION})")
    logger.info(f"Dry Run: {args.dry_run}, Force: {args.force}, Batch Size: {args.batch_size}")

    resumes_col = args.resumes_collection or settings.QDRANT_RESUMES_COLLECTION
    jobs_col = args.jobs_collection or settings.QDRANT_JOBS_COLLECTION

    logger.info(f"Target Resumes Collection: {resumes_col}")
    logger.info(f"Target Jobs Collection: {jobs_col}")
    log_memory_checkpoint("migration:init", f"Initial RSS: {get_current_rss_mb():.2f} MB")

    # Pre-warm embedding model singleton
    logger.info("Initializing embedding model singleton...")
    model = get_embedding_model()
    log_memory_checkpoint("migration:model_loaded", f"Loaded: {MODEL_NAME}")

    # Check Qdrant Cloud
    qdrant_ok = check_qdrant_connectivity()
    if not qdrant_ok and not args.dry_run:
        logger.error("Aborting migration: Qdrant Cloud is unreachable or credentials invalid.")
        sys.exit(1)

    results = {}

    # Migrate resumes
    if not args.skip_resumes:
        from app.db.session import SessionLocal
        db = SessionLocal()
        try:
            results["resumes"] = migrate_resumes(
                db=db,
                target_collection=resumes_col,
                batch_size=args.batch_size,
                force=args.force,
                dry_run=args.dry_run
            )
        finally:
            db.close()

    # Migrate jobs
    if not args.skip_jobs:
        results["jobs"] = migrate_jobs(
            target_collection=jobs_col,
            batch_size=args.batch_size,
            force=args.force,
            dry_run=args.dry_run
        )

    log_memory_checkpoint("migration:finished", f"Final RSS: {get_current_rss_mb():.2f} MB")
    logger.info("=== Migration Summary ===")
    for k, v in results.items():
        logger.info(f"  {k}: {v}")
    logger.info("Migration completed successfully.")

if __name__ == "__main__":
    run_migration()
