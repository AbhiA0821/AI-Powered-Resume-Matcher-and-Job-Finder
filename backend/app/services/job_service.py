from app.services.jobs import (
    get_jobs_from_db,
    get_job_by_id_from_db
)
from app.services.job_indexing import (
    build_job_embedding_text,
    construct_job_text_representation,
    generate_deterministic_job_point_id,
    index_all_jobs,
    index_live_jobs_in_qdrant,
    search_jobs_by_query_semantic,
    retrieve_matched_jobs_for_resume
)
from app.services.job_sources import (
    search_live_jobs,
    to_hireagent_job_dict
)

def search_jobs_by_title_or_skill(query: str, limit: int = 50):
    return get_jobs_from_db(search=query, limit=limit)

__all__ = [
    "get_jobs_from_db",
    "get_job_by_id_from_db",
    "search_jobs_by_title_or_skill",
    "build_job_embedding_text",
    "construct_job_text_representation",
    "generate_deterministic_job_point_id",
    "index_all_jobs",
    "index_live_jobs_in_qdrant",
    "search_jobs_by_query_semantic",
    "retrieve_matched_jobs_for_resume",
    "search_live_jobs",
    "to_hireagent_job_dict"
]
