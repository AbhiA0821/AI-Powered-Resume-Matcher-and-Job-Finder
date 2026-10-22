"""
HireAgent Embedding Benchmark Suite.
Measures and compares:
1. Process RSS before model initialization
2. Process RSS after model initialization
3. Model loading time
4. Peak RSS and latency during resume chunking and embedding
5. Peak RSS and latency during hybrid job matching
6. Peak RSS and latency during repeated dashboard overview polling
7. Retrieval quality & semantic discrimination on representative resume/job pairs
"""

import os
import sys
import time
import json
import psutil
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def run_single_model_benchmark(model_name: str) -> dict:
    code = f"""
import os
import sys
import time
import psutil
import json

PROJECT_ROOT = r"{PROJECT_ROOT}"
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

os.environ["EMBEDDING_MODEL"] = "{model_name}"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

proc = psutil.Process(os.getpid())

def get_rss():
    return round(proc.memory_info().rss / (1024 * 1024), 2)

# 1. RSS Before Model Init
rss_before_init = get_rss()

# 2. Model Init
t0 = time.time()
from fastembed import TextEmbedding
model = TextEmbedding("{model_name}", threads=1)
# Warm up
_ = list(model.embed(["warmup"]))[0]
load_time = round(time.time() - t0, 3)
rss_after_init = get_rss()

# 3. Resume Embedding Benchmark
sample_resume = (
    "Jane Doe - Senior Machine Learning & Backend Engineer\\n"
    "Summary: 6 years building high-throughput distributed microservices and RAG pipelines.\\n"
    "Skills: Python, FastAPI, PyTorch, Transformers, ONNX Runtime, Qdrant, Docker, PostgreSQL, Redis.\\n"
    "Experience: Tech Lead at DataFlow Systems. Architected semantic search matching engine serving 1M requests/day.\\n"
    "Optimized model inference latency by 75% using ONNX quantization and FastEmbed single-thread workers.\\n"
    "Education: B.S. in Computer Science."
)
chunks = [
    sample_resume[i:i+500] for i in range(0, len(sample_resume), 400)
]

t_res_start = time.time()
resume_vecs = list(model.embed(chunks))
resume_emb_time = round((time.time() - t_res_start) * 1000, 2)
rss_resume_peak = get_rss()

# 4. Job Matching Benchmark
from app.services.jobs import get_jobs_from_db
from app.services.matching_service import match_and_rank_jobs

cat_jobs = get_jobs_from_db(limit=25).get("jobs", [])

t_match_start = time.time()
ranked = match_and_rank_jobs(
    resume_text=sample_resume,
    jobs=cat_jobs,
    target_role="Machine Learning Engineer",
    top_k=10
)
match_latency = round((time.time() - t_match_start) * 1000, 2)
rss_matching_peak = get_rss()

# 5. Repeated Dashboard Polling Simulation
from unittest.mock import MagicMock
from app.models.user import User
from app.models.resume import Resume
from app.services.dashboard_service import assemble_dashboard_overview
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db.session import Base

engine = create_engine("sqlite:///:memory:")
Base.metadata.create_all(bind=engine)
Session = sessionmaker(bind=engine)
db = Session()
u = User(id=999, name="Bench User", email="bench@example.com", password_hash="pw")
db.add(u)
res = Resume(id=888, user_id=999, filename="bench.pdf", file_path="/tmp/bench.pdf", extracted_text=sample_resume)
db.add(res)
db.commit()

t_poll_start = time.time()
for _ in range(15):
    overview = assemble_dashboard_overview(db=db, current_user=u)
poll_latency_15 = round((time.time() - t_poll_start) * 1000, 2)
rss_poll_peak = get_rss()
db.close()

# 6. Semantic Discrimination on Representative Pairs
resume_p1 = "Senior Python Backend Developer with FastAPI, PostgreSQL, and Docker."
job_strong = "Backend Engineer at CloudScale. Role: Python Developer. Skills: Python, FastAPI, SQL, Docker, REST."
job_moderate = "Frontend UI Developer at PixelCraft. Role: React Developer. Skills: TypeScript, React, CSS, Next.js."
job_unrelated = "Certified Public Accountant at AuditGroup. Role: Accountant. Skills: GAAP, tax audit, ledger reconciliation."

vecs = list(model.embed([resume_p1, job_strong, job_moderate, job_unrelated]))
r_v = vecs[0]
js_v = vecs[1]
jm_v = vecs[2]
ju_v = vecs[3]

def cos_sim(a, b):
    return round(float(sum(float(x) * float(y) for x, y in zip(a, b))), 4)

sim_strong = float(cos_sim(r_v, js_v))
sim_moderate = float(cos_sim(r_v, jm_v))
sim_unrelated = float(cos_sim(r_v, ju_v))

results = {{
    "model": "{model_name}",
    "rss_before_init_mb": rss_before_init,
    "rss_after_init_mb": rss_after_init,
    "model_load_time_s": load_time,
    "resume_emb_latency_ms": resume_emb_time,
    "rss_resume_peak_mb": rss_resume_peak,
    "matching_latency_ms": match_latency,
    "rss_matching_peak_mb": rss_matching_peak,
    "poll_latency_15_ms": poll_latency_15,
    "rss_poll_peak_mb": rss_poll_peak,
    "retrieval_quality": {{
        "sim_strong_match": sim_strong,
        "sim_moderate_match": sim_moderate,
        "sim_unrelated_match": sim_unrelated,
        "discrimination_valid": bool(sim_strong > sim_moderate > sim_unrelated)
    }}
}}

print("RESULT_JSON:" + json.dumps(results))
"""

    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    for line in res.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])
    print(f"Error for {model_name}:")
    print(res.stderr)
    return {}

def main():
    print("=== Running HireAgent Embedding Model Benchmarks ===")
    models = [
        "BAAI/bge-small-en-v1.5",
        "sentence-transformers/all-MiniLM-L6-v2"
    ]
    all_results = {}
    for m in models:
        print(f"\\nBenchmarking {m}...")
        res = run_single_model_benchmark(m)
        all_results[m] = res
        print(f" -> Load time: {res.get('model_load_time_s')}s")
        print(f" -> RSS after init: {res.get('rss_after_init_mb')} MB")
        print(f" -> Matching peak RSS: {res.get('rss_matching_peak_mb')} MB (Latency: {res.get('matching_latency_ms')}ms)")
        print(f" -> Polling peak RSS: {res.get('rss_poll_peak_mb')} MB")
        rq = res.get('retrieval_quality', {})
        print(f" -> Similarity: Strong={rq.get('sim_strong_match')}, Moderate={rq.get('sim_moderate_match')}, Unrelated={rq.get('sim_unrelated_match')}")

    out_file = os.path.join(PROJECT_ROOT, "benchmark_results.json")
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\\nSaved results to {out_file}")

if __name__ == "__main__":
    main()
