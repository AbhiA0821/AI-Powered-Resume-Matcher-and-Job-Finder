"""
Master Evaluation Suite Runner for HireAgent.
Executes and coordinates all evaluation modules:
- Phase 2: Ragas RAG Pipeline Evaluation (--ragas)
- Phase 3: Information Retrieval Benchmark (--retrieval)
- Phase 4: BGE Embeddings & Hybrid Matching (--matching)
- Phase 5: LLM Quality & Provider Reliability (--llm)
- Phase 6: LangGraph & CrewAI Agents (--agents)
- Phase 7: Tool Calling & Security Isolation (--tools)
- Phase 9: Latency & End-to-End Performance (--perf)
- Run all: python -m evaluation.run_eval --all
"""

import sys
import argparse
import time

from evaluation.runners.eval_ragas import run_ragas_evaluation
from evaluation.runners.eval_retrieval import run_retrieval_evaluation
from evaluation.runners.eval_embeddings_matching import run_embeddings_matching_evaluation
from evaluation.runners.eval_llm_quality import run_llm_quality_evaluation
from evaluation.runners.eval_agents import run_agents_evaluation
from evaluation.runners.eval_tools_security import run_tools_security_evaluation
from evaluation.runners.eval_performance import run_performance_evaluation


def main():
    parser = argparse.ArgumentParser(description="HireAgent Comprehensive Evaluation Framework")
    parser.add_argument("--all", action="store_true", help="Run the complete end-to-end evaluation suite across all phases")
    parser.add_argument("--ragas", action="store_true", help="Run Phase 2 Ragas metrics evaluation")
    parser.add_argument("--retrieval", action="store_true", help="Run Phase 3 IR retrieval evaluation")
    parser.add_argument("--matching", action="store_true", help="Run Phase 4 BGE embeddings & hybrid matching evaluation")
    parser.add_argument("--llm", action="store_true", help="Run Phase 5 LLM quality & provider comparison")
    parser.add_argument("--agents", action="store_true", help="Run Phase 6 LangGraph & CrewAI agents evaluation")
    parser.add_argument("--tools", action="store_true", help="Run Phase 7 tool calling & security isolation evaluation")
    parser.add_argument("--perf", action="store_true", help="Run Phase 9 latency & performance evaluation")
    parser.add_argument("--live", action="store_true", help="Enable live external API calls where applicable")

    args = parser.parse_args()

    # Default to --all if no specific phase flag is provided
    run_all = args.all or not any([args.ragas, args.retrieval, args.matching, args.llm, args.agents, args.tools, args.perf])

    start_time = time.time()
    print("=" * 80)
    print("           HIREAGENT COMPREHENSIVE AI & PIPELINE EVALUATION SUITE           ")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print(f"Execution Mode: {'LIVE APIS ENABLED' if args.live else 'STANDARDIZED BENCHMARK / OFFLINE'}")
    print("=" * 80)

    results_summary = {}

    if run_all or args.ragas:
        print("\n>>> EXECUTING PHASE 2: RAGAS EVALUATION <<<")
        r = run_ragas_evaluation(use_live_llm=args.live)
        results_summary["ragas"] = r["overall_metrics"]

    if run_all or args.retrieval:
        print("\n>>> EXECUTING PHASE 3: RETRIEVAL EVALUATION <<<")
        ret = run_retrieval_evaluation()
        results_summary["retrieval"] = ret["mean_metrics"]
        results_summary["qdrant_infra"] = ret["qdrant_infrastructure"]["status"]

    if run_all or args.matching:
        print("\n>>> EXECUTING PHASE 4: EMBEDDINGS & HYBRID MATCHING <<<")
        m = run_embeddings_matching_evaluation()
        results_summary["embeddings"] = m["bge_embedding_evaluation"]["status"]
        results_summary["matching"] = m["hybrid_matcher_evaluation"]["status"]

    if run_all or args.llm:
        print("\n>>> EXECUTING PHASE 5: LLM QUALITY & RELIABILITY <<<")
        l = run_llm_quality_evaluation(use_live_apis=args.live)
        results_summary["llm_quality"] = {p: l["providers_evaluated"][p]["status"] for p in l["providers_evaluated"]}

    if run_all or args.agents:
        print("\n>>> EXECUTING PHASE 6: LANGGRAPH & CREWAI AGENTS <<<")
        ag = run_agents_evaluation()
        results_summary["langgraph"] = ag["langgraph_evaluation"]["status"]
        results_summary["crewai"] = ag["crewai_evaluation"]["status"]

    if run_all or args.tools:
        print("\n>>> EXECUTING PHASE 7: TOOL CALLING & SECURITY <<<")
        t = run_tools_security_evaluation()
        results_summary["tool_calling"] = t["tool_calling_evaluation"]["status"]
        results_summary["security"] = t["security_evaluation"]["status"]

    if run_all or args.perf:
        print("\n>>> EXECUTING PHASE 9: PERFORMANCE BENCHMARK <<<")
        p = run_performance_evaluation()
        results_summary["performance"] = {
            "cold_start": f"{p['cold_start_seconds']}s",
            "warm_mean": f"{p['warm_dashboard_mean_seconds']}s",
            "speedup": f"{p['warm_speedup_factor']}x"
        }

    total_time = round(time.time() - start_time, 2)
    print("\n" + "=" * 80)
    print(f"ALL EVALUATIONS COMPLETED IN {total_time}s")
    print("=" * 80)
    for k, v in results_summary.items():
        print(f" - {k.upper():<20}: {v}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
