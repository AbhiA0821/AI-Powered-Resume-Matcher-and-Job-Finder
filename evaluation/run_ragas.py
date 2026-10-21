"""
Direct entrypoint for Ragas evaluation: python -m evaluation.run_ragas
"""
import sys
from evaluation.runners.eval_ragas import run_ragas_evaluation

if __name__ == "__main__":
    live_flag = "--live" in sys.argv
    run_ragas_evaluation(use_live_llm=live_flag)
