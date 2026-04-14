"""
RAGAS Evaluator - Runs batch evaluation using RAGAS metrics.
"""

import json
import os
import sys

import pandas as pd
from datasets import Dataset

# Ensure the project root is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set environment variable to avoid tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def run_ragas_evaluation(rag_pipeline_func, test_questions_path, output_csv_name="ragas_results.csv"):
    """
    Runs batch evaluation on a list of test questions using RAGAS.
    """
    import logging
    # Temporarily suppress some noisy Ragas logs if needed
    logging.getLogger("ragas").setLevel(logging.WARNING)

    # 1. Load test questions
    with open(test_questions_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    print(f"\nLoaded {len(test_data)} test questions from {test_questions_path}")
    print("Running queries through the RAG pipeline...\n")

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    # 2. For each question, run the RAG pipeline
    for i, item in enumerate(test_data, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        print(f"[{i}/{len(test_data)}] Evaluating: {question[:60]}...")
        
        # Run the pipeline function (assumes it takes just the query string)
        result = rag_pipeline_func(question)
        
        # Extract fields
        answer = result["answer"]
        chunks = result["retrieved_chunks"]
        chunk_texts = [chunk["chunk_text"] for chunk in chunks]
        
        # Append to lists
        questions.append(question)
        answers.append(answer)
        contexts.append(chunk_texts)
        ground_truths.append([ground_truth])  # RAGAS often expects a list of ground truths or just string depending on version, newer uses list

    # 3. Build a dataset dictionary
    dataset_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }

    # 4. Convert to HuggingFace Dataset
    dataset = Dataset.from_dict(dataset_dict)
    
    print("\nStarting RAGAS evaluation (this may take a few minutes and uses OpenAI API)...")

    # Import ragas inside so it doesn't slow down script loading if not used
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    )

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]

    # 5. Run RAGAS evaluate()
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
    )

    # 6. Convert results to pandas dataframe
    df = result.to_pandas()

    # 7. Print the full results dataframe
    print("\n" + "=" * 80)
    print("DETAILED RESULTS DATAFRAME")
    print("=" * 80)
    # Print wide so columns aren't wrapped too badly
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df.head(10))

    # 8. Print a summary
    print("\n" + "=" * 40)
    print("=== RAGAS EVALUATION SUMMARY ===")
    
    # Calculate overall average based on the metrics used
    metric_names = [m.name for m in metrics]
    total = 0.0
    valid_metrics = 0
    
    for metric in metric_names:
        if metric in result:
            val = result[metric]
            print(f"{metric}: {val:.3f}")
            total += val
            valid_metrics += 1
            
    if valid_metrics > 0:
        overall_avg = total / valid_metrics
        print(f"Overall Average: {overall_avg:.3f}")
    
    print("=" * 40)

    # 9. Save results to CSV in the root folder
    csv_path = os.path.join(os.path.dirname(__file__), "..", output_csv_name)
    df.to_csv(csv_path, index=False)
    print(f"\n[SUCCESS] Detailed evaluation results saved to {csv_path}")

    # 10. Return the dataframe
    return df


if __name__ == "__main__":
    from main import run_rag, build_pipeline
    from evaluation.baseline_rag import run_baseline_rag

    print("Initializing components for evaluation...")
    
    # Build the retriever once
    retriever = build_pipeline()

    # Create a wrapper function that passes the retriever to run_rag automatically
    def hybrid_rag_wrapper(query):
        return run_rag(query, retriever)

    test_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test_questions.json"))
    
    print("\n" + "=" * 60)
    print("RUNNING HYBRID EVALUATION")
    print("=" * 60)
    df_hybrid = run_ragas_evaluation(hybrid_rag_wrapper, test_path, "ragas_results_hybrid.csv")

    print("\n" + "=" * 60)
    print("RUNNING BASELINE EVALUATION")
    print("=" * 60)
    df_baseline = run_ragas_evaluation(run_baseline_rag, test_path, "ragas_results_baseline.csv")

    # Metrics comparison
    print("\n" + "=" * 65)
    print("COMPARISON: BASELINE VS HYBRID RAG")
    print("=" * 65)
    
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    
    print(f"{'Metric':<20} | {'Baseline':<10} | {'Hybrid RAG':<12} | {'Improvement'}")
    print("-" * 65)
    
    for m in metrics:
        if m in df_baseline.columns and m in df_hybrid.columns:
            base_score = df_baseline[m].mean()
            hyb_score = df_hybrid[m].mean()
            diff = hyb_score - base_score
            sign = "+" if diff >= 0 else ""
            print(f"{m:<20} | {base_score:<10.3f} | {hyb_score:<12.3f} | {sign}{diff:.3f}")
    
    print("=" * 65)
