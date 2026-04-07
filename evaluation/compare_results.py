import os
import pandas as pd

def calculate_improvement():
    root_dir = os.path.dirname(os.path.dirname(__file__))
    hybrid_path = os.path.join(root_dir, "ragas_results_hybrid.csv")
    baseline_path = os.path.join(root_dir, "ragas_results_baseline.csv")

    if not os.path.exists(hybrid_path) or not os.path.exists(baseline_path):
        print("Error: Could not find both 'ragas_results_hybrid.csv' and 'ragas_results_baseline.csv'.")
        print("Make sure you run the ragas_evaluator.py script first.")
        return

    df_hybrid = pd.read_csv(hybrid_path)
    df_baseline = pd.read_csv(baseline_path)

    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    
    output_lines = []
    output_lines.append("=== PERFORMANCE COMPARISON ===")
    
    total_base = 0.0
    total_hyb = 0.0
    valid_metrics = 0
    
    # Calculate per-metric values
    for m in metrics:
        if m in df_baseline.columns and m in df_hybrid.columns:
            base_score = df_baseline[m].mean()
            hyb_score = df_hybrid[m].mean()
            
            total_base += base_score
            total_hyb += hyb_score
            valid_metrics += 1
            
            # Format improvement
            if base_score > 0:
                pct_improvement = ((hyb_score - base_score) / base_score) * 100
            else:
                pct_improvement = 0.0
                
            sign = "+" if pct_improvement >= 0 else ""
            metric_name = m.replace("_", " ").title() + ":"
            
            line = f"{metric_name:<18} Baseline {base_score:.3f} \u2192 Hybrid {hyb_score:.3f}  ({sign}{pct_improvement:.1f}%)"
            output_lines.append(line)
            print(line)

    # Calculate overall average
    if valid_metrics > 0:
        avg_base = total_base / valid_metrics
        avg_hyb = total_hyb / valid_metrics
        
        if avg_base > 0:
            avg_pct = ((avg_hyb - avg_base) / avg_base) * 100
        else:
            avg_pct = 0.0
            
        sign = "+" if avg_pct >= 0 else ""
        line = f"{'Overall Average':<18} Baseline {avg_base:.3f} \u2192 Hybrid {avg_hyb:.3f}  ({sign}{avg_pct:.1f}%)"
        output_lines.append(line)
        print(line)

    # Save to file
    out_path = os.path.join(root_dir, "final_comparison.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
        
    print(f"\nSaved this comparison to '{out_path}'")

if __name__ == "__main__":
    calculate_improvement()
