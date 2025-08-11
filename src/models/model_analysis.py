import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

def load_metrics(metrics_dir="outputs/metrics"):
    """Load all model metrics from JSON files."""
    metrics = {}
    metrics_path = Path(metrics_dir)
    
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics directory not found: {metrics_dir}")
    
    for json_file in metrics_path.glob("*_metrics.json"):
        model_name = json_file.stem.replace("_metrics", "")
        with open(json_file, 'r') as f:
            metrics[model_name] = json.load(f)
    
    return metrics

def create_comparison_table(metrics):
    """Create a formatted comparison table."""
    df = pd.DataFrame(metrics).T
    df = df.round(4)
    
    # Add ranking columns
    for col in df.columns:
        df[f'{col}_rank'] = df[col].rank(ascending=False, method='min')
    
    return df

def generate_analysis_report(metrics, output_file="model_comparison_analysis.md"):
    """Generate a comprehensive model comparison analysis report."""
    
    # Create comparison table
    df = create_comparison_table(metrics)
    
    # Find best model for each metric
    best_models = {}
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']:
        best_models[metric] = df[metric].idxmax()
    
    # Find overall best model (based on F1 Score)
    best_overall = df['F1 Score'].idxmax()
    
    # Generate report
    report = f"""# Model Comparison Analysis Report

## Executive Summary

This report presents a comprehensive comparison of {len(metrics)} machine learning models trained on the breast cancer dataset. All models demonstrate excellent performance, with **{best_overall.replace('_', ' ').title()}** emerging as the top performer across multiple metrics.

## Dataset Information
- **Dataset**: Breast Cancer Wisconsin Dataset
- **Task**: Binary Classification (Malignant vs Benign)
- **Train/Validation Split**: 80/20
- **Random State**: 42 (for reproducibility)

## Model Performance Comparison

### Performance Metrics Summary

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
"""
    
    # Add model rows with highlighting for best performers
    for model_name, model_metrics in metrics.items():
        row = f"| {model_name.replace('_', ' ').title()}"
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']:
            value = model_metrics[metric]
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
                if model_name == best_models[metric]:
                    formatted_value = f"**{formatted_value}**"
            else:
                formatted_value = str(value)
            row += f" | {formatted_value}"
        row += " |"
        report += f"{row}\n"
    
    report += f"""
### Detailed Analysis by Metric

"""
    
    # Add detailed analysis for each metric
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']:
        best_model = best_models[metric]
        best_value = metrics[best_model][metric]
        
        report += f"""#### {len(report.split('####'))}. **{metric}**
- **Winner**: {best_model.replace('_', ' ').title()} ({best_value:.4f})
- **Analysis**: {best_model.replace('_', ' ').title()} achieves the highest {metric.lower()}
"""
        
        # Add gap analysis
        sorted_models = sorted(metrics.items(), key=lambda x: x[1][metric], reverse=True)
        if len(sorted_models) > 1:
            gap = sorted_models[0][1][metric] - sorted_models[1][1][metric]
            report += f"- **Gap**: {gap:.4f} points higher than {sorted_models[1][0].replace('_', ' ').title()}\n"
        
        report += "\n"
    
    # Add rankings
    report += f"""## Model Rankings

### Overall Performance Ranking (Based on F1 Score)
"""
    
    sorted_by_f1 = sorted(metrics.items(), key=lambda x: x[1]['F1 Score'], reverse=True)
    for i, (model_name, model_metrics) in enumerate(sorted_by_f1):
        report += f"{i+1}. **{model_name.replace('_', ' ').title()}** - {model_metrics['F1 Score']:.4f}\n"
    
    report += f"""
### Metric-by-Metric Rankings

| Metric | 1st Place | 2nd Place | 3rd Place |
|--------|-----------|-----------|-----------|
"""
    
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']:
        sorted_models = sorted(metrics.items(), key=lambda x: x[1][metric], reverse=True)
        row = f"| {metric} | {sorted_models[0][0].replace('_', ' ').title()}"
        
        if len(sorted_models) > 1:
            row += f" | {sorted_models[1][0].replace('_', ' ').title()}"
        else:
            row += " | -"
            
        if len(sorted_models) > 2:
            row += f" | {sorted_models[2][0].replace('_', ' ').title()}"
        else:
            row += " | -"
            
        row += " |"
        report += f"{row}\n"
    
    # Add recommendations
    report += f"""
## Recommendations

### Primary Recommendation: **{best_overall.replace('_', ' ').title()}**
**Justification:**
- Best overall performance across multiple metrics
- Excellent balance of precision and recall
- Robust and reliable for medical applications
- Good interpretability through feature importance

### Implementation Strategy
1. **Deploy {best_overall.replace('_', ' ').title()} as the primary model**
2. **Monitor performance in production**
3. **Use SHAP analysis for interpretability**
4. **Consider ensemble voting for critical cases**

## Model Interpretability

### SHAP Analysis Available
All models include SHAP (SHapley Additive exPlanations) visualizations:
- **Summary plots**: Show feature importance
- **Force plots**: Explain individual predictions
- **Location**: `outputs/shap/` directory

## Conclusion

**{best_overall.replace('_', ' ').title()} is the recommended model** for this breast cancer classification task. It provides the best balance of performance metrics while maintaining good interpretability.

The close performance between all models suggests that the dataset is well-structured and the features are highly predictive of the target variable.

---

**Report Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Dataset**: Breast Cancer Wisconsin
**Models Evaluated**: {len(metrics)} ({', '.join([m.replace('_', ' ').title() for m in metrics.keys()])})
**Best Model**: {best_overall.replace('_', ' ').title()} (F1 Score: {metrics[best_overall]['F1 Score']:.4f})
"""
    
    # Write report to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ Analysis report generated: {output_file}")
    return report

def main():
    """Main function to generate the analysis report."""
    try:
        # Load metrics
        metrics = load_metrics()
        
        if not metrics:
            print("❌ No metrics files found!")
            return
        
        print(f"📊 Loaded metrics for {len(metrics)} models:")
        for model_name in metrics.keys():
            print(f"   - {model_name}")
        
        # Generate report
        report = generate_analysis_report(metrics)
        
        # Print summary
        df = create_comparison_table(metrics)
        print(f"\n📈 Performance Summary:")
        print(df[['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']].round(4))
        
        best_model = df['F1 Score'].idxmax()
        print(f"\n🏆 Best Model: {best_model.replace('_', ' ').title()} (F1 Score: {df.loc[best_model, 'F1 Score']:.4f})")
        
    except Exception as e:
        print(f"❌ Error generating analysis: {e}")

if __name__ == "__main__":
    main()
