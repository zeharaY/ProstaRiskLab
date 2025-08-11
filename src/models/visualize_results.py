import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

def create_comparison_plots(metrics, output_dir="outputs/visualizations"):
    """Create comparison plots for model metrics."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(metrics).T
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1. Bar plot for all metrics
    ax1 = axes[0, 0]
    df_plot = df[['Accuracy', 'Precision', 'Recall', 'F1 Score']].T
    df_plot.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('Performance Metrics Comparison')
    ax1.set_ylabel('Score')
    ax1.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.get_xticklabels(), rotation=45)
    
    # 2. ROC AUC comparison
    ax2 = axes[0, 1]
    roc_auc = df['ROC AUC']
    bars = ax2.bar(roc_auc.index, roc_auc.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax2.set_title('ROC AUC Comparison')
    ax2.set_ylabel('ROC AUC Score')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, roc_auc.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. F1 Score comparison
    ax3 = axes[1, 0]
    f1_scores = df['F1 Score']
    bars = ax3.bar(f1_scores.index, f1_scores.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax3.set_title('F1 Score Comparison')
    ax3.set_ylabel('F1 Score')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, f1_scores.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Radar chart (simplified as bar chart for now)
    ax4 = axes[1, 1]
    # Select key metrics for radar-like comparison
    radar_metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    radar_data = df[radar_metrics]
    
    # Create a heatmap-style visualization
    im = ax4.imshow(radar_data.values, cmap='YlOrRd', aspect='auto')
    ax4.set_xticks(range(len(radar_metrics)))
    ax4.set_yticks(range(len(radar_data.index)))
    ax4.set_xticklabels(radar_metrics, rotation=45)
    ax4.set_yticklabels(radar_data.index)
    ax4.set_title('Performance Heatmap')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Score')
    
    # Add value annotations
    for i in range(len(radar_data.index)):
        for j in range(len(radar_metrics)):
            text = ax4.text(j, i, f'{radar_data.iloc[i, j]:.3f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comparison plots saved: {plot_path}")
    
    # Create individual metric plots
    create_individual_plots(df, output_dir)
    
    return plot_path

def create_individual_plots(df, output_dir):
    """Create individual plots for each metric."""
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        values = df[metric]
        bars = plt.bar(values.index, values.values, color=colors)
        
        plt.title(f'{metric} Comparison', fontsize=14, fontweight='bold')
        plt.ylabel(metric, fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Highlight the best performer
        best_model = values.idxmax()
        best_bar = bars[list(values.index).index(best_model)]
        best_bar.set_color('#FFD93D')
        best_bar.set_edgecolor('black')
        best_bar.set_linewidth(2)
        
        plt.tight_layout()
        
        # Save individual plot
        plot_path = os.path.join(output_dir, f'{metric.lower().replace(" ", "_")}_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ {metric} plot saved: {plot_path}")

def create_summary_table(metrics, output_dir="outputs/visualizations"):
    """Create a summary table visualization."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(metrics).T.round(4)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=df.values,
                    rowLabels=df.index,
                    colLabels=df.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Color the header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color the best performers
    for col in df.columns:
        best_idx = df[col].idxmax()
        row_idx = list(df.index).index(best_idx) + 1  # +1 for header
        table[(row_idx, list(df.columns).index(col))].set_facecolor('#FFD93D')
    
    plt.title('Model Performance Summary', fontsize=16, fontweight='bold', pad=20)
    
    # Save table
    table_path = os.path.join(output_dir, 'performance_summary_table.png')
    plt.savefig(table_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Summary table saved: {table_path}")

def main():
    """Main function to generate all visualizations."""
    try:
        # Load metrics
        metrics = load_metrics()
        
        if not metrics:
            print("❌ No metrics files found!")
            return
        
        print(f"📊 Generating visualizations for {len(metrics)} models...")
        
        # Create all visualizations
        create_comparison_plots(metrics)
        create_summary_table(metrics)
        
        print("✅ All visualizations generated successfully!")
        print(f"📁 Check the 'outputs/visualizations/' directory for all plots")
        
    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")

if __name__ == "__main__":
    main()
