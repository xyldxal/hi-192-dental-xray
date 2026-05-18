import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

def generate_eda_charts():
    print("Loading data...")
    # Paths
    base_dir = Path(r"c:\Users\jamil\Downloads\hi192\hi-192-dental-xray")
    splits_dir = base_dir / "artifacts" / "multilabel_splits"
    output_dir = base_dir / "artifacts" / "multilabel_eda"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load splits
    train_df = pd.read_csv(splits_dir / "train_manifest.csv")
    val_df = pd.read_csv(splits_dir / "val_manifest.csv")
    test_df = pd.read_csv(splits_dir / "test_manifest.csv")

    # Combine for overall EDA
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # Identify label columns
    label_cols = [col for col in df.columns if col.startswith('label_') and col != 'label_count']
    clean_labels = [col.replace('label_', '').replace('_', ' ').title() for col in label_cols]
    
    print(f"Found {len(df)} total images and {len(label_cols)} label categories.")

    # 1. Label Distribution (Bar Chart)
    print("Generating Label Distribution Chart...")
    label_counts = df[label_cols].sum().values
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(clean_labels, label_counts, color=sns.color_palette("viridis", len(label_cols)))
    plt.title('Distribution of Dental Conditions (All Splits)', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Images', fontsize=12)
    plt.xlabel('Condition', fontsize=12)
    plt.xticks(rotation=15)
    
    # Add counts above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{int(height)}', ha='center', va='bottom', fontsize=11)
                
    plt.tight_layout()
    plt.savefig(output_dir / "01_label_distribution.png", dpi=300)
    plt.close()

    # 2. Co-occurrence Matrix (Heatmap)
    print("Generating Co-occurrence Matrix...")
    # Compute dot product of the label matrix with its transpose
    label_matrix = df[label_cols].values
    co_occurrence = np.dot(label_matrix.T, label_matrix)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(co_occurrence, annot=True, fmt='d', cmap='Blues', 
                xticklabels=clean_labels, yticklabels=clean_labels)
    plt.title('Condition Co-occurrence Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "02_co_occurrence_matrix.png", dpi=300)
    plt.close()

    # 3. Number of Labels per Image (Histogram)
    print("Generating Label Count Histogram...")
    plt.figure(figsize=(8, 5))
    label_counts_per_image = df['label_count']
    sns.countplot(x=label_counts_per_image, palette='Set2')
    plt.title('Number of Conditions per Image', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Labels', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    
    # Add percentages
    total = len(df)
    for p in plt.gca().patches:
        height = p.get_height()
        plt.text(p.get_x() + p.get_width()/2., height + 2,
                f'{height/total*100:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_dir / "03_labels_per_image.png", dpi=300)
    plt.close()

    # 4. Save a JSON summary of stats
    print("Saving statistical summary...")
    stats = {
        "total_images": int(len(df)),
        "train_images": int(len(train_df)),
        "val_images": int(len(val_df)),
        "test_images": int(len(test_df)),
        "label_frequencies": {clean_labels[i]: int(label_counts[i]) for i in range(len(label_cols))}
    }
    
    with open(output_dir / "eda_summary.json", 'w') as f:
        json.dump(stats, f, indent=4)
        
    print(f"\n✅ EDA Complete! All charts saved to: {output_dir}")
    print("You can insert these PNG files directly into your thesis document.")

if __name__ == "__main__":
    generate_eda_charts()
