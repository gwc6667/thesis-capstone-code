from pathlib import Path
import csv


def main():
    root = Path(__file__).resolve().parents[1]
    csv_path = root / 'results' / 'training_metrics.csv'
    if not csv_path.exists():
        print('No results file found. Run training scripts first.')
        return

    with csv_path.open('r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print('No rows found in training_metrics.csv')
        return

    print('Table 4.1: Performance Comparison of All Models\n')
    header = f"{'Model':<15}{'Accuracy':>12}{'Precision':>12}{'Recall':>12}{'F1 Score':>12}"
    print(header)
    print('-' * len(header))
    for row in rows:
        print(
            f"{row.get('model', ''):<15}"
            f"{float(row.get('final_val_acc', 0) or 0):>12.4f}"
            f"{float(row.get('final_precision', 0) or 0):>12.4f}"
            f"{float(row.get('final_recall', 0) or 0):>12.4f}"
            f"{float(row.get('final_f1_score', 0) or 0):>12.4f}"
        )


if __name__ == '__main__':
    main()
