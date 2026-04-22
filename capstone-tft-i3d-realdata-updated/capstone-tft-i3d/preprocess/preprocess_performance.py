from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import json

import numpy as np
import pandas as pd
import torch


OULAD_FILES = {
    'studentInfo.csv',
    'studentAssessment.csv',
    'studentVle.csv',
    'assessments.csv',
    'courses.csv',
}


def _resolve_oulad_dir(data_dir: Path) -> Path:
    data_dir = Path(data_dir)
    if OULAD_FILES.issubset({p.name for p in data_dir.glob('*.csv')}):
        return data_dir
    for candidate in data_dir.rglob('*'):
        if candidate.is_dir() and OULAD_FILES.issubset({p.name for p in candidate.glob('*.csv')}):
            return candidate
    raise FileNotFoundError(
        'Could not find OULAD CSV files. Place them under data/performance/ or a subfolder.'
    )


def build_oulad_tensors(data_dir: Path, seq_len: int = 12, cache: bool = True):
    data_dir = _resolve_oulad_dir(Path(data_dir))
    processed_dir = data_dir / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_path = processed_dir / f'oulad_tft_seq{seq_len}.pt'
    meta_path = processed_dir / f'oulad_tft_seq{seq_len}_meta.json'

    if cache and cache_path.exists() and meta_path.exists():
        saved = torch.load(cache_path, map_location='cpu')
        with meta_path.open('r', encoding='utf-8') as f:
            meta = json.load(f)
        return saved['x'], saved['y'], meta

    info = pd.read_csv(data_dir / 'studentInfo.csv')
    info = info.dropna(subset=['final_result']).copy()
    info['target'] = info['final_result'].isin(['Pass', 'Distinction']).astype(np.float32)

    courses = pd.read_csv(data_dir / 'courses.csv')
    course_lengths = {
        (row.code_module, row.code_presentation): float(row.module_presentation_length)
        for row in courses.itertuples(index=False)
    }

    assessments = pd.read_csv(data_dir / 'assessments.csv')
    stu_assess = pd.read_csv(data_dir / 'studentAssessment.csv')
    assess = stu_assess.merge(
        assessments[['id_assessment', 'code_module', 'code_presentation', 'date', 'weight']],
        on='id_assessment',
        how='left',
    )
    assess['date_used'] = assess['date_submitted'].fillna(assess['date']).fillna(0)
    assess['weight'] = assess['weight'].fillna(0)
    assess['score'] = assess['score'].fillna(0)

    assessment_bins = defaultdict(lambda: {'score_sum': 0.0, 'weight_sum': 0.0, 'count': 0.0})
    for row in assess.itertuples(index=False):
        length = course_lengths.get((row.code_module, row.code_presentation), 270.0)
        clipped_date = min(max(float(row.date_used), 0.0), max(length - 1.0, 0.0))
        bin_idx = int(min(seq_len - 1, (clipped_date / max(length, 1.0)) * seq_len))
        key = (row.code_module, row.code_presentation, int(row.id_student), bin_idx)
        assessment_bins[key]['score_sum'] += float(row.score) * max(float(row.weight), 1.0)
        assessment_bins[key]['weight_sum'] += max(float(row.weight), 1.0)
        assessment_bins[key]['count'] += 1.0

    vle_bins = defaultdict(lambda: {'clicks': 0.0, 'active_days': 0.0})
    seen_days = set()
    for chunk in pd.read_csv(data_dir / 'studentVle.csv', chunksize=500_000):
        chunk = chunk[['code_module', 'code_presentation', 'id_student', 'date', 'sum_click']].copy()
        for row in chunk.itertuples(index=False):
            length = course_lengths.get((row.code_module, row.code_presentation), 270.0)
            clipped_date = min(max(float(row.date), 0.0), max(length - 1.0, 0.0))
            bin_idx = int(min(seq_len - 1, (clipped_date / max(length, 1.0)) * seq_len))
            key = (row.code_module, row.code_presentation, int(row.id_student), bin_idx)
            vle_bins[key]['clicks'] += float(row.sum_click)
            day_key = (row.code_module, row.code_presentation, int(row.id_student), bin_idx, int(clipped_date))
            if day_key not in seen_days:
                seen_days.add(day_key)
                vle_bins[key]['active_days'] += 1.0

    # Build tensors with 5 features per time step.
    x_list, y_list = [], []
    feature_names = [
        'vle_clicks_norm',
        'active_days_norm',
        'assessment_score_norm',
        'assessment_count_norm',
        'studied_credits_norm',
    ]

    credits = info['studied_credits'].fillna(info['studied_credits'].median())
    credits_scale = max(float(credits.max()), 1.0)

    for row in info.itertuples(index=False):
        seq = np.zeros((seq_len, 5), dtype=np.float32)
        static_credit = float(getattr(row, 'studied_credits', 0.0) or 0.0) / credits_scale
        for t in range(seq_len):
            key = (row.code_module, row.code_presentation, int(row.id_student), t)
            vle_row = vle_bins.get(key, None)
            if vle_row is not None:
                seq[t, 0] = np.log1p(vle_row['clicks']) / 8.0
                seq[t, 1] = vle_row['active_days'] / 31.0
            assess_row = assessment_bins.get(key, None)
            if assess_row is not None and assess_row['weight_sum'] > 0:
                seq[t, 2] = (assess_row['score_sum'] / assess_row['weight_sum']) / 100.0
                seq[t, 3] = assess_row['count'] / 10.0
            seq[t, 4] = static_credit

        x_list.append(seq)
        y_list.append(float(row.target))

    x = torch.tensor(np.stack(x_list), dtype=torch.float32)
    y = torch.tensor(np.asarray(y_list), dtype=torch.float32)

    meta = {
        'num_samples': int(len(y_list)),
        'seq_len': seq_len,
        'input_size': 5,
        'feature_names': feature_names,
        'positive_rate': float(np.mean(y_list)),
        'source': str(data_dir),
    }

    if cache:
        torch.save({'x': x, 'y': y}, cache_path)
        with meta_path.open('w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)

    return x, y, meta


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / 'data' / 'performance'
    x, y, meta = build_oulad_tensors(data_dir=data_dir, seq_len=12, cache=True)
    print(f'Built OULAD tensors from: {meta["source"]}')
    print(f'x shape: {tuple(x.shape)}')
    print(f'y shape: {tuple(y.shape)}')
    print(f'positive rate: {meta["positive_rate"]:.4f}')


if __name__ == '__main__':
    main()
