from __future__ import annotations

from pathlib import Path
import json
from collections import Counter


VALID_VIDEO_EXTENSIONS = {'.avi', '.mp4', '.mov', '.mkv'}


def resolve_enterface_dir(data_dir: Path) -> Path:
    data_dir = Path(data_dir)
    candidates = [
        data_dir / 'enterface_database',
        data_dir / 'enterface database',
        data_dir,
    ]
    for candidate in candidates:
        if candidate.exists() and any(p.is_dir() and p.name.lower().startswith('subject') for p in candidate.iterdir()):
            return candidate
    for candidate in data_dir.rglob('*'):
        if candidate.is_dir() and any(p.is_dir() and p.name.lower().startswith('subject') for p in candidate.iterdir()):
            return candidate
    raise FileNotFoundError(
        'Could not find the eNTERFACE video folder. Place it under data/emotions/enterface_database.'
    )


def index_enterface_videos(data_dir: Path):
    root = resolve_enterface_dir(Path(data_dir))
    samples = []
    for subject_dir in sorted([p for p in root.iterdir() if p.is_dir() and p.name.lower().startswith('subject')], key=lambda p: p.name.lower()):
        subject_name = subject_dir.name
        for emotion_dir in sorted([p for p in subject_dir.iterdir() if p.is_dir()]):
            emotion = emotion_dir.name.lower().strip()
            for sentence_dir in sorted([p for p in emotion_dir.iterdir() if p.is_dir()]):
                for video_path in sorted(sentence_dir.iterdir()):
                    if video_path.suffix.lower() not in VALID_VIDEO_EXTENSIONS:
                        continue
                    samples.append({
                        'path': str(video_path),
                        'subject': subject_name,
                        'emotion': emotion,
                        'sentence': sentence_dir.name,
                    })
    if not samples:
        raise FileNotFoundError('No video files found in the eNTERFACE directory.')
    return root, samples


def build_emotion_index(data_dir: Path):
    root, samples = index_enterface_videos(data_dir)
    emotions = sorted({sample['emotion'] for sample in samples})
    label_map = {emotion: idx for idx, emotion in enumerate(emotions)}
    for sample in samples:
        sample['label'] = label_map[sample['emotion']]
    summary = {
        'source': str(root),
        'num_samples': len(samples),
        'num_subjects': len({sample['subject'] for sample in samples}),
        'emotions': emotions,
        'counts': dict(Counter(sample['emotion'] for sample in samples)),
    }
    return samples, label_map, summary


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / 'data' / 'emotions'
    samples, label_map, summary = build_emotion_index(data_dir)
    print(f'Emotion data source: {summary["source"]}')
    print(f'Samples: {summary["num_samples"]}')
    print(f'Subjects: {summary["num_subjects"]}')
    print(f'Emotion labels: {label_map}')

    output = root / 'results' / 'enterface_index_summary.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
