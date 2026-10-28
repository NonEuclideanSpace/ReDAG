import argparse
import json
from pathlib import Path
import unicodedata

from agentic_image.core.types import Canvas, CheckSpec, PublicCheckAtom, PublicTask, RequestSpan, UserRequest


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='data/training_expansion')
    parser.add_argument('--pairs', default='data/expanded-ai-pairs-v1/renderer.jsonl')
    parser.add_argument('--output', default='data/public/expanded_training128.tasks.jsonl')
    args = parser.parse_args()
    root = Path(args.root)
    selected = read(root / 'training_selection_v1.frozen.jsonl')
    pairs = {row['sample_id']: row for row in read(args.pairs)}
    originals = {row['sample_id']: row for folder in root.glob('candidates*') if (folder / 'candidates.frozen.jsonl').is_file() for row in read(folder / 'candidates.frozen.jsonl')}
    tasks = []
    if len(selected) != 128 or len({row['family_id'] for row in selected}) != 128:
        raise ValueError('Exactly the frozen 128 independent families are required')
    private_ids = {row['target_image_id'] for row in pairs.values()}
    for selection in selected:
        sample = selection['sample_id']
        original, pair = originals[sample], pairs[sample]
        text = unicodedata.normalize('NFC', original['instruction'])
        if pair['family_id'] != selection['family_id'] or pair['condition_recipe']['plan']['global_text'] != original['instruction']:
            raise ValueError('Frozen public identity or request changed')
        source = pair['condition_recipe']['source_image_id']
        if source in private_ids:
            raise ValueError('A private target cannot be a public source')
        task = PublicTask(task_id=sample, split_id='train', family_id=selection['family_id'],
            source_dataset=selection['source_dataset'], license_record=original['license_record'],
            user_request=UserRequest(text=text, source_image_id=source, spans=(RequestSpan(span_id='span:request', start=0, end=len(text), text=text),)),
            canvas=Canvas(height=512, width=512), task_mode='edit',
            check_spec=CheckSpec(extractor_version='original-request-public-only-fallback-v1', fallback=True,
                atoms=(PublicCheckAtom(check_id='check:request', origin='request', raw_span_ids=('span:request',), predicate=text, expected_value=text, confidence=0.0),)))
        row = task.model_dump(mode='json')
        if any(identifier in json.dumps(row) for identifier in private_ids):
            raise ValueError('A hidden target identifier leaked into the public task')
        tasks.append(row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and read(output) != tasks:
        raise ValueError('Frozen public tasks may not change')
    output.write_text(''.join(json.dumps(row) + '\n' for row in tasks))
    report = {'tasks': 128, 'independent_families': 128, 'public_source_and_original_request_only': True,
              'private_target_identifiers': 0, 'private_reward_or_calibration_included': False,
              'public_checks': 'Unverified original-request fallback; no fabricated structured checker extraction',
              'source_identity': 'Shared exact source canvas from the frozen renderer training package',
              'public_task_schema_validation': 'passed', 'output_bytes': output.stat().st_size}
    output.with_suffix('.report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
