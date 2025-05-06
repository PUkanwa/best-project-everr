import json
import re

INPUT_FILE = "jobs-new.jsonl"
OUTPUT_FILE = "jobs-new_with_offsets.jsonl"

def annotate_offsets(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            data = json.loads(line)
            text = data.get('text', '')
            entities = data.get('entities', [])
            annotated = []

            for ent in entities:
                span = ent['text']
                # find all occurrences of span
                offsets = [(m.start(), m.end()) for m in re.finditer(re.escape(span), text)]
                if not offsets:
                    print(f"Warning: '{span}' not found in text ID {data.get('id')}.")
                    continue

                # pick the first offset that doesn't overlap previously annotated spans
                used = [(a['start'], a['end']) for a in annotated]
                for start, end in offsets:
                    if not any(s < end and start < e for s,e in used):
                        annotated.append({
                            'start': start,
                            'end':   end,
                            'label': ent.get('label', ''),
                            'text':  span
                        })
                        break
                else:
                    print(f"Warning: no non-overlapping occurrence for '{span}' in ID {data.get('id')}.")

            # sort by start offset and replace entities
            annotated.sort(key=lambda x: x['start'])
            data['entities'] = annotated

            fout.write(json.dumps(data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    annotate_offsets(INPUT_FILE, OUTPUT_FILE)
    print(f"🔍 Finished annotating. Output → {OUTPUT_FILE}")
