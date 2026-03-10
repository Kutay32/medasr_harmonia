import re
import jiwer
import Levenshtein


def normalize(s: str) -> str:
    # Safely convert to lowercase considering both English and Turkish.
    # The default str.lower() converts English 'I' to 'i'.
    # If we want to strictly handle Turkish we must be careful not to break English text.
    # Since the MedASR output is typically English, converting 'I' to 'ı' breaks words like "Indication" -> "ındıcatıon".
    # We will remove the hardcoded I -> ı replacement and use standard lowercasing, 
    # but handle the Turkish 'İ' to 'i' explicitly just in case.
    s = s.replace('İ', 'i')
    s = s.lower()
    s = s.replace('</s>', '')
    # Allow Turkish characters: ç, ğ, ı, ö, ş, ü
    s = re.sub(r"[^ a-z0-9'çğıöşü]", ' ', s)
    s = ' '.join(s.split())
    return s


def _colored(text, color):
    if color == 'red':
        return f"\033[91m{text}\033[0m"
    elif color == 'green':
        return f"\033[92m{text}\033[0m"
    return text


def evaluate(
    ref_text: str,
    hyp_text: str,
    delete_color: str = 'red',
    insert_color: str = 'green',
) -> None:
    print('HYP:', hyp_text)
    normalized_ref = normalize(ref_text)
    normalized_hyp = normalize(hyp_text)

    ref_words = normalized_ref.split()
    hyp_words = normalized_hyp.split()

    measures = jiwer.process_words([normalized_ref], [normalized_hyp])

    edits = Levenshtein.editops(ref_words, hyp_words)

    r = 0
    diff = ''

    for op, i, j in edits:
        if r < i:
            diff += ' ' + ' '.join(ref_words[r:i])
        r = i

        if op == 'replace':
            diff += (
                f' {_colored(f"{{-{ref_words[i]}-}}", delete_color)}'
                f' {_colored(f"{{+{hyp_words[j]}+}}", insert_color)}'
            )
            r += 1
        elif op == 'insert':
            diff += f' {_colored(f"{{+{hyp_words[j]}+}}", insert_color)}'
        elif op == 'delete':
            diff += f' {_colored(f"{{-{ref_words[i]}-}}", delete_color)}'
            r += 1

    if r < len(ref_words):
        diff += ' ' + ' '.join(ref_words[r:])

    print(
        f'WER: {measures.wer * 100:.2f}%: '
        f'insertions {measures.insertions}, deletions {measures.deletions}, '
        f'substitutions {measures.substitutions}, '
        f'ref tokens {len(ref_words)}'
    )
    print(diff)
