# Deep Past Challenge - Akkadian Translation

ByT5-based Akkadian-to-English translation for the Kaggle Deep Past Initiative competition.

## Setup

```bash
# Create venv and install dependencies
uv sync

# Activate the environment
source .venv/bin/activate

# Configure Kaggle API credentials (place kaggle.json in ~/.kaggle/)
# Then download competition data
kaggle competitions download -c deep-past-initiative-machine-translation
```

## Usage

```bash
# Full inference (Kaggle paths)
python byt_ensemble.py

# Dry run - downloads byt5-small, uses synthetic data (good for testing pipeline)
python byt_ensemble.py --dry-run

# Local run with custom model and data
python byt_ensemble.py --model /path/to/model --test-data data/test.csv --output-dir ./output

# Scaled down for quick testing (fewer beams, smaller batch)
python byt_ensemble.py --dry-run --batch-size 1 --num-beams 2 --num-workers 0
```

## Development

```bash
# Lint
uvx ruff check .

# Format
uvx ruff format .
```

---

## Competition Overview

The Deep Past Challenge asks: Can AI decode 4,000-year-old business records?

Four thousand years ago, Assyrian merchants left behind one of the world's richest archives of everyday and commercial life. Tens of thousands of clay tablets record debts settled, caravans dispatched, and day-to-day family matters. Today, half of these tablets remain silent—not because they're damaged, but because so few people can read the language pressed into their clay.

Nearly 23,000 tablets survive documenting the Old Assyrian trading networks that connected Mesopotamia to Anatolia. Only half have been translated, and fewer than a dozen scholars in the world are specialized to read the rest.

**Task**: Build neural machine-translation models that convert transliterated Akkadian into English.

**Challenge**: Akkadian is a low-resource, morphologically complex language where a single word can encode what takes multiple words in English.

### Evaluation

Submissions are evaluated by the **Geometric Mean of BLEU and chrF++ scores**, with each score's sufficient statistics aggregated across the entire corpus (micro-average).

See the [SacreBLEU library](https://github.com/mjpost/sacrebleu) for implementation details.

### Submission Format

```csv
id,translation
0,Thus Kanesh, say to the -payers, our messenger, every single colony, and the...
1,In the letter of the City (it is written): From this day on, whoever buys meteoric...
2,As soon as you have heard our letter, who(ever) over there has either sold it to...
```

### Timeline

| Date | Milestone |
|------|-----------|
| Dec 16, 2025 | Start Date |
| Mar 16, 2026 | Entry & Team Merger Deadline |
| Mar 23, 2026 | Final Submission Deadline |

All deadlines are at 11:59 PM UTC.

### Prizes

| Place | Prize |
|-------|-------|
| 1st | $15,000 |
| 2nd | $10,000 |
| 3rd | $8,000 |
| 4th | $7,000 |
| 5th | $5,000 |
| 6th | $5,000 |

### Code Requirements

- CPU/GPU Notebook ≤ 9 hours run-time
- Internet access disabled
- Freely & publicly available external data allowed (including pre-trained models)
- Submission file must be named `submission.csv`

---

## Dataset Notes

See [DATA.md](DATA.md) for detailed dataset documentation.

### Key Formatting Considerations

**Transliteration challenges:**
- Hyphenated syllables with superscripts/subscripts
- Capitalization encodes meaning (proper nouns vs Sumerian logograms)
- Determinatives in curly brackets: `{ki}`, `{d}`, etc.
- Broken text markers: `<gap>` (single sign) and `<big_gap>` (multiple signs)

**Preprocessing suggestions:**
- Remove: `!`, `?`, `/`, `:`, `.`, `˹ ˺`, `[ ]`
- Replace: `[x]` → `<gap>`, `…` → `<big_gap>`
- Normalize: `Ḫ ḫ` → `H h`

### Character Reference

| Character | CDLI | ORACC | Unicode |
|-----------|------|-------|---------|
| á | a2 | a₂ | — |
| š | sz | š | U+0161 |
| ṣ | s, | ṣ | U+1E63 |
| ṭ | t, | ṭ | U+1E6D |
| ḫ | h | h | U+1E2B |
| ₀-₉ | 0-9 | subscript | U+2080-U+2089 |

### Determinatives

| Code | Meaning | Usage |
|------|---------|-------|
| `{d}` | dingir (god/deity) | Precedes divine names |
| `{ki}` | earth | Follows place names |
| `{m}` | masculine | Precedes male names |
| `{mi}` | feminine | Precedes female names |
| `{uru}` | city | Precedes settlement names |

---

## Citation

```
Abdulla, F., Agarwal, R., Anderson, A., Barjamovic, G., Lassen, A.,
Ryan Holbrook, and María Cruz. Deep Past Challenge - Translate Akkadian to English.
https://kaggle.com/competitions/deep-past-initiative-machine-translation, 2025. Kaggle.
```
