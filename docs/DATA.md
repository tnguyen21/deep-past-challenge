# Dataset Documentation

## Download

```bash
kaggle competitions download -c deep-past-initiative-machine-translation
```

## Overview

The competition data comprises transliterations of over 8,000 Old Assyrian cuneiform texts with comprehensive metadata. Aligned English translations are provided for a subset. Additionally, unprocessed texts from ~900 scholarly publications are available for creating additional training data.

**Note**: This is a Code Competition. The `test.csv` contains only dummy data. The full test set is used during scoring.

---

## Core Files

### train.csv

~1,500 transliterations with English translations.

| Field | Description |
|-------|-------------|
| `oare_id` | Identifier in the OARE database (unique per text) |
| `transliteration` | Akkadian transliteration of the tablet |
| `translation` | Corresponding English translation |

### test.csv

~4,000 sentences from ~400 unique documents (example data; replaced during scoring).

| Field | Description |
|-------|-------------|
| `id` | Unique identifier for each sentence |
| `text_id` | Unique identifier for each document |
| `line_start`, `line_end` | Sentence boundaries within the tablet (str type: `1`, `1'`, `1''`) |
| `transliteration` | Akkadian transliteration to translate |

### sample_submission.csv

Example submission in the correct format.

---

## Supplemental Data

### published_texts.csv

~8,000 transliterations with metadata (no translations).

| Field | Description |
|-------|-------------|
| `oare_id` | OARE database identifier |
| `online transcript` | URL of transliteration on DPI website |
| `cdli_id` | CDLI identifier(s), separated by `\|` |
| `aliases` | Other published labels (publication numbers, museum IDs) |
| `label` | Primary designation |
| `description` | Basic text description |
| `genre_label` | Genre (not available for all) |
| `transliteration_orig` | Original from OARE |
| `transliteration` | Cleaned version |

### publications.csv

OCR output from ~880 scholarly PDFs containing translations (often in languages other than English).

| Field | Description |
|-------|-------------|
| `pdf_name` | Source PDF filename |
| `page` | Page number |
| `page_text` | Extracted text |
| `has_akkadian` | Whether text contains Akkadian |

### bibliography.csv

Bibliographic data for `publications.csv`.

| Field | Description |
|-------|-------------|
| `pdf_name` | ID matching `publications.csv` |
| `title`, `author`, `journal`, `volume`, `year`, `pages` | Standard bibliographic fields |

### OA_Lexicon_eBL.csv

Old Assyrian word list with lexical equivalents.

| Field | Description |
|-------|-------------|
| `type` | Word type (word, PN=person name, GN=geographic name) |
| `form` | String-literal word as in transliteration |
| `norm` | Normalized form (hyphens removed, vowel length indicated) |
| `lexeme` | Lemmatized dictionary form |
| `eBL` | URL to electronic Babylonian Library |

### eBL_Dictionary.csv

Complete Akkadian dictionary from the eBL database.

### resources.csv

List of resources for additional data.

### Sentences_Oare_FirstWord_LinNum.csv

Aid for sentence-level alignment of `train.csv` data.

---

## Building Additional Training Data

### Suggested Workflow

1. **Locate texts and translations**: Match transliterations with translations in OCR output using document IDs, aliases, or museum numbers.

2. **Convert to English**: Source translations may be in French, German, Turkish, etc. Convert all to English for consistency.

3. **Create sentence-level alignments**: Break both Akkadian transliteration and English translation into sentences and align pairwise.

---

## Bibliography

The bibliography reflects secondary sources used for translations. Each work should be cited if used when generating machine translations.

Additional primary source citations:
- https://cdli.earth/publications
- https://cdli.ox.ac.uk/wiki/abbreviations_for_assyriology
