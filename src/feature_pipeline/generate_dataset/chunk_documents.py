import re

from core import get_logger

logger = get_logger(__name__)


def chunk_documents(
    documents: list[str], min_length: int = 1000, max_length: int = 2000
):
    chunked_documents = []
    for document in documents:
        chunks = extract_substrings(
            document, min_length=min_length, max_length=max_length
        )
        chunked_documents.extend(chunks)

    return chunked_documents


def extract_substrings(
    text: str, min_length: int = 1000, max_length: int = 2000
) -> list[str]:
    # Pattern to find sentence endings: punctuation (.!?) optionally followed by bracketed year.
    # Negative lookbehinds ensure we don't split abbreviations like Dr. or e.g.
    # Uses re.VERBOSE for readability.
    sentence_end_pattern = re.compile(
        r"""
        (?<!\w\.\w.)       # Negative lookbehind: not preceded by word.word.
        (?<![A-Z][a-z]\.)  # Negative lookbehind: not preceded by Abbrev.
        [.?!]              # Sentence ending punctuation
        (?:                # Optional non-capturing group for bracketed year
            \s*            # Optional whitespace
            \[             # Opening bracket
            \s*\d{4}\s*    # Year (YYYY) with optional surrounding whitespace
            (?:            # Optional non-capturing group for amendment
                ,\s*amended\s*\d{4}\s* # ", amended YYYY" with optional whitespace
            )?             # End amendment group (optional)
            \]             # Closing bracket
        )?                 # End bracketed year group (optional)
        """,
        re.VERBOSE | re.UNICODE,
    )

    sentences = []
    start_index = 0
    # Find all sentence ending points using the pattern
    for match in sentence_end_pattern.finditer(text):
        end_index = match.end()
        sentence = text[start_index:end_index].strip()
        if sentence:
            sentences.append(sentence)
        start_index = end_index

    # Add the last part of the text if any remains after the last match
    if start_index < len(text):
        sentence = text[start_index:].strip()
        if sentence:
            sentences.append(sentence)

    # --- Chunking Logic ---
    extracts = []
    current_chunk = ""
    current_chunk_len = 0  # Keep track of length to avoid repeated len() calls

    for sentence in sentences:
        sentence_len = len(sentence)

        # Case 1: Single sentence exceeds max_length
        if sentence_len > max_length:
            # Add the current chunk if it's valid
            if current_chunk_len >= min_length:
                extracts.append(current_chunk)

            # Add the oversized sentence as its own chunk if it meets min length,
            # otherwise log a warning (or could implement splitting logic here).
            if sentence_len >= min_length:
                extracts.append(sentence)
            else:
                logger.warning(
                    f"Sentence longer than max_length ({sentence_len} > {max_length}) but shorter than min_length. Discarding: '{sentence[:100]}...'"
                )

            # Reset current chunk
            current_chunk = ""
            current_chunk_len = 0
            continue

        # Case 2: Adding sentence would exceed max_length
        # Calculate potential length (+1 for the space)
        potential_len = (
            current_chunk_len + sentence_len + (1 if current_chunk_len > 0 else 0)
        )

        if potential_len > max_length:
            # Finalize the current chunk if it's valid
            if current_chunk_len >= min_length:
                extracts.append(current_chunk)

            # Start a new chunk with the current sentence
            current_chunk = sentence
            current_chunk_len = sentence_len
        # Case 3: Adding sentence fits within max_length
        else:
            if current_chunk_len > 0:
                current_chunk += " " + sentence
                current_chunk_len = potential_len
            else:
                current_chunk = sentence
                current_chunk_len = sentence_len

    # Add the last remaining chunk if it's valid
    if current_chunk_len >= min_length:
        extracts.append(current_chunk)

    return extracts
