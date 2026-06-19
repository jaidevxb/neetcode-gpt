from typing import List, Dict

class Solution:
    def _greedy_tokenize(self, text: str, vocab: Dict[str, int]) -> List[str]:
        tokens = []
        i = 0
        n = len(text)
        max_len = max((len(tok) for tok in vocab), default=1)

        while i < n:
            matched = None
            # Try longest possible substring first, down to length 1
            for length in range(min(max_len, n - i), 0, -1):
                candidate = text[i:i + length]
                if candidate in vocab:
                    matched = candidate
                    break
            if matched is not None:
                tokens.append(matched)
                i += len(matched)
            else:
                # No match found, consume a single character
                tokens.append(text[i])
                i += 1

        return tokens

    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        return [self._greedy_tokenize(str(num), vocab) for num in numbers]

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        return len(self._greedy_tokenize(text, vocab))

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        words = text.split(" ")
        words = [w for w in words if w != ""]
        word_count = len(words)
        token_count = self.count_tokens(text, vocab)
        if word_count == 0:
            return 0.0
        return round(token_count / word_count, 4)
