import argparse
import heapq
import pickle
from pathlib import Path

import numpy as np
import regex as re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from multiprocessing import Pool
from tqdm import tqdm

from cs336_basics.filechunks import get_file_chunks

type TokenId = int
type TokenText = bytes
type TokenPair = tuple[TokenId, TokenId]
type TokenHeapKey = tuple[int, ...]
type TokenPairHeapItem = tuple[int, TokenHeapKey, TokenHeapKey, TokenPair]

PRE_TOKENIZER_SPLIT = rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PRE_TOKENIZER_SPLIT_RE = re.compile(PRE_TOKENIZER_SPLIT)
EOT_TOKEN = "<|endoftext|>"
MAX_VOCAB = 10_000


class Tokenizer:
    def __init__(
        self,
        *,
        vocab: dict[int, bytes] | None = None,
        merges: list[tuple[bytes, bytes]] | None = None,
        special_tokens: list[str] | None = None,
    ):
        ## Initial vocabulary, all single bytes
        if vocab is None:
            print("Initializing vocab with single byte tokens.")
            vocab = {i: bytes([i]) for i in range(256)}
            vocab[len(vocab)] = EOT_TOKEN.encode()
        # Special tokens added at the end.
        self.special_tokens = [EOT_TOKEN] if special_tokens is None else special_tokens
        existing_tokens = set(vocab.values())
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in existing_tokens:
                vocab[len(vocab)] = token_bytes
                existing_tokens.add(token_bytes)
        self.special_tokens = sorted(
            self.special_tokens,
            key=len,
            reverse=True,
        )
        self.id_to_text: dict[TokenId, TokenText] = vocab
        self.special_token_bytes_set = set(token.encode() for token in self.special_tokens)
        self.special_re = re.compile(b"|".join(re.escape(token.encode()) for token in self.special_tokens))

        ## Token encoding dictionary: text (bytes) -> TokenId
        self.text_to_id: dict[TokenText, TokenId] = {text: tokenId for (tokenId, text) in self.id_to_text.items()}
        self.token_id_to_heap_key: dict[TokenId, TokenHeapKey] = {
            token_id: self._token_heap_key(token_text) for token_id, token_text in self.id_to_text.items()
        }

        ## Merge Tree : Keeps track of which pair of tokens got merged.
        ## Tokenization needs to be done in the same order.
        self.merged_pair_to_id: dict[TokenPair, TokenId] = {}
        self.merged_id_to_pair: dict[TokenId, TokenPair] = {}

        # If merges is provided, we can initialize the above merge tree.
        self.merges = merges or []
        if merges is not None:
            for t1, t2 in merges:
                token1_id = self.text_to_id[t1]
                token2_id = self.text_to_id[t2]
                token_merged_id = self.text_to_id[t1 + t2]
                token_pair: TokenPair = (token1_id, token2_id)
                self.merged_pair_to_id[token_pair] = token_merged_id
                self.merged_id_to_pair[token_merged_id] = token_pair

        ## Precomputed tokenization of all the words.
        ## Will help tokenize entire document after training.
        # It is built during training, but this map will be useful later
        # too so we will make this a property of self.
        self.word_to_tokens: dict[bytes, list[TokenId]] = {}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    load_state = from_files

    def save_state(self, vocab_filepath: str, merges_filepath: str):
        vocab_path = Path(vocab_filepath)
        merges_path = Path(merges_filepath)
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        merges_path.parent.mkdir(parents=True, exist_ok=True)

        with open(vocab_path, "wb") as f:
            pickle.dump(self.id_to_text, f)
        with open(merges_path, "wb") as f:
            pickle.dump(self.merges, f)

    to_files = save_state

    def _create_new_token(self, tokenText: TokenText) -> TokenId:
        tokenId = len(self.id_to_text)  # This is automatically the max existing id + 1
        self.id_to_text[tokenId] = tokenText
        self.text_to_id[tokenText] = tokenId
        self.token_id_to_heap_key[tokenId] = self._token_heap_key(tokenText)
        return tokenId

    def _merge_tokens(self, token1: TokenId, token2: TokenId) -> TokenId:
        # print(f"Merging tokens {self._token(token1)} and {self._token(token2)}")
        text1 = self.id_to_text[token1]
        text2 = self.id_to_text[token2]
        new_text = text1 + text2
        new_id = self._create_new_token(new_text)
        token_pair = (token1, token2)
        self.merges.append((text1, text2))
        self.merged_pair_to_id[token_pair] = new_id
        self.merged_id_to_pair[new_id] = token_pair
        return new_id

    def _check_token_pair(self, word, token_pair):
        tokens = self.word_to_tokens[word]
        for X, Y in zip(tokens, tokens[1:]):
            if (X, Y) == token_pair:
                return True
        return False

    def _token_heap_key(self, token_text: bytes) -> tuple[int, ...]:
        # This reverses bytes ordering so heapq chooses the lexicographically
        # largest token text when frequencies are tied.
        return tuple(255 - byte for byte in token_text) + (256,)

    def _pair_heap_item(self, token_pair: TokenPair, freq: int) -> TokenPairHeapItem:
        t1, t2 = token_pair
        return (
            -freq,
            self.token_id_to_heap_key[t1],
            self.token_id_to_heap_key[t2],
            token_pair,
        )

    def train_tokenizer(self, word_frequencies: Counter[bytes], max_merges):
        """BPE Tokenizer training.

        Core idea: we will find the token pair with highest frequency and merge
        the pair into one token.
        """
        ## For efficient tokenization we pre-split the text into words.
        # This is recommended by the assignment. They also make sure EOT token
        # does not appear in the pre-tokenized words.
        # Core idea requires us to have a loop which merges tokens with highest
        # frequency.

        # We would need to keep a map of words -> tokens (to count freqs)
        # and of tokens -> words (to perform updates).

        # It is best to compute the map of words -> tokens first and not each
        # time, otherwise we would be O(N_WORDS) at every merge step.
        self.word_to_tokens = {word: self.tokenize(word) for word in word_frequencies}

        # Index tracking a token appears in which words.
        # The tokens -> words map would help, after merging, to know which words
        # in the first map need to be updated. This is initialized first by
        # mapping the words in word_to_tokens map.
        token_to_words: dict[TokenId, set[bytes]] = defaultdict(set)
        for word, tokens in self.word_to_tokens.items():
            for token in tokens:
                token_to_words[token].add(word)

        # Step 1. From word frequencies, determine initial pair frequencies.
        token_pair_frequencies: dict[TokenPair, int] = defaultdict(int)
        for word, freq in word_frequencies.items():
            tokens = self.word_to_tokens[word]
            for t1, t2 in zip(tokens, tokens[1:]):
                token_pair_frequencies[(t1, t2)] += freq

        # Priority queue for selecting the most frequent token pair.
        # Python has a min-heap, so we negate the frequency and reverse the
        # lexicographic bytes ordering used as the tie-breaker.
        token_pair_heap = []
        for token_pair, freq in token_pair_frequencies.items():
            heapq.heappush(token_pair_heap, self._pair_heap_item(token_pair, freq))

        for i in tqdm(range(max_merges), "BPE Merges"):
            # Step 2. Determine most frequent token pair.
            # Ensure lexicographical order in case of ties.
            while token_pair_heap:
                neg_freq, _, _, most_common_token_pair = heapq.heappop(token_pair_heap)
                freq = -neg_freq
                if token_pair_frequencies.get(most_common_token_pair, 0) == freq:
                    break
            else:
                break

            if freq == 0:
                # no more pairs to merge, all words have been assigned
                # unique tokens
                break
            t1, t2 = most_common_token_pair

            # Step 3. Merge largest pair and define new token id.
            new_token_id = self._merge_tokens(t1, t2)

            # Step 4. Update self.word_to_tokens for the words which contain the token pair.
            # How will we find the words that might contain the token pair? Using token_to_words.
            words_to_update = token_to_words[t1].intersection(token_to_words[t2])
            words_to_update = {word for word in words_to_update if self._check_token_pair(word, most_common_token_pair)}

            pair_frequency_deltas: dict[TokenPair, int] = defaultdict(int)
            for word in words_to_update:
                old_tokenization = self.word_to_tokens[word]
                new_tokenization = self.update_tokenization(old_tokenization, new_token_id)
                self.word_to_tokens[word] = new_tokenization

                # Step 5. Update token pair frequencies according to new tokenization
                # of this word.
                freq = word_frequencies[word]
                for X, Y in zip(old_tokenization, old_tokenization[1:]):
                    token_pair = (X, Y)
                    pair_frequency_deltas[token_pair] -= freq
                for X, Y in zip(new_tokenization, new_tokenization[1:]):
                    token_pair = (X, Y)
                    pair_frequency_deltas[token_pair] += freq

            for token_pair, freq_delta in pair_frequency_deltas.items():
                updated_freq = token_pair_frequencies[token_pair] + freq_delta
                if updated_freq <= 0:
                    token_pair_frequencies.pop(token_pair, None)
                else:
                    token_pair_frequencies[token_pair] = updated_freq
                    heapq.heappush(token_pair_heap, self._pair_heap_item(token_pair, updated_freq))

            # Step 6. Update token_to_words for the old tokens.
            # The old tokens *might have disappeared* from the word,
            # But not necessarily.
            # Also we need to check if `word` exists in the set before removing
            # because it could be in the updatable words due to the other token.
            for word in words_to_update:
                if t1 not in self.word_to_tokens[word]:
                    if word in token_to_words[t1]:
                        token_to_words[t1].remove(word)
                if t2 not in self.word_to_tokens[word]:
                    if word in token_to_words[t2]:
                        token_to_words[t2].remove(word)

            # Step 7. We also need to remember to update token_to_words for the
            # NEW TOKEN.
            for word in words_to_update:
                if new_token_id in self.word_to_tokens[word]:
                    token_to_words[new_token_id].add(word)

    def update_tokenization(self, old_tokens: list[TokenId], new_token_id: TokenId) -> list[TokenId]:
        old_token_pair = self.merged_id_to_pair[new_token_id]
        new_tokenization = []
        i = 0
        while i < len(old_tokens):
            t1, t2 = old_tokens[i], old_tokens[i + 1] if i + 1 < len(old_tokens) else None
            if (t1, t2) == old_token_pair:
                new_tokenization.append(new_token_id)
                i += 2
            else:
                new_tokenization.append(t1)
                i += 1
        return new_tokenization

    def tokenize(self, word: bytes) -> list[TokenId]:
        # Basically we will merge the bytes in the correct order (increasing ids)
        # NOTE: INEFFICIENT. Make this faster.
        # assert len(word) > 0
        word_bytes = [bytes([b]) for b in word]
        tokens: list[TokenId] = [self.text_to_id[byte] for byte in word_bytes]
        # Python 3.7+ ensures this iteration happens in insertion order
        # Even if it didn't, we could replace this by a for..range loop
        for merged_token_id, token_pair in self.merged_id_to_pair.items():
            # we need to replace token_pair if it occurs in `tokens`
            new_tokens = []
            i = 0
            while i < len(tokens):
                if tokens[i] == token_pair[0] and i + 1 < len(tokens) and tokens[i + 1] == token_pair[1]:
                    new_tokens.append(merged_token_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def repr_token(self, token: TokenId):
        r = repr(self.id_to_text[token])[2:-1]
        return f"<{r}>"

    def encode(self, text: str) -> list[TokenId]:
        # print(f"Special tokens: {self.special_tokens}")
        # print(f"Encoding text: {text[:50]}")
        # first we need to split text into words then tokenize each word
        tokens = []
        words = self.pre_tokenize(text.encode("utf-8"))
        words = list(words)
        for word in words:
            if word in self.special_token_bytes_set:
                # print("special token: ", word)
                tokens.append(self.text_to_id[word])
            else:
                # optimization: memoize
                # if len(words) < 50:
                # print("word: ", word)
                if word not in self.word_to_tokens:
                    self.word_to_tokens[word] = self.tokenize(word)
                tokens += self.word_to_tokens[word]
        return tokens

    def decode(self, tokens: list[TokenId]) -> str:
        text = b""
        for token in tokens:
            text += self.id_to_text[token]
        return text.decode("utf-8", errors="replace")

    def pre_tokenize(self, raw_bytes: bytes, spl_tokens: bool = True) -> Iterator[bytes]:
        # Ideally compile/cache this once in __init__ if special_tokens is fixed.
        if len(self.special_tokens) == 0:
            for word_match in PRE_TOKENIZER_SPLIT_RE.finditer(raw_bytes):
                yield word_match.group()
            return
        pos = 0
        for special_match in self.special_re.finditer(raw_bytes):
            start, end = special_match.span()
            # Tokenize normal text before the special token, without slicing.
            for word_match in PRE_TOKENIZER_SPLIT_RE.finditer(raw_bytes, pos, start):
                yield word_match.group()
            # Yield the special token itself if requested.
            if spl_tokens:
                yield special_match.group()
            pos = end
        # Tokenize the remaining normal text after the last special token.
        for word_match in PRE_TOKENIZER_SPLIT_RE.finditer(raw_bytes, pos, len(raw_bytes)):
            yield word_match.group()

    def encode_iterable(self, text: Iterable[str]) -> Iterable[TokenId]:
        for line in text:
            yield from self.encode(line)

    def encode_file_to_numpy(self, input_file: str, output_file: str, dtype=None):
        if dtype is None:
            max_id = max(self.id_to_text)
            dtype = np.uint16 if max_id < 2**16 else np.uint32
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Show progress while encoding lines
        with open(input_file, encoding="utf-8") as f:
            token_ids = np.fromiter(self.encode_iterable(tqdm(f, desc="Encoding", unit="lines")), dtype=dtype)
        np.save(output_path, token_ids)
        return token_ids

    def count_words(self, raw_bytes: bytes) -> Counter[bytes]:
        return Counter(self.pre_tokenize(raw_bytes, spl_tokens=False))

    def train_on_file(self, input_file: str, max_vocab: int = MAX_VOCAB):
        cache_path = Path("out") / Path(input_file).with_suffix(".word_counts.pkl").name

        if cache_path.exists():
            word_frequencies = pickle.load(open(cache_path, "rb"))
        else:
            print("Reading file and running pre-tokenization")
            mp_pool = Pool()
            word_frequencies: Counter[bytes] = Counter()
            chunk_gen = get_file_chunks(input_file)
            for partial_counts in mp_pool.imap_unordered(self.count_words, chunk_gen):
                word_frequencies.update(partial_counts)
            pickle.dump(word_frequencies, open(cache_path, "wb"))

        assert EOT_TOKEN not in word_frequencies
        print(f"Pre-tokenization completed. {len(word_frequencies)} unique words found. Applying merges.")
        self.train_tokenizer(word_frequencies, max_merges=max_vocab - len(self.id_to_text))


def main():
    parser = argparse.ArgumentParser(description="Tokenizer script")
    parser.add_argument("--input_file", default="data/owt_train.txt", help="Input file path (.txt)")
    parser.add_argument("--prefix", default="owt", help="Prefix for output tokenizer files")
    parser.add_argument("--output_file", default=None, help="Output file path (.npy)")
    parser.add_argument("--vocab_file", default=None, help="Output vocab path (.pkl)")
    parser.add_argument("--merges_file", default=None, help="Output merges path (.pkl)")
    parser.add_argument("--max_vocab", default=MAX_VOCAB, type=int, help="Maximum tokenizer vocabulary size")
    args = parser.parse_args()

    output_file = args.output_file or f"out/{args.prefix}_train.npy"
    vocab_file = args.vocab_file or f"out/{args.prefix}_tokenizer_vocab.pkl"
    merges_file = args.merges_file or f"out/{args.prefix}_tokenizer_merges.pkl"

    t = Tokenizer()
    t.train_on_file(args.input_file, max_vocab=args.max_vocab)
    t.save_state(vocab_file, merges_file)
    token_ids = t.encode_file_to_numpy(args.input_file, output_file)
    print("Final token vocabulary: ")
    for id in t.id_to_text:
        print(f"{id:3d} : {t.repr_token(id)}")

    print(f"Serialized tokenizer to {vocab_file} and {merges_file}.")
    print(f"Wrote {len(token_ids)} token ids to {output_file}.")
    print("Tokenization completed successfully.")


if __name__ == "__main__":
    main()
