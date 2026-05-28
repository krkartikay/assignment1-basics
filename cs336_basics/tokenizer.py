import argparse
import regex as re
from collections import Counter, defaultdict
from collections.abc import Iterable

type TokenId = int
type TokenText = bytes
type TokenPair = tuple[TokenId, TokenId]

PRE_TOKENIZER_SPLIT = rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
EOT_TOKEN = "<|endoftext|>"
MAX_MERGES = 3500


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
        self.id_to_text: dict[TokenId, TokenText] = vocab

        # Special tokens added at the end.
        self.special_tokens = special_tokens or [EOT_TOKEN]
        self.special_token_bytes_set = set(token.encode() for token in self.special_tokens)

        ## Token encoding dictionary: text (bytes) -> TokenId
        self.text_to_id: dict[TokenText, TokenId] = {text: tokenId for (tokenId, text) in self.id_to_text.items()}

        ## Merge Tree : Keeps track of which pair of tokens got merged.
        ## Tokenization needs to be done in the same order.
        self.merged_pair_to_id: dict[TokenPair, TokenId] = {}
        self.merged_id_to_pair: dict[TokenId, TokenPair] = {}

        # If merges is provided, we can initialize the above merge tree.
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

    def _create_new_token(self, tokenText: TokenText) -> TokenId:
        tokenId = len(self.id_to_text)  # This is automatically the max existing id + 1
        self.id_to_text[tokenId] = tokenText
        self.text_to_id[tokenText] = tokenId
        return tokenId

    def _merge_tokens(self, token1: TokenId, token2: TokenId) -> TokenId:
        # print(f"Merging tokens {self._token(token1)} and {self._token(token2)}")
        text1 = self.id_to_text[token1]
        text2 = self.id_to_text[token2]
        new_text = text1 + text2
        new_id = self._create_new_token(new_text)
        token_pair = (token1, token2)
        self.merged_pair_to_id[token_pair] = new_id
        self.merged_id_to_pair[new_id] = token_pair
        return new_id

    def train_tokenizer(self, text: bytes):
        """BPE Tokenizer training.

        Core idea: we will find the token pair with highest frequency and merge
        the pair into one token.
        """
        ## For efficient tokenization we pre-split the text into words.
        # This is recommended by the assignment. They also make sure EOT token
        # does not appear in the pre-tokenized words.
        word_frequencies = self.count_words(text)
        assert EOT_TOKEN not in word_frequencies

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
        token_pairs_counter = Counter()
        for word, freq in word_frequencies.items():
            tokens = self.word_to_tokens[word]
            for t1, t2 in zip(tokens, tokens[1:]):
                token_pairs_counter[(t1, t2)] += freq

        for i in range(MAX_MERGES):
            # Step 2. Determine most frequent token pair.
            # TODO: ensure lexicographical order in case of ties.
            most_common_token_pair, freq = token_pairs_counter.most_common(n=1)[0]
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
            for word in words_to_update:
                old_tokenization = self.word_to_tokens[word]
                new_tokenization = self.tokenize(word)
                self.word_to_tokens[word] = new_tokenization

                # Step 5. Update token pair frequencies according to new tokenization
                # of this word.
                freq = word_frequencies[word]
                for X, Y in zip(old_tokenization, old_tokenization[1:]):
                    token_pairs_counter[(X, Y)] -= freq
                for X, Y in zip(new_tokenization, new_tokenization[1:]):
                    token_pairs_counter[(X, Y)] += freq

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

    def save(self, filename: str):
        pass

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

    def pre_tokenize(self, raw_bytes: bytes) -> Iterable[bytes]:
        spl_tokens_regex = b"|".join(
            f"({re.escape(token)})".encode()
            for token in sorted(self.special_tokens, key=lambda x: len(x), reverse=True)
        )
        documents: list[bytes] = re.split(spl_tokens_regex, raw_bytes)
        words_in_all_docs = []
        for doc in documents:
            if not doc:
                continue
            if re.match(spl_tokens_regex, doc):
                # This document is actually a special token, we can directly add it to the list of words.
                words_in_all_docs.append(doc)
                continue
            words = (match.group() for match in re.finditer(PRE_TOKENIZER_SPLIT, doc))
            words_in_all_docs += words
        return words_in_all_docs

    def encode_iterable(self, text: Iterable[str]) -> Iterable[TokenId]:
        for line in text:
            yield from self.encode(line)

    def count_words(self, raw_bytes: bytes) -> Counter[bytes]:
        return Counter(self.pre_tokenize(raw_bytes))


def run_tokenizer(input_file, output_file):
    # ok, we will optimize the chunking later
    raw_bytes = open(input_file, "rb").read()
    t = Tokenizer()
    t.train_tokenizer(raw_bytes)
    print("Final token vocabulary: ")
    for id, text in t.id_to_text.items():
        print(f"{id:3d} : {t.repr_token(id)}")
    t.save(output_file)


def main():
    parser = argparse.ArgumentParser(description="Tokenizer script")
    parser.add_argument("--input_file", default="data/owt_train.txt", help="Input file path (.txt)")
    parser.add_argument("--output_file", default="out/owt_train.bin", help="Output file path (.bin)")
    args = parser.parse_args()

    run_tokenizer(args.input_file, args.output_file)

    print("Tokenization completed successfully.")


if __name__ == "__main__":
    main()
