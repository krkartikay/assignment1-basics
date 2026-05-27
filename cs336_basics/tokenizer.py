import argparse
import regex as re
from collections import Counter, defaultdict

type TokenId = int
type TokenText = bytes
type TokenPair = tuple[TokenId, TokenId]

PRE_TOKENIZER_SPLIT = rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
EOT_TOKEN = b"<|endoftext|>"
MAX_MERGES = 1000


class Tokenizer:
    def __init__(self):
        ## Initial vocabulary, all single bytes + spl token
        self.id_to_text: dict[TokenId, TokenText] = {i: bytes([i]) for i in range(256)}
        self.id_to_text[256] = EOT_TOKEN  # Special token

        ## Token encoding dictionary: text (bytes) -> TokenId
        self.text_to_id: dict[TokenText, TokenId] = {text: tokenId for (tokenId, text) in self.id_to_text.items()}

        ## Merge Tree : Keeps track of which pair of tokens got merged.
        ## Tokenization needs to be done in the same order.
        self.merged_pair_to_id: dict[TokenPair, TokenId] = {}
        self.merged_id_to_pair: dict[TokenId, TokenPair] = {}

        ## Precomputed tokenization of all the words.
        ## Will help tokenize entire document after training.
        # It is built during training, but this map will be useful later
        # too so we will make this a property of self.
        self.word_to_tokens: dict[bytes, list[TokenId]] = {}

    def _create_new_token(self, tokenText: TokenText) -> TokenId:
        tokenId = len(self.id_to_text)  # This is automatically the max existing id + 1
        assert tokenId == max(id for id in self.id_to_text) + 1
        self.id_to_text[tokenId] = tokenText
        self.text_to_id[tokenText] = tokenId
        return tokenId

    def _merge_tokens(self, token1: TokenId, token2: TokenId) -> TokenId:
        print(f"Merging tokens {self._token(token1)} and {self._token(token2)}")
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
        word_frequencies = pre_tokenize(text)
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
            most_common_token_pair, _freq = token_pairs_counter.most_common(n=1)[0]
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
        assert len(word) > 0
        tokens: list[TokenId] = list(word)
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

    def tokenize_word_greedy(self, word: bytes) -> list[TokenId]:
        """Tokenizes the word in a greedy way (longest token picked each time)

        THIS IS WRONG AND SHOULD NOT BE USED
        REAL BPE MERGES TOKENS IN TRAINING ORDER
        NOTE TO SELF: KEEPING THIS WRONG CODE ONLY AS REMINDER TO NOT REPEAT
                      THE MISTAKE.
        """
        tokens = []
        remaining_word = word
        while len(remaining_word) > 0:
            # Find the longest prefix each time and replace with TokenId
            for prefix_len in range(len(remaining_word)):
                prefix = remaining_word[:prefix_len]
                if prefix in self.text_to_id:
                    tokens.append(self.text_to_id[prefix])
        return tokens

    def save(self, filename: str):
        pass

    def _token(self, token: TokenId):
        return f"<{self.id_to_text[token].decode(errors='replace')}>"


def pre_tokenize(raw_bytes: bytes) -> Counter[bytes]:
    documents = raw_bytes.split(EOT_TOKEN)
    word_counts = Counter()
    for doc in documents:
        words = (match.group() for match in re.finditer(PRE_TOKENIZER_SPLIT, doc))
        word_counts.update(words)
    return word_counts


def run_tokenizer(input_file, output_file):
    # ok, we will optimize the chunking later
    raw_bytes = open(input_file, "rb").read()
    t = Tokenizer()
    t.train_tokenizer(raw_bytes)
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
