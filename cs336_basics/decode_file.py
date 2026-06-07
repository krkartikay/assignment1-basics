import argparse
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode token ids from a .npy file.")
    parser.add_argument("--input_file", default="out/owt_train.npy", help="Input token ids path (.npy)")
    parser.add_argument("--vocab_file", default="out/owt_tokenizer_vocab.pkl", help="Tokenizer vocab path (.pkl)")
    parser.add_argument("--merges_file", default="out/owt_tokenizer_merges.pkl", help="Tokenizer merges path (.pkl)")
    parser.add_argument("--start", default=0, type=int, help="First token offset to decode")
    parser.add_argument("--count", default=500, type=int, help="Number of tokens to decode")
    args = parser.parse_args()

    token_ids = np.load(Path(args.input_file), mmap_mode="r")
    tokenizer = Tokenizer.from_files(args.vocab_file, args.merges_file)

    end = min(args.start + args.count, len(token_ids))
    ids = token_ids[args.start : end].astype(int).tolist()

    decoded = tokenizer.decode(ids)
    decoded_bytes = decoded.encode("utf-8")
    print(f"Loaded {len(token_ids)} token ids from {args.input_file}.")
    print(f"Decoded token slice [{args.start}:{end}].")
    print(f"Tokens: {len(ids)}")
    print(f"Characters: {len(decoded)}")
    print(f"UTF-8 bytes: {len(decoded_bytes)}")
    print(f"Bytes per token: {len(decoded_bytes) / len(ids):.3f}")
    print("Decoded text:")
    for id in ids:
        decoded_token = tokenizer.decode([id])
        print(f"<{decoded_token}>", end="")


if __name__ == "__main__":
    main()
