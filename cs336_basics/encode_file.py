import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from cs336_basics.tokenizer import Tokenizer


def token_dtype(tokenizer: Tokenizer) -> np.dtype:
    max_id = max(tokenizer.id_to_text)
    return np.dtype(np.uint16 if max_id < 2**16 else np.uint32)


def save_token_ids(token_ids: np.ndarray, output_file: str) -> int:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".npy":
        np.save(output_path, token_ids)
    elif output_path.suffix == ".bin":
        token_ids.tofile(output_path)
    else:
        raise ValueError("Output file must end in .npy or .bin")

    return output_path.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode a text file to token ids.")
    parser.add_argument("--input_file", default="data/owt_train.txt", help="Input text path")
    parser.add_argument(
        "--output_file",
        default="out/owt_train.npy",
        help="Output token ids path (.npy or .bin)",
    )
    parser.add_argument(
        "--vocab_file",
        default="out/owt_tokenizer_vocab.pkl",
        help="Tokenizer vocab path (.pkl)",
    )
    parser.add_argument(
        "--merges_file",
        default="out/owt_tokenizer_merges.pkl",
        help="Tokenizer merges path (.pkl)",
    )
    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(args.vocab_file, args.merges_file)
    dtype = token_dtype(tokenizer)

    with open(args.input_file, encoding="utf-8") as f:
        token_ids = np.fromiter(tokenizer.encode_iterable(tqdm(f, desc="Encoding", unit="lines")), dtype=dtype)

    input_bytes = Path(args.input_file).stat().st_size
    output_bytes = save_token_ids(token_ids, args.output_file)
    compression_ratio = input_bytes / output_bytes if output_bytes else float("inf")

    print(f"Loaded tokenizer from {args.vocab_file} and {args.merges_file}.")
    print(f"Encoded {args.input_file} to {args.output_file}.")
    print(f"Tokens: {len(token_ids)}")
    print(f"Dtype: {dtype}")
    print(f"Input bytes: {input_bytes}")
    print(f"Output bytes: {output_bytes}")
    print(f"Bytes per token: {input_bytes / len(token_ids):.3f}")
    print(f"Compression ratio: {compression_ratio:.3f}x")


if __name__ == "__main__":
    main()
