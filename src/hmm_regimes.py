import argparse

def main(args):
    print(f"Training HMM on {args.input} -> saving to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hidden Markov Model regime classifier")
    parser.add_argument("--in", dest="input", type=str, required=True)
    parser.add_argument("--out", dest="output", type=str, required=True)
    args = parser.parse_args()

    main(args)
