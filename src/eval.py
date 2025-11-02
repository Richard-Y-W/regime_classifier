import argparse

def main(args):
    print(f"Evaluating model {args.model} on data {args.input}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model performance")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--in", dest="input", type=str, required=True)
    args = parser.parse_args()

    main(args)
