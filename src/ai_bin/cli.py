
import argparse, pandas as pd
from .utils import prepare_binary, stratified_split

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    pbin = sub.add_parser("prepare_bin", help="Prepare binary data (real vs fake)")
    pbin.add_argument("--train", required=True, help="raw train.csv (Kaggle)")
    pbin.add_argument("--out-train", required=True, help="output processed train.csv")
    pbin.add_argument("--out-val", required=True, help="output processed val.csv")
    pbin.add_argument("--use-keyword-location", action="store_true")
    pbin.add_argument("--lowercase", action="store_true")
    pbin.add_argument("--seed", type=int, default=42)
    pbin.add_argument("--val-size", type=float, default=0.2)

    args = ap.parse_args()
    if args.cmd == "prepare_bin":
        raw = pd.read_csv(args.train)
        proc = prepare_binary(raw, use_keyword_location=args.use_keyword_location, lowercase=args.lowercase)
        if "label" not in proc.columns:
            raise SystemExit("Input train CSV must contain 'target' to derive labels.")
        tr, va = stratified_split(proc, test_size=args.val_size, seed=args.seed)
        tr.to_csv(args.out_train, index=False)
        va.to_csv(args.out_val, index=False)
        print({"train": args.out_train, "val": args.out_val, "n_train": len(tr), "n_val": len(va)})

if __name__ == "__main__":
    main()
