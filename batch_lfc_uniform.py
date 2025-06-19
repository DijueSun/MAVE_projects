import argparse      # ← MUST be present
import sys
from pathlib import Path
import numpy as np
import pandas as pd




def lfc_vs_uniform(df, pseudo=1.0):
    """
    Expect columns: ID, raw.  Adds raw_norm, ref_norm (=uniform), LFC
    """
    n = len(df)
    df["raw_norm"] = (df["raw"] + pseudo) / (df["raw"].sum() + pseudo * n)
    df["ref_norm"] = 1.0 / n                    # uniform expectation
    df["LFC"]      = np.log2(df["raw_norm"] / df["ref_norm"])
    return df[["ID", "raw", "LFC"]]


def process_one(file_path, pseudo):
    """
    Read a single *.lib_counts.tsv.gz file and return an LFC table (DataFrame)
    """
    df = pd.read_csv(file_path, sep="\t", comment="#",
                     names=["ID", "NAME", "SEQ", "COUNT", "UNIQUE", "SAMPLE"],
                     compression="gzip")
    df = df[["ID", "COUNT"]].rename(columns={"COUNT": "raw"})
    return lfc_vs_uniform(df, pseudo=pseudo)


def main(root, outdir, pseudo):
    root   = Path(root).expanduser()
    outdir = Path(outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    lib_files = sorted(root.rglob("*.lib_counts.tsv.gz"))
    if not lib_files:
        sys.exit(f"No *.lib_counts.tsv.gz files found under {root}")

    for fp in lib_files:
        sample = fp.stem.split(".")[0]           # e.g. CELL_index_PQCL131
        print(f"▶  {sample}")
        lfc_df = process_one(fp, pseudo)
        out_tsv = outdir / f"{sample}_LFC.tsv"
        lfc_df.to_csv(out_tsv, sep="\t", index=False)
        print(f"   saved {out_tsv}")

    print(f"\nDone.  {len(lib_files)} files processed; results in {outdir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Batch LFC calculation vs uniform expectation")
    p.add_argument("root",
                   help="Top-level directory to search (recursively) for *.lib_counts.tsv.gz")
    p.add_argument("--outdir", default="LFC_tables",
                   help="Where to write the per-sample LFC tables")
    p.add_argument("--pseudo", type=float, default=1.0,
                   help="Pseudo-count added to every oligo (default 1)")
    args = p.parse_args()
    main(args.root, args.outdir, args.pseudo)
