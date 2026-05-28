#!/usr/bin/env python3
import argparse, sys, csv, io
from typing import Dict

DIGITS = "123456789"
ROWS = "ABCDEFGHI"
COLS = "123456789"
CELLS = [r+c for r in ROWS for c in COLS]

ROW_UNITS = [[r+c for c in COLS] for r in ROWS]
COL_UNITS = [[r+c for r in ROWS] for c in COLS]
BOX_UNITS = [[r+c for r in rs for c in cs] 
             for rs in ("ABC","DEF","GHI") for cs in ("123","456","789")]
UNITS = {s: [u for u in (ROW_UNITS + COL_UNITS + BOX_UNITS) if s in u] for s in CELLS}
PEERS = {s: set(sum(UNITS[s], [])) - {s} for s in CELLS}

def parse_grid(grid: str) -> Dict[str, str]:
    chars = [c for c in grid if c in DIGITS or c in "0."]
    if len(chars) != 81:
        raise ValueError("Grid must contain exactly 81 digits/./0 after cleaning")
    values = {s: DIGITS for s in CELLS}
    for s, ch in zip(CELLS, chars):
        if ch in DIGITS:
            if not assign(values, s, ch):
                return {}
    return values

def assign(values: Dict[str, str], s: str, d: str) -> Dict[str, str]:
    other = values[s].replace(d, "")
    if all(eliminate(values, s, d2) for d2 in other):
        return values
    return {}

def eliminate(values: Dict[str, str], s: str, d: str) -> bool:
    if d not in values[s]:
        return True
    values[s] = values[s].replace(d, "")
    if len(values[s]) == 0:
        return False
    if len(values[s]) == 1:
        d2 = values[s]
        if not all(eliminate(values, p, d2) for p in PEERS[s]):
            return False
    for u in UNITS[s]:
        places = [s2 for s2 in u if d in values[s2]]
        if len(places) == 0:
            return False
        if len(places) == 1:
            if not assign(values, places[0], d):
                return False
    return True

def solved(values: Dict[str, str]) -> bool:
    return all(len(values[s]) == 1 for s in CELLS)

def search(values: Dict[str, str]) -> Dict[str, str]:
    if not values:
        return {}
    if solved(values):
        return values
    s = min((c for c in CELLS if len(values[c]) > 1), key=lambda c: len(values[c]))
    for d in values[s]:
        new_vals = values.copy()
        if assign(new_vals, s, d):
            res = search(new_vals)
            if res:
                return res
    return {}

def normalize_to_81_chars(text: str) -> str:
    text = text.strip()
    if "," in text:
        reader = csv.reader(io.StringIO(text))
        tokens = []
        for row in reader:
            for cell in row:
                cell = cell.strip()
                tokens.extend(list(cell))
        chars = [c for c in tokens if c in DIGITS or c in "0."]
        if len(chars) != 81:
            raise ValueError("CSV did not contain 81 valid cells")
        return "".join(chars)
    if "\n" in text or " " in text or "\t" in text:
        chars = [c for c in text if c in DIGITS or c in "0."]
        if len(chars) != 81:
            lines = [ln for ln in text.splitlines() if ln.strip()]
            flat = []
            for ln in lines:
                for tok in ln.replace(",", " ").split():
                    tok = tok.strip()
                    if len(tok) == 1 and (tok in DIGITS or tok in "0."):
                        flat.append(tok)
                    elif len(tok) > 1:
                        flat.extend([c for c in tok if c in DIGITS or c in "0."])
            chars = [c for c in flat if c in DIGITS or c in "0."]
        if len(chars) != 81:
            raise ValueError("Text format did not resolve to 81 cells")
        return "".join(chars)
    if len([c for c in text if c in DIGITS or c in "0."]) == 81:
        return "".join([c for c in text if c in DIGITS or c in "0."])
    raise ValueError("Unrecognized format; need 81 digits/./0 total")

def solve_text(text: str) -> str:
    vals = parse_grid(normalize_to_81_chars(text))
    if not vals:
        raise SystemExit("Invalid puzzle (contradiction in givens).")
    sol = search(vals)
    if not sol:
        raise SystemExit("No solution found (puzzle may be unsolvable).")
    # Return as 9 lines
    lines = []
    for r in ROWS:
        lines.append(" ".join(sol[r+c] for c in COLS))
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser(description="Solve a Sudoku from a file or stdin.")
    ap.add_argument("path", nargs="?", help="Path to TXT/CSV puzzle (or omit to read stdin)")
    args = ap.parse_args()
    if args.path:
        with open(args.path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()
    print(solve_text(text))

if __name__ == "__main__":
    main()
