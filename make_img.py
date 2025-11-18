#!/usr/bin/env python
# tty_view_vivid.py  — vivid image preview in terminal (no GUI, no saving)
# Usage:
#   python tty_view_vivid.py /path/to/img.jpg [--width 120] [--resample box|nearest|bicubic] [--halfblock]
# Tips for vivid color:
#   - Ensure truecolor: export COLORTERM=truecolor
#   - In tmux: set -g default-terminal "tmux-256color"; set -ga terminal-overrides ",xterm-256color:RGB"

import argparse, shutil, os
from PIL import Image

RESET = "\x1b[0m"

def supports_truecolor() -> bool:
    ct = os.environ.get("COLORTERM","").lower()
    return "truecolor" in ct or "24bit" in ct

def resample_filter(name: str):
    name = name.lower()
    if name == "nearest": return Image.NEAREST
    if name == "box":     return Image.BOX     # preserves edges/colors nicely
    if name == "bicubic": return Image.BICUBIC # smooth but can desaturate
    return Image.BOX

def print_block_bg(r,g,b):
    # background color block
    return f"\x1b[48;2;{r};{g};{b}m "

def print_halfblock(top, bot):
    # top pixel as FG, bottom as BG, draw '▀'
    tr,tg,tb = top
    br,bg,bb = bot
    return f"\x1b[38;2;{tr};{tg};{tb}m\x1b[48;2;{br};{bg};{bb}m▀"

def show_in_tty(path: str, width: int|None, mode: str, use_halfblock: bool):
    if not supports_truecolor():
        # still works with 256-color terminals, but colors may be dull
        pass

    im = Image.open(path).convert("RGB")
    term_cols = shutil.get_terminal_size((80, 24)).columns
    tw = min(width or term_cols, 200)  # cap to avoid super-wide spam

    w, h = im.size
    if use_halfblock:
        # Two image rows per terminal row using '▀'
        th = max(1, int(h * (tw / w)))  # no 0.5 factor here
        im = im.resize((tw, max(2, th*2)), resample_filter(mode))
        px = im.load()
        out_lines = []
        for y in range(0, im.height - 1, 2):
            row = []
            for x in range(im.width):
                top = px[x, y]
                bot = px[x, y+1]
                row.append(print_halfblock(top, bot))
            out_lines.append("".join(row) + RESET)
        print("\n".join(out_lines))
        print(RESET, end="")
    else:
        # One image row per terminal row, adjust aspect ratio with 0.5 factor
        th = max(1, int(h * (tw / w) * 0.5))
        im = im.resize((tw, th), resample_filter(mode))
        px = im.load()
        for y in range(im.height):
            row = []
            for x in range(im.width):
                r,g,b = px[x, y]
                row.append(print_block_bg(r,g,b))
            print("".join(row) + RESET)
        print(RESET, end="")

def main():
    ap = argparse.ArgumentParser(description="Vivid image preview in terminal (ANSI truecolor).")
    ap.add_argument("path", help="Path to image.")
    ap.add_argument("--width", type=int, default=None, help="Target terminal width (cells). Default: terminal width.")
    ap.add_argument("--resample", default="box", choices=["box","nearest","bicubic"],
                    help="Resize filter. 'box' or 'nearest' keep saturation better. Default: box.")
    ap.add_argument("--halfblock", action="store_true",
                    help="Use '▀' to draw two pixels per cell (more detail, more vivid).")
    args = ap.parse_args()
    show_in_tty(args.path, args.width, args.resample, args.halfblock)

if __name__ == "__main__":
    main()

