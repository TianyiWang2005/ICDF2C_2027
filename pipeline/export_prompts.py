"""Write every prompt in pipeline/prompts.py to prompts/<NAME>.txt (verbatim, for the paper appendix)."""
import os

from pipeline import prompts as P

os.makedirs("prompts", exist_ok=True)
for k, v in vars(P).items():
    if k.isupper() and isinstance(v, str):
        open(f"prompts/{k}.txt", "w", encoding="utf-8").write(v)
        print("prompts/" + k + ".txt")
