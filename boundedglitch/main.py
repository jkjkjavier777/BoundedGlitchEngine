"""
BoundedGlitchEngine: Main entry point
Run from root directory: python main.py
"""

import argparse
import sys
from pathlib import Path

# Add boundedglitch to path
sys.path.insert(0, str(Path(__file__).parent / 'boundedglitch'))

from engine.bot import BoundedGlitchEngine


def main():
    parser = argparse.ArgumentParser(description="BoundedGlitchEngine Bot")
    parser.add_argument('--checkpoint', default='boundedglitch/gpt_model/checkpoints/model.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--tokenizer', default='boundedglitch/gpt_model/checkpoints/tokenizer.json',
                        help='Path to tokenizer')
    parser.add_argument('--device', default='cpu', help='cuda or cpu')

    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"[!] Checkpoint not found: {args.checkpoint}")
        print("\nFirst, train the model:")
        print(f"  cd boundedglitch && python scripts/train.py")
        sys.exit(1)

    if not Path(args.tokenizer).exists():
        print(f"[!] Tokenizer not found: {args.tokenizer}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("BOUNDEDGLITCHENGINE")
    print("=" * 70 + "\n")

    print("[*] Initializing engine...")
    engine = BoundedGlitchEngine(args.checkpoint, args.tokenizer, device=args.device)
    print("[✓] Engine ready\n")

    engine.interactive_mode()


if __name__ == "__main__":
    main()
