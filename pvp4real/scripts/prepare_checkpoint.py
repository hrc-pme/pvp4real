#!/usr/bin/env python3
"""
Utility script to prepare checkpoint directories for resume training.

This script helps extract .zip checkpoint files and prepare them for use
with the resume_from parameter in pvp.hitl.py.

Usage:
    python prepare_checkpoint.py <checkpoint.zip> [output_dir]

Example:
    python prepare_checkpoint.py /workspace/pvp4real/models/stretch3_hitl/pvp4real_stretch3_step100.zip
    
The script will:
1. Extract the .zip file to a directory (or use provided output_dir)
2. Verify all required checkpoint files are present
3. Display the directory path to use with --resume_from
"""

import argparse
import zipfile
from pathlib import Path
import sys


def verify_checkpoint_files(directory: Path) -> bool:
    """Verify that all required checkpoint files are present."""
    required_files = [
        "policy.pth",
        "pytorch_variables.pth",
        "_stable_baselines3_version",
    ]
    
    optional_files = [
        "actor.optimizer.pth",
        "critic.optimizer.pth",
        "data",
        "system_info.txt",
    ]
    
    missing_required = []
    for fname in required_files:
        if not (directory / fname).exists():
            missing_required.append(fname)
    
    if missing_required:
        print(f"❌ Error: Missing required checkpoint files: {missing_required}")
        return False
    
    print("✅ All required checkpoint files present:")
    for fname in required_files:
        print(f"   - {fname}")
    
    print("\n📋 Optional files found:")
    for fname in optional_files:
        if (directory / fname).exists():
            print(f"   - {fname}")
    
    return True


def extract_checkpoint(zip_path: Path, output_dir: Path = None) -> Path:
    """Extract checkpoint .zip file to a directory."""
    if not zip_path.exists():
        print(f"❌ Error: Checkpoint file not found: {zip_path}")
        sys.exit(1)
    
    if not zip_path.suffix == '.zip':
        print(f"❌ Error: File is not a .zip archive: {zip_path}")
        sys.exit(1)
    
    # Determine output directory
    if output_dir is None:
        # Extract to same directory with _extracted suffix
        output_dir = zip_path.parent / f"{zip_path.stem}_extracted"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📦 Extracting checkpoint from: {zip_path}")
    print(f"📂 Output directory: {output_dir}")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print("✅ Extraction completed successfully")
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        sys.exit(1)
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Prepare checkpoint directory for resume training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract checkpoint to auto-generated directory
  python prepare_checkpoint.py models/checkpoint_step100.zip
  
  # Extract to specific directory
  python prepare_checkpoint.py models/checkpoint_step100.zip models/resume_checkpoint
  
  # Verify existing directory (no extraction)
  python prepare_checkpoint.py models/checkpoint_extracted --verify-only
        """
    )
    
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to checkpoint .zip file or directory to verify"
    )
    
    parser.add_argument(
        "output_dir",
        type=str,
        nargs='?',
        default=None,
        help="Output directory for extraction (optional, auto-generated if not provided)"
    )
    
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify checkpoint files, don't extract"
    )
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint_path)
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    print("=" * 60)
    print("PVP4Real Checkpoint Preparation Utility")
    print("=" * 60)
    print()
    
    # If checkpoint_path is a directory, verify it
    if checkpoint_path.is_dir():
        print(f"📂 Verifying checkpoint directory: {checkpoint_path}")
        if verify_checkpoint_files(checkpoint_path):
            print("\n" + "=" * 60)
            print("✅ Checkpoint directory is ready for use!")
            print("=" * 60)
            print("\nTo resume training, use:")
            print(f"  --resume_from {checkpoint_path.absolute()}")
            print("\nOr in config.yaml:")
            print(f"  resume_from: \"{checkpoint_path.absolute()}\"")
        else:
            sys.exit(1)
    
    # If checkpoint_path is a file, extract it
    elif checkpoint_path.is_file():
        if args.verify_only:
            print("❌ Error: --verify-only can only be used with directories")
            sys.exit(1)
        
        extracted_dir = extract_checkpoint(checkpoint_path, output_dir)
        print()
        
        if verify_checkpoint_files(extracted_dir):
            print("\n" + "=" * 60)
            print("✅ Checkpoint is ready for resume training!")
            print("=" * 60)
            print("\nTo resume training, use:")
            print(f"  python pvp.hitl.py --resume_from {extracted_dir.absolute()}")
            print("\nOr in config.yaml:")
            print(f"  is_resume_training: true")
            print(f"  resume_from: \"{extracted_dir.absolute()}\"")
            print("\nOr use the .zip file directly:")
            print(f"  python pvp.hitl.py --resume_from {checkpoint_path.absolute()}")
        else:
            print("\n⚠️  Warning: Checkpoint files are incomplete or corrupted")
            sys.exit(1)
    
    else:
        print(f"❌ Error: Path not found: {checkpoint_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
