#!/usr/bin/env python3
"""
Prepare submission ZIP for WBA2026 embeddings.

Usage:
	python scripts/prepare_submission_file.py <embeddings_gen1> <embeddings_gen2> <distance>

The script accepts two embedding files (NumPy .npy or .bl2) and a distance flag
('euclidean' or 'cosine'). It validates that each embedding array has shape
(40200, D) (D can be any positive integer), then creates a ZIP named
`wba2026-embedding-submissoin.zip` containing both embedding files and a
manifest JSON describing the metric and filenames.

On validation failure the script exits with a non-zero status and prints an error message.
"""

import sys
import os
import json
import argparse
import zipfile

from torch import Tensor
import torch
import numpy as np

try:
	import blosc2
except Exception:
	print('ERROR: blosc2 is required to run this script. Install it with `pip install blosc2`.', file=sys.stderr)
	sys.exit(1)


def load_array(path):
	path = str(path)
	return torch.as_tensor(np.load(path, allow_pickle=False) if path.endswith('.npy') else blosc2.load_tensor(path))


def validate_shape(arr: Tensor, expected_rows=40200):
	if arr.ndim != 2:
		return False, f'expected 2D array, got ndim={arr.ndim}'
	if arr.shape[0] != expected_rows:
		return False, f'expected first dimension {expected_rows}, got {arr.shape[0]}'
	if arr.shape[1] <= 0:
		return False, f'invalid embedding dimension {arr.shape[1]}'
	if not torch.isfinite(torch.as_tensor(arr)).all():
		return False, 'embeddings contain NaN or infinity'
	return True, None


def main():
	parser = argparse.ArgumentParser(description='Prepare WBA2026 embedding submission ZIP')
	parser.add_argument('gen1_embeddings', help='Path to embeddings file for gen1 (npy or bl2)')
	parser.add_argument('gen2_embeddings', help='Path to embeddings file for gen2 (npy or bl2)')
	parser.add_argument('distance', choices=['euclidean', 'cosine'], help='Distance metric to declare')
	parser.add_argument('modality', choices=['video', '3DMM', 'landmarks', 'video+landmarks'], help='Modality to declare (default: %(default)s)')
	parser.add_argument('--out', default='wba2026-embedding-submissoin.zip', help='Output zip filename')
	args = parser.parse_args()

	if not os.path.exists(args.gen1_embeddings):
		print(f'ERROR: gen1 embeddings not found: {args.gen1_embeddings}', file=sys.stderr)
		sys.exit(2)
	if not os.path.exists(args.gen2_embeddings):
		print(f'ERROR: gen2 embeddings not found: {args.gen2_embeddings}', file=sys.stderr)
		sys.exit(2)

	try:
		arr1 = load_array(args.gen1_embeddings)
	except Exception as e:
		print(f'ERROR loading gen1 embeddings: {e}', file=sys.stderr)
		sys.exit(3)

	try:
		arr2 = load_array(args.gen2_embeddings)
	except Exception as e:
		print(f'ERROR loading gen2 embeddings: {e}', file=sys.stderr)
		sys.exit(3)

	ok, msg = validate_shape(arr1)
	if not ok:
		print(f'ERROR: gen1 embeddings validation failed: {msg}', file=sys.stderr)
		sys.exit(4)

	ok, msg = validate_shape(arr2)
	if not ok:
		print(f'ERROR: gen2 embeddings validation failed: {msg}', file=sys.stderr)
		sys.exit(4)
	if arr1.shape[1] != arr2.shape[1]:
		parser.error('both generations must use the same embedding dimension')
	if args.distance == 'cosine' and any((a.float().norm(dim=1) <= 1e-12).any() for a in (arr1, arr2)):
		parser.error('cosine embeddings must not contain zero vectors')
	if os.path.basename(args.gen1_embeddings) == os.path.basename(args.gen2_embeddings):
		parser.error('generation embedding filenames must differ inside the ZIP')

	# create a zip containing the two files and a manifest.json
	out_zip = args.out
	manifest = {
		'matching': args.distance,
		'gen1': os.path.basename(args.gen1_embeddings),
		'gen2': os.path.basename(args.gen2_embeddings),
		'modality': args.modality,
	}

	try:
		with zipfile.ZipFile(out_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
			# add embeddings (store with their basenames)
			zf.write(args.gen1_embeddings, arcname=os.path.basename(args.gen1_embeddings))
			zf.write(args.gen2_embeddings, arcname=os.path.basename(args.gen2_embeddings))
			# add manifest
			manifest_bytes = json.dumps(manifest, indent=2).encode('utf-8')
			zf.writestr('manifest.json', manifest_bytes)
	except Exception as e:
		print(f'ERROR creating zip file: {e}', file=sys.stderr)
		sys.exit(5)

	print(f'Created {out_zip} with files: {manifest["gen1"]}, {manifest["gen2"]}')
	print('Manifest:')
	print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
	main()
