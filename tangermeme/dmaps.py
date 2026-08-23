from __future__ import annotations

import contextlib
from collections.abc import Sequence

import torch

from tqdm import trange

from ._compat import _autocast_supported, _preserve_model_state, _resolve_device
from .utils import _validate_input, one_hot_encode


def calculate_saliency_map(
	model: torch.nn.Module,
	X: torch.Tensor,
	args: Sequence[torch.Tensor] | None = None,
	target: int = 0,
	batch_size: int = 32,
	dtype: str | torch.dtype | None = None,
	device: str | torch.device | None = None,
	verbose: bool = False,
) -> torch.Tensor:
	"""Calculate input * gradient saliency maps for a batch of sequences.

	This function will take a PyTorch model and a batch of one-hot encoded
	sequences and, for each sequence, compute the gradient of the model's
	`target`-th output with respect to the input and multiply it by the
	input itself, summing over the alphabet axis to yield one importance
	score per position. This is the same batched-gradient strategy used by
	`deep_lift_shap`: the batch elements do not interact during the forward
	pass, so a single `backward()` call on the summed output yields the
	correct per-example gradient for every example in the batch at once.

	As with `predict`, each batch is moved to `device` and cast to `dtype`
	before the forward/backward pass and moved back to the CPU afterward,
	so `X` can be kept in a memory-efficient dtype (e.g. int8) even though
	gradients require floating point.

	NOTE: like `deep_lift_shap`, this requires a model that returns a single
	`(batch_size, n_targets)` tensor. If your model returns something more
	complicated, wrap it so that the forward pass yields such a tensor.


	Parameters
	----------
	model: torch.nn.Module
		The PyTorch model to use to calculate saliency maps.

	X: torch.Tensor, shape=(-1, len(alphabet), length)
		A one-hot encoded set of sequences to calculate saliency maps for.

	args: tuple or list or None, optional
		An optional set of additional arguments to pass into the model, with
		the same semantics as in `predict`. Default is None.

	target: int, optional
		The output of the model to calculate the saliency map for. This
		indexes the last dimension of the model's predictions. Default is 0.

	batch_size: int, optional
		The number of examples to process at a time. Default is 32.

	dtype: str or torch.dtype or None, optional
		The dtype to use with mixed precision autocasting. If None, use the
		dtype of the *model*. Default is None.

	device: str or torch.device or None, optional
		The device to move the model and batches to. If None, use CUDA when
		available and fall back to CPU otherwise. The model's original
		device and training mode are restored after the call. Default is
		None.

	verbose: bool, optional
		Whether to display a progress bar. Default is False.


	Returns
	-------
	saliency: torch.Tensor, shape=(-1, length)
		The saliency map for each input sequence.
	"""

	_validate_input(X, "X", shape=(-1, -1, -1), ohe=True, allow_N=True)

	if X.shape[0] == 0:
		raise ValueError("calculate_saliency_map requires at least one "
			"example; got X with shape[0] == 0.")

	device = _resolve_device(device)

	if dtype is None:
		try:
			dtype = next(model.parameters()).dtype
		except (StopIteration, AttributeError):
			dtype = torch.float32
	elif isinstance(dtype, str):
		dtype = getattr(torch, dtype)

	if args is not None:
		for arg in args:
			if arg.shape[0] != X.shape[0]:
				raise ValueError("Arguments must have the same first " +
					"dimension as X")

	use_autocast = _autocast_supported(device, dtype)

	saliency = []
	with _preserve_model_state(model, device):
		batch_size_ = min(batch_size, X.shape[0])

		for start in trange(0, X.shape[0], batch_size_, disable=not verbose):
			end = start + batch_size_
			X_ = X[start:end].to(device).type(dtype)

			if X_.shape[0] == 0:
				continue

			X_.requires_grad_()

			if args is not None:
				args_ = [a[start:end].type(dtype).to(device) for a in args]
			else:
				args_ = []

			if use_autocast:
				autocast_ctx = torch.autocast(device_type=device.type,
					dtype=dtype)
			else:
				autocast_ctx = contextlib.nullcontext()

			with torch.autograd.set_grad_enabled(True):
				with autocast_ctx:
					y = model(X_, *args_)

					if not isinstance(y, torch.Tensor):
						raise ValueError("calculate_saliency_map requires "
							"a model that returns a single tensor output; "
							"got {}.".format(type(y)))

					y_target = y[:, target]

				grad, = torch.autograd.grad(y_target.sum(), X_)

			sal = (grad.detach().abs() * X_.detach()).sum(dim=1)
			saliency.append(sal.cpu())

	return torch.cat(saliency)


def dependency_map(
	model: torch.nn.Module,
	X: str | torch.Tensor,
	args: Sequence[torch.Tensor] | None = None,
	target: int = 0,
	batch_size: int = 32,
	dtype: str | torch.dtype | None = None,
	device: str | torch.device | None = None,
	verbose: bool = False,
) -> torch.Tensor:
	"""Generate dependency maps for a batch of sequences using a model.

	A dependency map quantifies, for each pair of positions (i, j) in a
	sequence, how much substituting the base at position i changes the
	saliency (input * gradient importance, see `calculate_saliency_map`) at
	position j. It is built by taking every possible single-position
	substitution of the input sequence, recomputing the saliency map for
	each one, and comparing it to the saliency map of the unperturbed
	sequence.

	Like `saturation_mutagenesis`, this method needs to run the model on
	every single-character substitution of every input sequence -- for an
	alphabet of size A and a sequence of length L, that is A * L forward and
	backward passes per sequence. All of these substitutions across all
	sequences are generated up front (as a memory-efficient int8 tensor, the
	same trick `saturation_mutagenesis` uses) and then streamed through
	`calculate_saliency_map` in batches of `batch_size`, which keeps the
	peak memory footprint independent of A * L while still batching the
	actual model calls for speed. Unlike the original single-sequence,
	one-substitution-at-a-time implementation, this also means multiple
	sequences can be processed in one call.

	For every position i, the A substitutions include the one that leaves
	the base unchanged (a "no-op" edit that reconstructs the original
	sequence); its saliency map is therefore identical to the unperturbed
	one, giving a difference of exactly zero. Averaging over all A
	substitutions (rather than filtering the no-op edit out per position)
	keeps the substitution tensor a plain, uniformly-shaped grid -- the
	trade-off is a uniform rescaling of the result by a factor of
	(A - 1) / A relative to averaging over only the true substitutions,
	which does not change the relative structure of the map.


	Parameters
	----------
	model: torch.nn.Module
		The PyTorch model to use to compute the dependency maps.

	X: str or torch.Tensor, shape=(-1, len(alphabet), length)
		The input sequence(s) to calculate the dependency map for. Can be a
		single string, which is one-hot encoded internally and treated as a
		batch of size 1, or a one-hot encoded tensor with an arbitrary
		alphabet size and batch size.

	args: tuple or list or None, optional
		An optional set of additional arguments to pass into the model, with
		the same semantics as in `predict`. Default is None.

	target: int, optional
		The output of the model to calculate the dependency map for. This
		indexes the last dimension of the model's predictions. Default is 0.

	batch_size: int, optional
		The number of substitutions to run through the model at a time.
		Default is 32.

	dtype: str or torch.dtype or None, optional
		The dtype to use with mixed precision autocasting. If None, use the
		dtype of the *model*. Default is None.

	device: str or torch.device or None, optional
		The device to move the model and batches to. If None, use CUDA when
		available and fall back to CPU otherwise. Default is None.

	verbose: bool, optional
		Whether to display a progress bar over the substitutions. Default is
		False.


	Returns
	-------
	dependency_maps: torch.Tensor, shape=(-1, length, length)
		The dependency map for each input sequence. `dependency_maps[:, j, i]`
		is the average absolute change in the saliency at position j induced
		by substituting the base at position i.
	"""

	if isinstance(X, str):
		X = one_hot_encode(X).unsqueeze(0)

	_validate_input(X, "X", shape=(-1, -1, -1), ohe=True, allow_N=True)

	if X.shape[0] == 0:
		raise ValueError("dependency_map requires at least one example; "
			"got X with shape[0] == 0.")

	if args is not None:
		for arg in args:
			if arg.shape[0] != X.shape[0]:
				raise ValueError("Arguments must have the same first " +
					"dimension as X")

	n, alphabet_size, length = X.shape
	edits_per_seq = alphabet_size * length

	base_saliency = calculate_saliency_map(model, X, args=args, target=target,
		batch_size=batch_size, dtype=dtype, device=device)

	# Lay out, for every sequence, every (character, position) substitution
	# in [character, position] order -- mirroring the edit grid built in
	# `saturation_mutagenesis` -- so the result reshapes cleanly afterward.
	seq_idx = torch.arange(n).repeat_interleave(edits_per_seq)
	edit_chars = torch.arange(alphabet_size).repeat_interleave(length).repeat(n)
	edit_positions = torch.arange(length).repeat(alphabet_size * n)
	rows = torch.arange(seq_idx.shape[0])

	X_edits = X.type(torch.int8).cpu()[seq_idx]
	X_edits[rows, :, edit_positions] = 0
	X_edits[rows, edit_chars, edit_positions] = 1

	args_ = tuple(a[seq_idx] for a in args) if args is not None else None

	edit_saliency = calculate_saliency_map(model, X_edits, args=args_,
		target=target, batch_size=batch_size, dtype=dtype, device=device,
		verbose=verbose)

	diff = (edit_saliency - base_saliency[seq_idx]).abs()
	diff = diff.reshape(n, alphabet_size, length, length)

	return diff.mean(dim=1).transpose(1, 2)