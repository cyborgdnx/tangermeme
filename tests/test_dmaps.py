# test_dmaps.py
# Contact: Tanmay Debnath <tanmaydebnath.tut.jp@gmail.com>

import torch
torch.use_deterministic_algorithms(True, warn_only=True)
torch.manual_seed(0)


import numpy as np
import pytest

from tangermeme.utils import one_hot_encode
from tangermeme.utils import random_one_hot

from tangermeme.dmaps import calculate_saliency_map
from tangermeme.dmaps import dependency_map

from .toy_models import FlattenDense
from .toy_models import ConvAvgDense

from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal


class _LinearConv(torch.nn.Module):
	"""Two Conv1d layers with no activation between them, mean-pooled to a
	scalar. Composing linear ops keeps the whole model linear (unlike
	`Conv1` in toy_models.py, which lacks `padding='same'` and so does not
	reduce to a single scalar output), so its Jacobian w.r.t. the input is
	constant -- used to check that dependency_map stays diagonal-only for
	linear models with more than a single Linear layer.
	"""

	def __init__(self):
		super(_LinearConv, self).__init__()
		self.conv1 = torch.nn.Conv1d(4, 6, 3, padding='same')
		self.conv2 = torch.nn.Conv1d(6, 1, 3, padding='same')

	def forward(self, X):
		return self.conv2(self.conv1(X)).mean(dim=-1)


@pytest.fixture
def X0():
	return random_one_hot((1, 4, 10), random_state=0).float()


###
# calculate_saliency_map
###


def test_calculate_saliency_map_shape(X0, device):
	model = FlattenDense(seq_len=10, n_outputs=1)
	sal = calculate_saliency_map(model, X0.clone(), device=device)

	assert isinstance(sal, np.ndarray)
	assert sal.shape == (10,)


def test_calculate_saliency_map_non_negative(X0, device):
	# The map is built from the absolute value of the gradient, so it can
	# never be negative regardless of the sign of the underlying gradient.
	torch.manual_seed(0)
	model = ConvAvgDense(n_outputs=1)
	sal = calculate_saliency_map(model, X0.clone(), device=device)

	assert np.all(sal >= 0)


def test_calculate_saliency_map_does_not_mutate_input_values(X0, device):
	model = FlattenDense(seq_len=10, n_outputs=1)
	X_before = X0.clone()

	calculate_saliency_map(model, X0.clone(), device=device)
	assert_array_equal(X0, X_before)


def test_calculate_saliency_map_matches_closed_form_linear(device):
	# For a single Linear layer the gradient of the output w.r.t. the input
	# is exactly the weight matrix, independent of X. So the saliency at
	# each position is |W| at that position dotted with the one-hot column.
	torch.manual_seed(1)
	seq_len = 6
	model = FlattenDense(seq_len=seq_len, n_outputs=1)
	model.eval()

	X = random_one_hot((1, 4, seq_len), random_state=2).float()
	sal = calculate_saliency_map(model, X.clone(), device=device)

	W = model.dense.weight.detach().reshape(4, seq_len)
	expected = (W.abs() * X[0]).sum(dim=0).numpy()

	assert_array_almost_equal(sal, expected, 5)


def test_calculate_saliency_map_matches_autograd_nonlinear(device):
	# Ground-truth cross-check against a plain torch.autograd.grad call,
	# independent of anything internal to dmaps.py. Uses a model with a
	# nonlinearity (ReLU) so the gradient genuinely depends on X.
	torch.manual_seed(3)
	model = ConvAvgDense(n_outputs=1)
	model.eval()

	X = random_one_hot((1, 4, 15), random_state=4).float()
	sal = calculate_saliency_map(model, X.clone(), device=device)

	Xg = X.clone().to(device).requires_grad_()
	y = model.to(device)(Xg)
	g, = torch.autograd.grad(y, Xg, grad_outputs=torch.ones_like(y))
	expected = (g.detach().abs() * Xg.detach()).sum(dim=1).squeeze(0)
	expected = expected.cpu().numpy()

	assert_array_almost_equal(sal, expected, 5)


def test_calculate_saliency_map_zero_column_gives_zero_saliency(device):
	# An all-zero ("N") column contributes nothing to (grad * X), so its
	# saliency must be exactly zero no matter what the gradient is.
	torch.manual_seed(0)
	model = ConvAvgDense(n_outputs=1)
	X = random_one_hot((1, 4, 12), random_state=0).float()
	X[0, :, 5] = 0

	sal = calculate_saliency_map(model, X.clone(), device=device)
	assert sal[5] == 0


def test_calculate_saliency_map_rejects_non_floating_input(device):
	# one_hot_encode returns an int8 tensor; requires_grad_() only works on
	# floating-point tensors, so an un-cast one-hot input must fail loudly
	# rather than silently produce garbage.
	model = FlattenDense(seq_len=6, n_outputs=1)
	X = one_hot_encode('ACGTAC').unsqueeze(0)

	with pytest.raises(RuntimeError, match="floating point"):
		calculate_saliency_map(model, X, device=device)


def test_calculate_saliency_map_rejects_batch_size_greater_than_one(device):
	# The backward() call is hardcoded for a single-example, single-output
	# prediction; a batch of more than one example must fail rather than
	# silently return a saliency map for the wrong example.
	model = FlattenDense(seq_len=10, n_outputs=1)
	X = random_one_hot((2, 4, 10), random_state=0).float()

	with pytest.raises(RuntimeError):
		calculate_saliency_map(model, X, device=device)


###
# dependency_map
###


def test_dependency_map_shape_string_input(device):
	torch.manual_seed(0)
	seq = 'ACGTACGTAC'
	model = FlattenDense(seq_len=len(seq), n_outputs=1)

	dm = dependency_map(model, seq, device=device)
	assert isinstance(dm, np.ndarray)
	assert dm.shape == (len(seq), len(seq))


def test_dependency_map_shape_tensor_input(device):
	torch.manual_seed(0)
	model = FlattenDense(seq_len=10, n_outputs=1)
	X = random_one_hot((1, 4, 10), random_state=0).float()

	dm = dependency_map(model, X, device=device)
	assert isinstance(dm, np.ndarray)
	assert dm.shape == (10, 10)


def test_dependency_map_string_and_tensor_inputs_agree(device):
	# A string and its equivalent one-hot tensor describe the same sequence
	# and must produce numerically identical dependency maps.
	torch.manual_seed(0)
	seq = 'ACGTACGTAC'
	model = FlattenDense(seq_len=len(seq), n_outputs=1)

	dm_str = dependency_map(model, seq, device=device)

	X = one_hot_encode(seq).float().unsqueeze(0)
	dm_tensor = dependency_map(model, X, device=device)

	assert_array_almost_equal(dm_str, dm_tensor, 5)


def test_dependency_map_non_negative(device):
	torch.manual_seed(0)
	seq = 'ACGTACGTAC'
	model = ConvAvgDense(n_outputs=1)

	dm = dependency_map(model, seq, device=device)
	assert np.all(dm >= 0)


def test_dependency_map_deterministic(device):
	# No dropout/randomness is involved, so repeated calls on the same
	# (model, sequence) pair must be exactly reproducible.
	torch.manual_seed(0)
	seq = 'ACGTACGTAC'
	model = ConvAvgDense(n_outputs=1)
	model.eval()

	dm1 = dependency_map(model, seq, device=device)
	dm2 = dependency_map(model, seq, device=device)
	assert_array_almost_equal(dm1, dm2, 8)


def test_dependency_map_length_one(device):
	torch.manual_seed(0)
	model = FlattenDense(seq_len=1, n_outputs=1)

	dm = dependency_map(model, 'A', device=device)
	assert dm.shape == (1, 1)


def test_dependency_map_diagonal_only_for_linear_model(device):
	# A model with no nonlinearity between layers has a constant Jacobian:
	# the gradient at position j cannot depend on the base present at any
	# other position i. So mutating position i can only ever change the
	# saliency *at* position i -- the dependency map must be diagonal.
	torch.manual_seed(0)
	seq = 'ACGTACGTACGTACG'

	for model in (FlattenDense(seq_len=len(seq), n_outputs=1), _LinearConv()):
		dm = dependency_map(model, seq, device=device)
		off_diagonal = dm - np.diag(np.diag(dm))
		assert_array_almost_equal(off_diagonal, np.zeros_like(dm), 5)


def test_dependency_map_captures_nonlinear_interactions(device):
	# Conversely, a model with a real nonlinearity (ReLU) and a global
	# pooling step lets every position's gradient depend on every other
	# position's base, so the off-diagonal entries should NOT all vanish.
	torch.manual_seed(0)
	seq = 'ACGTACGTACGTACG'
	model = ConvAvgDense(n_outputs=1)

	dm = dependency_map(model, seq, device=device)
	off_diagonal = dm - np.diag(np.diag(dm))
	assert not np.allclose(off_diagonal, 0, atol=1e-6)


def test_dependency_map_matches_bruteforce(device):
	# Full validation of the internal algorithm: for every position i, the
	# column dependency_map[:, i] must equal the mean absolute difference,
	# over every alternate base (A/C/G/T minus the original, plus an 'N'
	# deletion), between the saliency map of the mutated sequence and that
	# of the original sequence.
	torch.manual_seed(0)
	seq = 'ACGTAC'
	model = FlattenDense(seq_len=len(seq), n_outputs=1)
	model.eval()

	dm = dependency_map(model, seq, device=device)

	alphabet = 'ACGT'
	X0 = one_hot_encode(seq).float().unsqueeze(0)
	base_saliency = calculate_saliency_map(model, X0.clone(), device=device)

	expected = np.zeros((len(seq), len(seq)))
	for i, original_base in enumerate(seq):
		diffs = []
		for mutant in ['A', 'T', 'C', 'G', 'N']:
			if mutant == original_base:
				continue

			X_mut = X0.clone()
			X_mut[0, :, i] = 0
			if mutant != 'N':
				X_mut[0, alphabet.index(mutant), i] = 1

			mutated_saliency = calculate_saliency_map(model, X_mut,
				device=device)
			diffs.append(np.abs(mutated_saliency - base_saliency))

		expected[:, i] = np.mean(diffs, axis=0)

	assert_array_almost_equal(dm, expected, 4)


def test_dependency_map_rejects_batch_size_greater_than_one():
	# dependency_map operates on a single sequence at a time; passing a
	# batch must fail with a clear error instead of silently mutating only
	# the first example while reporting a shape that implies otherwise.
	model = FlattenDense(seq_len=10, n_outputs=1)
	X = random_one_hot((2, 4, 10), random_state=0).float()

	with pytest.raises(ValueError, match="single"):
		dependency_map(model, X, device='cpu')


def test_dependency_map_verbose_progress_print(capsys):
	# Sanity check on the print-based progress reporting: the completion
	# message is always printed, and the periodic progress line fires once
	# the loop passes a multiple of 50 positions.
	torch.manual_seed(0)
	seq_len = 55
	model = FlattenDense(seq_len=seq_len, n_outputs=1)

	dependency_map(model, 'A' * seq_len, device='cpu')

	captured = capsys.readouterr()
	assert 'Processing mutation at position 50/55' in captured.out
	assert 'Dependency map generation complete.' in captured.out
