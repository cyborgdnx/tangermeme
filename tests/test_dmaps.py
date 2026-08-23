import torch
torch.use_deterministic_algorithms(True, warn_only=True)
torch.manual_seed(0)


import numpy
import pytest

from tangermeme.utils import one_hot_encode
from tangermeme.utils import random_one_hot

from tangermeme.dmaps import calculate_saliency_map
from tangermeme.dmaps import dependency_map

from .toy_models import FlattenDense
from .toy_models import ConvAvgDense
from .toy_models import ConvDense

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


@pytest.fixture
def X():
	return random_one_hot((8, 4, 10), random_state=0).float()


###
# calculate_saliency_map
###


def test_calculate_saliency_map_shape(X0, device):
	model = FlattenDense(seq_len=10, n_outputs=1)
	sal = calculate_saliency_map(model, X0.clone(), device=device)

	assert isinstance(sal, torch.Tensor)
	assert sal.shape == (1, 10)


def test_calculate_saliency_map_batch_shape(X, device):
	model = FlattenDense(seq_len=10, n_outputs=1)
	sal = calculate_saliency_map(model, X.clone(), device=device)

	assert isinstance(sal, torch.Tensor)
	assert sal.shape == (8, 10)


def test_calculate_saliency_map_non_negative(X, device):
	# The map is built from the absolute value of the gradient, so it can
	# never be negative regardless of the sign of the underlying gradient.
	torch.manual_seed(0)
	model = ConvAvgDense(n_outputs=1)
	sal = calculate_saliency_map(model, X.clone(), device=device)

	assert torch.all(sal >= 0)


def test_calculate_saliency_map_does_not_mutate_input_values(X, device):
	model = FlattenDense(seq_len=10, n_outputs=1)
	X_before = X.clone()

	calculate_saliency_map(model, X.clone(), device=device)
	assert_array_equal(X.numpy(), X_before.numpy())


def test_calculate_saliency_map_accepts_int8_input(device):
	# Unlike the original implementation, the dtype used for the forward /
	# backward pass is resolved internally (defaulting to the model's own
	# parameter dtype), the same as `predict`, so a low-precision int8 input
	# -- which cannot itself require grad -- must work without error.
	model = FlattenDense(seq_len=6, n_outputs=1)
	X = one_hot_encode('ACGTAC').unsqueeze(0)
	assert X.dtype == torch.int8

	sal = calculate_saliency_map(model, X, device=device)
	assert sal.shape == (1, 6)


def test_calculate_saliency_map_matches_closed_form_linear(device):
	# For a single Linear layer the gradient of the output w.r.t. the input
	# is exactly the weight matrix, independent of X. So the saliency at
	# each position is |W| at that position dotted with the one-hot column.
	torch.manual_seed(1)
	seq_len = 6
	model = FlattenDense(seq_len=seq_len, n_outputs=1)
	model.eval()

	X = random_one_hot((5, 4, seq_len), random_state=2).float()
	sal = calculate_saliency_map(model, X.clone(), device=device)

	W = model.dense.weight.detach().reshape(4, seq_len)
	expected = (W.abs().unsqueeze(0) * X).sum(dim=1).numpy()

	assert_array_almost_equal(sal.numpy(), expected, 5)


def test_calculate_saliency_map_matches_autograd_nonlinear(device):
	# Ground-truth cross-check against a plain torch.autograd.grad call,
	# independent of anything internal to dmaps.py. Uses a model with a
	# nonlinearity (ReLU) so the gradient genuinely depends on X, and a
	# batch of examples so that batched gradients are checked against their
	# per-example equivalents.
	torch.manual_seed(3)
	model = ConvAvgDense(n_outputs=1)
	model.eval()

	X = random_one_hot((6, 4, 15), random_state=4).float()
	sal = calculate_saliency_map(model, X.clone(), device=device)

	Xg = X.clone().to(device).requires_grad_()
	y = model.to(device)(Xg)
	g, = torch.autograd.grad(y, Xg, grad_outputs=torch.ones_like(y))
	expected = (g.detach().abs() * Xg.detach()).sum(dim=1)
	expected = expected.cpu().numpy()

	assert_array_almost_equal(sal.numpy(), expected, 5)


def test_calculate_saliency_map_zero_column_gives_zero_saliency(device):
	# An all-zero ("N") column contributes nothing to (grad * X), so its
	# saliency must be exactly zero no matter what the gradient is.
	torch.manual_seed(0)
	model = ConvAvgDense(n_outputs=1)
	X = random_one_hot((1, 4, 12), random_state=0).float()
	X[0, :, 5] = 0

	sal = calculate_saliency_map(model, X.clone(), device=device)
	assert sal[0, 5] == 0


def test_calculate_saliency_map_batching_is_batch_size_invariant(X, device):
	# The result for each example must not depend on which other examples
	# happen to share its batch -- this is the assumption the whole
	# per-example-gradient batching strategy rests on.
	torch.manual_seed(0)
	model = ConvAvgDense(n_outputs=1)

	sal_bs1 = calculate_saliency_map(model, X.clone(), batch_size=1,
		device=device)
	sal_bs3 = calculate_saliency_map(model, X.clone(), batch_size=3,
		device=device)
	sal_full = calculate_saliency_map(model, X.clone(), batch_size=64,
		device=device)

	assert_array_almost_equal(sal_bs1.numpy(), sal_bs3.numpy(), 5)
	assert_array_almost_equal(sal_bs1.numpy(), sal_full.numpy(), 5)


def test_calculate_saliency_map_target_selects_output(device):
	torch.manual_seed(0)
	model = FlattenDense(seq_len=10, n_outputs=3)
	X = random_one_hot((4, 4, 10), random_state=0).float()

	sal0 = calculate_saliency_map(model, X.clone(), target=0, device=device)
	sal1 = calculate_saliency_map(model, X.clone(), target=1, device=device)

	assert not torch.allclose(sal0, sal1)


def test_calculate_saliency_map_args_are_used(device):
	# FlattenDense.forward(X, alpha=0, beta=1) scales the (linear) output by
	# beta, which scales the gradient -- and so the saliency map -- by the
	# same constant factor.
	torch.manual_seed(0)
	model = FlattenDense(seq_len=10, n_outputs=1)
	X = random_one_hot((3, 4, 10), random_state=0).float()
	alpha = torch.zeros(3, 1)
	beta = torch.full((3, 1), 2.0)

	sal = calculate_saliency_map(model, X.clone(), args=[alpha, beta],
		device=device)
	sal_default = calculate_saliency_map(model, X.clone(), device=device)

	assert_array_almost_equal(sal.numpy(), (sal_default * 2).numpy(), 4)


def test_calculate_saliency_map_rejects_mismatched_args(device):
	model = FlattenDense(seq_len=10, n_outputs=1)
	X = random_one_hot((3, 4, 10), random_state=0).float()

	with pytest.raises(ValueError, match="same first"):
		calculate_saliency_map(model, X, args=[torch.zeros(2)], device=device)


def test_calculate_saliency_map_rejects_multi_output_model(device):
	# ConvDense returns a tuple of two tensors; calculate_saliency_map only
	# supports models with a single tensor output, the same constraint
	# `deep_lift_shap` documents.
	model = ConvDense(n_outputs=1)
	X = random_one_hot((2, 4, 100), random_state=0).float()

	with pytest.raises(ValueError, match="single tensor"):
		calculate_saliency_map(model, X, device=device)


def test_calculate_saliency_map_rejects_empty_batch(device):
	model = FlattenDense(seq_len=10, n_outputs=1)
	X = random_one_hot((0, 4, 10), random_state=0).float()

	with pytest.raises(ValueError, match="at least one example"):
		calculate_saliency_map(model, X, device=device)


###
# dependency_map
###


def test_dependency_map_shape_string_input(device):
	torch.manual_seed(0)
	seq = 'ACGTACGTAC'
	model = FlattenDense(seq_len=len(seq), n_outputs=1)

	dm = dependency_map(model, seq, device=device)
	assert isinstance(dm, torch.Tensor)
	assert dm.shape == (1, len(seq), len(seq))


def test_dependency_map_shape_tensor_input(device):
	torch.manual_seed(0)
	model = FlattenDense(seq_len=10, n_outputs=1)
	X = random_one_hot((1, 4, 10), random_state=0).float()

	dm = dependency_map(model, X, device=device)
	assert isinstance(dm, torch.Tensor)
	assert dm.shape == (1, 10, 10)


def test_dependency_map_shape_batch_tensor_input(device):
	torch.manual_seed(0)
	model = FlattenDense(seq_len=10, n_outputs=1)
	X = random_one_hot((5, 4, 10), random_state=0).float()

	dm = dependency_map(model, X, device=device)
	assert isinstance(dm, torch.Tensor)
	assert dm.shape == (5, 10, 10)


def test_dependency_map_string_and_tensor_inputs_agree(device):
	# A string and its equivalent one-hot tensor describe the same sequence
	# and must produce numerically identical dependency maps.
	torch.manual_seed(0)
	seq = 'ACGTACGTAC'
	model = FlattenDense(seq_len=len(seq), n_outputs=1)

	dm_str = dependency_map(model, seq, device=device)

	X = one_hot_encode(seq).float().unsqueeze(0)
	dm_tensor = dependency_map(model, X, device=device)

	assert_array_almost_equal(dm_str.numpy(), dm_tensor.numpy(), 5)


def test_dependency_map_batch_matches_looping_one_at_a_time(device):
	# Processing several sequences in one call must give the same result,
	# per sequence, as calling dependency_map once per sequence -- batching
	# across sequences is only a speed/memory optimization.
	torch.manual_seed(0)
	model = ConvAvgDense(n_outputs=1)
	X = random_one_hot((4, 4, 10), random_state=0).float()

	dm_batch = dependency_map(model, X.clone(), device=device)
	dm_looped = torch.cat([dependency_map(model, X[i:i+1].clone(),
		device=device) for i in range(X.shape[0])])

	assert_array_almost_equal(dm_batch.numpy(), dm_looped.numpy(), 4)


def test_dependency_map_non_negative(device):
	torch.manual_seed(0)
	seq = 'ACGTACGTAC'
	model = ConvAvgDense(n_outputs=1)

	dm = dependency_map(model, seq, device=device)
	assert torch.all(dm >= 0)


def test_dependency_map_deterministic(device):
	# No dropout/randomness is involved, so repeated calls on the same
	# (model, sequence) pair must be exactly reproducible.
	torch.manual_seed(0)
	seq = 'ACGTACGTAC'
	model = ConvAvgDense(n_outputs=1)
	model.eval()

	dm1 = dependency_map(model, seq, device=device)
	dm2 = dependency_map(model, seq, device=device)
	assert_array_almost_equal(dm1.numpy(), dm2.numpy(), 8)


def test_dependency_map_length_one(device):
	torch.manual_seed(0)
	model = FlattenDense(seq_len=1, n_outputs=1)

	dm = dependency_map(model, 'A', device=device)
	assert dm.shape == (1, 1, 1)


def test_dependency_map_diagonal_only_for_linear_model(device):
	# A model with no nonlinearity between layers has a constant Jacobian:
	# the gradient at position j cannot depend on the base present at any
	# other position i. So substituting position i can only ever change the
	# saliency *at* position i -- the dependency map must be diagonal.
	torch.manual_seed(0)
	seq = 'ACGTACGTACGTACG'

	for model in (FlattenDense(seq_len=len(seq), n_outputs=1), _LinearConv()):
		dm = dependency_map(model, seq, device=device)[0]
		off_diagonal = dm - torch.diag(torch.diag(dm))
		assert_array_almost_equal(off_diagonal.numpy(),
			numpy.zeros(dm.shape), 5)


def test_dependency_map_captures_nonlinear_interactions(device):
	# Conversely, a model with a real nonlinearity (ReLU) and a global
	# pooling step lets every position's gradient depend on every other
	# position's base, so the off-diagonal entries should NOT all vanish.
	torch.manual_seed(0)
	seq = 'ACGTACGTACGTACG'
	model = ConvAvgDense(n_outputs=1)

	dm = dependency_map(model, seq, device=device)[0]
	off_diagonal = dm - torch.diag(torch.diag(dm))
	assert not numpy.allclose(off_diagonal.numpy(), 0, atol=1e-6)


def test_dependency_map_matches_bruteforce(device):
	# Full validation of the internal algorithm: for every position i, the
	# column dependency_map[0, :, i] must equal the mean absolute difference,
	# over every one of the 4 possible bases at position i (including the
	# base already there, whose difference is exactly zero), between the
	# saliency map of the substituted sequence and that of the original.
	torch.manual_seed(0)
	seq = 'ACGTAC'
	model = FlattenDense(seq_len=len(seq), n_outputs=1)
	model.eval()

	dm = dependency_map(model, seq, device=device)[0]

	alphabet = 'ACGT'
	X0 = one_hot_encode(seq).float().unsqueeze(0)
	base_saliency = calculate_saliency_map(model, X0.clone(),
		device=device)[0]

	expected = numpy.zeros((len(seq), len(seq)))
	for i in range(len(seq)):
		diffs = []
		for base in alphabet:
			X_mut = X0.clone()
			X_mut[0, :, i] = 0
			X_mut[0, alphabet.index(base), i] = 1

			mutated_saliency = calculate_saliency_map(model, X_mut,
				device=device)[0]
			diffs.append((mutated_saliency - base_saliency).abs().numpy())

		expected[:, i] = numpy.mean(diffs, axis=0)

	assert_array_almost_equal(dm.numpy(), expected, 4)


def test_dependency_map_args_are_used(device):
	torch.manual_seed(0)
	seq = 'ACGTAC'
	model = FlattenDense(seq_len=len(seq), n_outputs=1)

	dm_default = dependency_map(model, seq, device=device)
	dm_scaled = dependency_map(model, seq, args=[torch.zeros(1, 1),
		torch.full((1, 1), 2.0)], device=device)

	assert_array_almost_equal(dm_scaled.numpy(), (dm_default * 2).numpy(), 4)


def test_dependency_map_rejects_empty_batch(device):
	model = FlattenDense(seq_len=10, n_outputs=1)
	X = random_one_hot((0, 4, 10), random_state=0).float()

	with pytest.raises(ValueError, match="at least one example"):
		dependency_map(model, X, device=device)


def test_dependency_map_verbose_does_not_error(device):
	torch.manual_seed(0)
	model = FlattenDense(seq_len=6, n_outputs=1)
	dependency_map(model, 'ACGTAC', device=device, verbose=True)