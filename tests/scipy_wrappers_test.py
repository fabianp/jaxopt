# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import random
from jax import test_util as jtu
from jax import tree_util
import jax.numpy as jnp

from jaxopt import objective
from jaxopt import ScipyBoundedLeastSquares
from jaxopt import ScipyBoundedMinimize
from jaxopt import ScipyLeastSquares
from jaxopt import ScipyMinimize
from jaxopt import ScipyRootFinding
from jaxopt._src import scipy_wrappers
from jaxopt._src import test_util
from jaxopt._src.tree_util import tree_scalar_mul

import numpy as onp
import scipy as osp
from sklearn import datasets


class JnpToOnpTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()

    m, n = 10, 9
    self.x_split_indices = [4, 6, 9]
    self.x_reshape_shape = [-1, 2]
    self.y_split_indices = [6, 8]
    self.y_reshape_shape = [-1, 3]

    key = random.PRNGKey(0)
    key_x, key_w1, key_w2 = random.split(key, 3)
    self.x = random.normal(key_x, [m])
    self.w1 = random.normal(key_w1, [m, m])
    self.w2 = random.normal(key_w2, [m, n])

  @staticmethod
  def _do_pytree(array, split_indices, reshape_shape):
    """Creates a (hardcoded) PyTree out of flat 1D array for testing purposes."""
    pytree = jnp.split(array, split_indices)
    pytree[0] = pytree[0].reshape(reshape_shape)
    pytree[-1] = {'-1': pytree[-1]}
    return pytree

  @staticmethod
  def _undo_pytree(pytree):
    """Recovers flat 1D array from (hardcoded) PyTree for testing purposes."""
    pytree = pytree.copy()
    pytree[0] = pytree[0].reshape([-1])
    pytree[-1] = pytree[-1]['-1']
    return jnp.concatenate(pytree)

  def test_vals_and_jac(self):
    fun_flat = lambda x: jnp.dot(jnp.arctan(jnp.dot(x, self.w1)), self.w2)

    def fun(x_pytree):
      """Wraps fun_flat with (hardcoded) yTree input / output."""
      x = self._undo_pytree(x_pytree)
      y = fun_flat(x)
      return self._do_pytree(y, self.y_split_indices, self.y_reshape_shape)

    # Tests function output.
    x_pytree = self._do_pytree(self.x,
                               self.x_split_indices,
                               self.x_reshape_shape)
    y_pytree = fun(x_pytree)
    self.assertArraysAllClose(fun_flat(self.x), self._undo_pytree(y_pytree))

    # Tests jnp_to_onp.
    self.assertArraysAllClose(self._undo_pytree(y_pytree),
                              scipy_wrappers.jnp_to_onp(y_pytree))

    # Tests Jacobian.
    x_pytree_topology = scipy_wrappers.pytree_topology_from_example(x_pytree)
    y_pytree_topology = scipy_wrappers.pytree_topology_from_example(y_pytree)
    jac_jnp_to_onp = scipy_wrappers.make_jac_jnp_to_onp(x_pytree_topology,
                                                        y_pytree_topology)

    jac_flat = jax.jacrev(fun_flat)(self.x)
    jac_pytree = jax.jacrev(fun)(x_pytree)
    self.assertArraysAllClose(jac_flat, jac_jnp_to_onp(jac_pytree))


class ScipyMinimizeTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()

    self.n_samples, self.n_features, self.n_classes = 50, 5, 3
    self.data = datasets.make_classification(n_samples=self.n_samples,
                                             n_features=self.n_features,
                                             n_classes=self.n_classes,
                                             n_informative=3,
                                             random_state=0)
    self.default_l2reg = float(self.n_samples)

    self.solver_kwargs = {'method': 'L-BFGS-B',
                          'tol': 1e-3,
                          'options': {'maxiter': 500}}

    def logreg_fun(params, *args, **kwargs):
      params = tree_util.tree_leaves(params)
      return objective.l2_multiclass_logreg(params[0], *args, **kwargs)
    self.logreg_fun = logreg_fun

    def logreg_with_intercept_fun(params, *args, **kwargs):
      params = tree_util.tree_leaves(params)  # Relies on 'W' < 'b'!
      return objective.l2_multiclass_logreg_with_intercept(
          params, *args, **kwargs)
    self.logreg_with_intercept_fun = logreg_with_intercept_fun

  def _init_pytree(self, pytree_type: str):
    pytree_init = jnp.zeros([self.n_features, self.n_classes])
    if pytree_type == 'tuple':
      pytree_init = (pytree_init,)
    elif pytree_type == 'dict':
      pytree_init = {'W': pytree_init}
    return pytree_init

  def _init_pytree_with_intercept(self, pytree_type):
    pytree_init = (jnp.zeros([self.n_features, self.n_classes]),
                   jnp.zeros([self.n_classes]))
    if pytree_type == 'dict':
      pytree_init = {'W': pytree_init[0], 'b': pytree_init[1]}
    return pytree_init

  @parameterized.product(pytree_type=['array'])
  def test_logreg(self, pytree_type: str):
    lbfgs = ScipyMinimize(fun=self.logreg_fun,
                          **self.solver_kwargs)
    pytree_fit, _ = lbfgs.run(self._init_pytree(pytree_type),
                              l2reg=self.default_l2reg,
                              data=self.data)

    # Compare against sklearn.
    pytree_skl = test_util.logreg_skl(X=self.data[0],
                                      y=self.data[1],
                                      lam=self.default_l2reg,
                                      fit_intercept=False)
    for array_skl, array_fit in zip(tree_util.tree_leaves(pytree_skl),
                                    tree_util.tree_leaves(pytree_fit)):
      self.assertArraysAllClose(array_skl, array_fit, atol=1e-3)

  @parameterized.product(pytree_type=['tuple', 'dict'])
  def test_logreg_with_intercept(self, pytree_type: str):
    lbfgs = ScipyMinimize(fun=self.logreg_with_intercept_fun,
                          **self.solver_kwargs)
    pytree_fit, _ = lbfgs.run(self._init_pytree_with_intercept(pytree_type),
                              l2reg=self.default_l2reg,
                              data=self.data)

    # Compare against sklearn.
    pytree_skl = test_util.logreg_skl(X=self.data[0],
                                      y=self.data[1],
                                      lam=self.default_l2reg,
                                      fit_intercept=True)
    for array_skl, array_fit in zip(tree_util.tree_leaves(pytree_skl),
                                    tree_util.tree_leaves(pytree_fit)):
      self.assertArraysAllClose(array_skl, array_fit, atol=1e-3)

  @parameterized.product(pytree_type=['array'])
  def test_logreg_implicit_diff(self, pytree_type: str):
    # Make sure the decorator works, evaluating the Jacobian at sklearn's sol.
    pytree_skl = test_util.logreg_skl(X=self.data[0],
                                      y=self.data[1],
                                      lam=self.default_l2reg)
    lbfgs = ScipyMinimize(fun=self.logreg_fun,
                          implicit_diff=True,
                          **self.solver_kwargs)
    def wrapper(hyperparams):
      sol_skl = pytree_skl
      if pytree_type == 'tuple':
        sol_skl = (sol_skl,)
      elif pytree_type == 'dict':
        sol_skl = {'W': sol_skl}
      return lbfgs.run(sol_skl, hyperparams, self.data).params

    jac_num = test_util.logreg_skl_jac(X=self.data[0],
                                       y=self.data[1],
                                       lam=self.default_l2reg)
    jac_custom = jax.jacrev(wrapper)(self.default_l2reg)
    for array_num, array_custom in zip(tree_util.tree_leaves(jac_num),
                                       tree_util.tree_leaves(jac_custom)):
      self.assertArraysAllClose(array_num, array_custom, atol=1e-3)

  @parameterized.product(pytree_type=['tuple', 'dict'])
  def test_logreg_with_intercept_implicit_diff(self, pytree_type: str):
    # Make sure the decorator works, evaluating the Jacobian at sklearn's sol.
    pytree_skl = test_util.logreg_skl(X=self.data[0],
                                      y=self.data[1],
                                      lam=self.default_l2reg,
                                      fit_intercept=True)
    lbfgs = ScipyMinimize(fun=self.logreg_with_intercept_fun,
                          implicit_diff=True,
                          **self.solver_kwargs)
    def wrapper(hyperparams):
      sol_skl = pytree_skl
      if pytree_type == 'dict':
        sol_skl = {'W': sol_skl[0], 'b': sol_skl[1]}
      return lbfgs.run(sol_skl, hyperparams, self.data).params

    jac_num = test_util.logreg_skl_jac(X=self.data[0],
                                       y=self.data[1],
                                       lam=self.default_l2reg,
                                       fit_intercept=True)
    jac_custom = jax.jacrev(wrapper)(self.default_l2reg)
    for array_num, array_custom in zip(tree_util.tree_leaves(jac_num),
                                       tree_util.tree_leaves(jac_custom)):
      self.assertArraysAllClose(array_num, array_custom, atol=1e-3)


class ScipyBoundedMinimizeTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()

    self.n_samples, self.n_features = 50, 5
    self.data = datasets.make_regression(n_samples=self.n_samples,
                                         n_features=self.n_features,
                                         n_informative=3,
                                         random_state=0)
    self.pytree_init = jnp.zeros(self.n_features)
    self.pytree_bounds = (-jnp.ones(self.n_features),
                          +jnp.ones(self.n_features))

    def fun(params, data):
      params = self._undo_dict_pytree(params)
      return objective.least_squares(params, data)
    self.fun = fun

    self.solver_kwargs = {'method': 'L-BFGS-B',
                          'tol': 1e-3,
                          'options': {'maxiter': 500}}

  def _scipy_sol(self, init_params, **kwargs):
    return osp.optimize.lsq_linear(self.data[0], self.data[1], **kwargs).x

  def _scipy_box_sol(self, init_params, box_len, **kwargs):
    if 'bounds' not in kwargs:
      kwargs['bounds'] = self.pytree_bounds
    kwargs['bounds'] = (box_len * kwargs['bounds'][0],
                        box_len * kwargs['bounds'][1])
    return self._scipy_sol(init_params, **kwargs)

  def _scipy_box_jac(self, init_params, box_len, eps=1e-2, **kwargs):
    return (
        self._scipy_box_sol(init_params, box_len + eps, **kwargs) -
        self._scipy_box_sol(init_params, box_len - eps, **kwargs)) / (2. * eps)

  @staticmethod
  def _do_dict_pytree(pytree):
    return {'p0': pytree[0], 'p1': pytree[1:3], 'p2': pytree[3:]}

  @staticmethod
  def _undo_dict_pytree(pytree):
    # Relies on 'p0' < 'p1' < 'p2'!
    return jnp.hstack(tree_util.tree_leaves(pytree))

  @parameterized.product(pytree_type=['array', 'dict'])
  def test_fwd(self, pytree_type: str):
    pytree_init = self.pytree_init
    pytree_bounds = self.pytree_bounds
    if pytree_type == 'dict':
      pytree_init = self._do_dict_pytree(pytree_init)
      pytree_bounds = (self._do_dict_pytree(pytree_bounds[0]),
                       self._do_dict_pytree(pytree_bounds[1]))

    lbfgs = ScipyBoundedMinimize(fun=self.fun, **self.solver_kwargs)
    pytree_fit, _ = lbfgs.run(pytree_init,
                              bounds=pytree_bounds,
                              data=self.data)

    # Checks box constraints.
    for array_lb, array_fit, array_ub in zip(
        tree_util.tree_leaves(pytree_bounds[0]),
        tree_util.tree_leaves(pytree_fit),
        tree_util.tree_leaves(pytree_bounds[1])):
      self.assertTrue(jnp.all(array_lb <= array_fit).item())
      self.assertTrue(jnp.all(array_fit <= array_ub).item())

    # Compares objective values against SciPy.
    pytree_osp = self._scipy_sol(self.pytree_init, bounds=self.pytree_bounds)
    if pytree_type == 'dict':
      pytree_osp = self._do_dict_pytree(pytree_osp)

    for array_osp, array_fit in zip(tree_util.tree_leaves(pytree_osp),
                                    tree_util.tree_leaves(pytree_fit)):
      self.assertArraysAllClose(array_osp, array_fit, atol=1e-3)

  @parameterized.product(pytree_type=['dict'])
  def test_bwd_box_len(self, pytree_type: str):
    pytree_init = self.pytree_init
    pytree_bounds = self.pytree_bounds
    if pytree_type == 'dict':
      pytree_init = self._do_dict_pytree(pytree_init)
      pytree_bounds = (self._do_dict_pytree(pytree_bounds[0]),
                       self._do_dict_pytree(pytree_bounds[1]))

    lbfgs = ScipyBoundedMinimize(fun=self.fun,
                                 implicit_diff=True,
                                 **self.solver_kwargs)
    # NOTE: cannot use solution as init since changing box_len might make the
    # init infeasible.
    def wrapper(box_len):
      scaled_bounds = (tree_scalar_mul(box_len, pytree_bounds[0]),
                       tree_scalar_mul(box_len, pytree_bounds[1]))
      return lbfgs.run(pytree_init, scaled_bounds, self.data).params

    box_len = 10.0
    jac_num = self._scipy_box_jac(self.pytree_init,
                                  box_len,
                                  bounds=self.pytree_bounds)
    if pytree_type == 'dict':
      jac_num = self._do_dict_pytree(jac_num)
    jac_custom = jax.jacrev(wrapper)(box_len)
    for array_num, array_custom in zip(tree_util.tree_leaves(jac_num),
                                       tree_util.tree_leaves(jac_custom)):
      self.assertArraysAllClose(array_num, array_custom, atol=1e-2)


class ScipyRootFindingTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()

    n = 6
    key = random.PRNGKey(0)
    key_a, key_x = random.split(key)
    self.a = random.normal(key_a, [n, n])
    self.a1, self.a2 = self.a[:, :n // 2], self.a[:, n // 2:]
    self.x = random.normal(key_x, [n])
    self.x1, self.x2 = self.x[:n // 2], self.x[n // 2:]
    self.b = self.a.dot(self.x)
    self.b1, self.b2 = self.b[:n // 2], self.b[n // 2:]
    self.fun = lambda x, b: self.a.dot(x) - b
    def fun_pytree(x, b):
      x, treedef = tree_util.tree_flatten(x)  # Relies on 'x1' < 'x2'!
      b = tree_util.tree_leaves(b)  # Relies on 'b1' < 'b2'!
      o = self.a1.dot(x[0]) + self.a2.dot(x[1])
      o1, o2 = o[:n // 2], o[n // 2:]
      return tree_util.tree_unflatten(treedef, [o1 - b[0], o2 - b[1]])
    self.fun_pytree = fun_pytree

    self.solver_kwargs = {'method': 'hybr',
                          'implicit_diff': True}

  def _init_pytree_and_params_fun(self, pytree_type: str):
    pytree_init = (jnp.zeros_like(self.x1), jnp.zeros_like(self.x2))
    b = (self.b1, self.b2)
    if pytree_type == 'dict':
      pytree_init = {'x1': pytree_init[0], 'x2': pytree_init[1]}
      b = {'b1': b[0], 'b2': b[1]}
    return pytree_init, b

  def test_linalg_inv(self):
    root = ScipyRootFinding(optimality_fun=self.fun, **self.solver_kwargs)
    pytree_fit, _ = root.run(jnp.zeros_like(self.x), self.b)
    self.assertArraysAllClose(pytree_fit, self.x, atol=1e-3)

  def test_linalg_inv_idf(self):
    root = ScipyRootFinding(optimality_fun=self.fun, **self.solver_kwargs)
    def wrapper(b):
      return root.run(jnp.zeros_like(self.x), b).params

    jac_theo = jnp.linalg.inv(self.a)
    jac_idf = jax.jacrev(wrapper)(self.b)
    self.assertArraysAllClose(jac_theo, jac_idf, atol=1e-3)

  @parameterized.product(pytree_type=['tuple'])
  def test_linalg_inv_pytree(self, pytree_type: str):
    pytree_init, b = self._init_pytree_and_params_fun(pytree_type)
    root = ScipyRootFinding(optimality_fun=self.fun_pytree,
                            **self.solver_kwargs)
    pytree_fit, _ = root.run(pytree_init, b)
    for array_true, array_fit in zip(tree_util.tree_leaves((self.x1, self.x2)),
                                     tree_util.tree_leaves(pytree_fit)):
      self.assertArraysAllClose(array_true, array_fit, atol=1e-3)

  @parameterized.product(pytree_type=['dict'])
  def test_linalg_inv_pytree_idf(self, pytree_type: str):
    pytree_init, b = self._init_pytree_and_params_fun(pytree_type)
    root = ScipyRootFinding(optimality_fun=self.fun_pytree,
                            **self.solver_kwargs)
    def wrapper(b):
      return root.run(pytree_init, b).params

    jac_theo = jnp.linalg.inv(self.a)
    jac_idf = jax.jacrev(wrapper)(b)
    ## `jnp.block` requires inputs to be (nested) lists.
    if pytree_type == 'dict':
      jac_idf = [list(blk_row.values()) for blk_row in jac_idf.values()]
    else:
      jac_idf = [list(blk_row) for blk_row in jac_idf]
    jac_idf = jnp.block(jac_idf)
    self.assertArraysAllClose(jac_theo, jac_idf, atol=1e-3)


class ScipyLeastSquaresTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()

    self.slope = 2.5
    self.pytree_init = jnp.asarray([0.0, 0.0, 0.0, 0.0])
    self.pytree_true = jnp.asarray([1.5, -1.5, -0.25, 1.0])

    def model_fun(params, slope, x):
      """Implements a toy non-linear curve fitting problem on 2D.

        The model is defined as
          `f(theta; x) = offset + slope * sigmoid(w1 * x1 + w2 * x2 - t0)`
        where, for the purposes of testing, `theta = (w1, w2, t0, offset)` are
        treated as the parameters to be fit via least-squares and `slope` as a
        hyperparameter used to test implicit differentiation.

      Args:
        params: a pytree containing the parameters (w1, w2, t0, offset) of the
          model.
        slope: a float with the slope hyperparameter.
        x: a np.ndarray<float>[batch, 2] representing the covariates.

      Returns:
        a np.ndarray<float>[batch] with the model output for each row in x.
      """
      params = self._undo_dict_pytree(params)
      activation = jnp.dot(x, params[:2]) - params[2]
      return params[3] + slope * jax.nn.sigmoid(activation)

    def fun(params, slope, data):
      """Computes the residuals of `model_fun` above at `data`."""
      x, y_true = data
      return y_true - model_fun(params, slope, x)
    self.fun = fun

    n_samples = 25

    key = random.PRNGKey(0)
    x = random.normal(key, [n_samples, 2])
    y_true = model_fun(self.pytree_true, self.slope, x)
    self.data = (x, y_true)
    self.onp_data = tree_util.tree_map(lambda t: onp.asarray(t, onp.float64),
                                       self.data)

    self.solver_kwargs = {'method': 'trf'}

  @staticmethod
  def _do_dict_pytree(pytree):
    return {'p0': pytree[:2], 'p1': pytree[2], 'p2': pytree[3]}

  @staticmethod
  def _undo_dict_pytree(pytree):
    # Relies on 'p0' < 'p1' < 'p2'!
    return jnp.hstack(tree_util.tree_leaves(pytree))

  def _scipy_sol(self, init_params, slope, **kwargs):
    def scipy_fun(params):
      x, y_true = self.onp_data
      activation = onp.dot(x, params[:2]) - params[2]
      y_pred = params[3] + slope * onp.reciprocal(1.0 + onp.exp(-activation))
      return y_true - y_pred
    return osp.optimize.least_squares(scipy_fun, init_params, **kwargs).x

  def _scipy_slope_jac(self, init_params, slope, eps=1e-2, **kwargs):
    return (self._scipy_sol(init_params, slope + eps, **kwargs) -
            self._scipy_sol(init_params, slope - eps, **kwargs)) / (2. * eps)

  @parameterized.product(pytree_type=['array', 'dict'],
                         loss=['linear', 'arctan'],
                         f_scale=[0.2, 1.0])
  def test_fwd(self, pytree_type: str, loss: str, f_scale: float):
    pytree_true, pytree_init = self.pytree_true, self.pytree_init
    if pytree_type == 'dict':
      pytree_true = self._do_dict_pytree(pytree_true)
      pytree_init = self._do_dict_pytree(pytree_init)

    lsq = ScipyLeastSquares(fun=self.fun,
                            loss=loss,
                            f_scale=f_scale,
                            **self.solver_kwargs)
    pytree_fit, _ = lsq.run(pytree_init, self.slope, self.data)
    for array_true, array_fit in zip(tree_util.tree_leaves(pytree_true),
                                     tree_util.tree_leaves(pytree_fit)):
      self.assertArraysAllClose(array_true, array_fit, atol=1e-3)

  @parameterized.product(pytree_type=['dict'],
                         loss=['huber'],
                         f_scale=[0.2])
  def test_bwd(self, pytree_type: str, loss: str, f_scale: float):
    pytree_true = self.pytree_true
    if pytree_type == 'dict':
      pytree_true = self._do_dict_pytree(pytree_true)

    lsq = ScipyLeastSquares(fun=self.fun,
                            loss=loss,
                            f_scale=f_scale,
                            implicit_diff=True,
                            **self.solver_kwargs)
    def wrapper(slope):
      return lsq.run(pytree_true, slope, self.data).params

    jac_num = self._scipy_slope_jac(self.pytree_true, self.slope,
                                    **{'loss': loss, 'f_scale': f_scale})
    if pytree_type == 'dict':
      jac_num = self._do_dict_pytree(jac_num)
    jac_custom = jax.jacrev(wrapper)(self.slope)
    for array_num, array_custom in zip(tree_util.tree_leaves(jac_num),
                                       tree_util.tree_leaves(jac_custom)):
      self.assertArraysAllClose(array_num, array_custom, atol=1e-3)


class ScipyBoundedLeastSquaresTest(ScipyLeastSquaresTest):

  def setUp(self):
    super().setUp()

    pytree_lb = jnp.asarray([-1.0, -1.0, -1.0, -1.0])
    pytree_ub = jnp.asarray([+1.0, +1.0, +1.0, +1.0])
    self.pytree_bounds = (pytree_lb, pytree_ub)

  def _scipy_box_sol(self, init_params, box_len, **kwargs):
    if 'bounds' not in kwargs:
      kwargs['bounds'] = self.pytree_bounds
    kwargs['bounds'] = (box_len * kwargs['bounds'][0],
                        box_len * kwargs['bounds'][1])
    return self._scipy_sol(init_params, self.slope, **kwargs)

  def _scipy_box_jac(self, init_params, box_len, eps=1e-2, **kwargs):
    return (
        self._scipy_box_sol(init_params, box_len + eps, **kwargs) -
        self._scipy_box_sol(init_params, box_len - eps, **kwargs)) / (2. * eps)

  @parameterized.product(pytree_type=['array', 'dict'],
                         loss=['linear', 'cauchy'],
                         f_scale=[1.0])
  def test_fwd(self, pytree_type: str, loss: str, f_scale: float):
    pytree_init = self.pytree_init
    pytree_bounds = self.pytree_bounds
    if pytree_type == 'dict':
      pytree_init = self._do_dict_pytree(pytree_init)
      pytree_bounds = (self._do_dict_pytree(pytree_bounds[0]),
                       self._do_dict_pytree(pytree_bounds[1]))

    lsq = ScipyBoundedLeastSquares(fun=self.fun,
                                   loss=loss,
                                   f_scale=f_scale,
                                   **self.solver_kwargs)
    pytree_fit, _ = lsq.run(pytree_init, pytree_bounds, self.slope, self.data)

    # Checks box constraints.
    for array_lb, array_fit, array_ub in zip(
        tree_util.tree_leaves(pytree_bounds[0]),
        tree_util.tree_leaves(pytree_fit),
        tree_util.tree_leaves(pytree_bounds[1])):
      self.assertTrue(jnp.all(array_lb <= array_fit).item())
      self.assertTrue(jnp.all(array_fit <= array_ub).item())

    # Compares against SciPy.
    pytree_osp = self._scipy_sol(
        self.pytree_init,
        self.slope,
        **{'bounds': self.pytree_bounds, 'loss': loss, 'f_scale': f_scale})
    if pytree_type == 'dict':
      pytree_osp = self._do_dict_pytree(pytree_osp)

    for array_osp, array_fit in zip(tree_util.tree_leaves(pytree_osp),
                                    tree_util.tree_leaves(pytree_fit)):
      self.assertArraysAllClose(array_osp, array_fit, atol=1e-3)

  @parameterized.product(pytree_type=['dict'],
                         loss=['huber'],
                         f_scale=[1.0])
  def test_bwd_slope(self, pytree_type: str, loss: str, f_scale: float):
    sol_osp = self._scipy_sol(
        self.pytree_init,
        self.slope,
        **{'bounds': self.pytree_bounds, 'loss': loss, 'f_scale': f_scale})
    pytree_bounds = self.pytree_bounds
    pytree_osp = sol_osp
    if pytree_type == 'dict':
      pytree_bounds = (self._do_dict_pytree(pytree_bounds[0]),
                       self._do_dict_pytree(pytree_bounds[1]))
      pytree_osp = self._do_dict_pytree(pytree_osp)

    lsq = ScipyBoundedLeastSquares(fun=self.fun,
                                   loss=loss,
                                   f_scale=f_scale,
                                   implicit_diff=True,
                                   **self.solver_kwargs)
    def wrapper(slope):
      return lsq.run(pytree_osp, pytree_bounds, slope, self.data).params

    jac_num = self._scipy_slope_jac(
        sol_osp,
        self.slope,
        **{'bounds': self.pytree_bounds, 'loss': loss, 'f_scale': f_scale})
    if pytree_type == 'dict':
      jac_num = self._do_dict_pytree(jac_num)
    jac_custom = jax.jacrev(wrapper)(self.slope)
    for array_num, array_custom in zip(tree_util.tree_leaves(jac_num),
                                       tree_util.tree_leaves(jac_custom)):
      self.assertArraysAllClose(array_num, array_custom, atol=1e-2)

  @parameterized.product(pytree_type=['array'],
                         loss=['soft_l1'],
                         f_scale=[0.2])
  def test_bwd_box_len(self, pytree_type: str, loss: str, f_scale: float):
    pytree_init = self.pytree_init
    pytree_bounds = self.pytree_bounds
    if pytree_type == 'dict':
      pytree_init = self._do_dict_pytree(pytree_init)
      pytree_bounds = (self._do_dict_pytree(pytree_bounds[0]),
                       self._do_dict_pytree(pytree_bounds[1]))

    lsq = ScipyBoundedLeastSquares(fun=self.fun,
                                   loss=loss,
                                   f_scale=f_scale,
                                   implicit_diff=True,
                                   **self.solver_kwargs)
    # NOTE: cannot use solution as init since changing box_len might make the
    # init infeasible.
    def wrapper(box_len):
      scaled_bounds = (tree_scalar_mul(box_len, pytree_bounds[0]),
                       tree_scalar_mul(box_len, pytree_bounds[1]))
      return lsq.run(pytree_init, scaled_bounds, self.slope, self.data).params

    box_len = 1.0
    jac_num = self._scipy_box_jac(
        self.pytree_init,
        box_len,
        **{'bounds': self.pytree_bounds, 'loss': loss, 'f_scale': f_scale})
    if pytree_type == 'dict':
      jac_num = self._do_dict_pytree(jac_num)
    jac_custom = jax.jacrev(wrapper)(box_len)
    for array_num, array_custom in zip(tree_util.tree_leaves(jac_num),
                                       tree_util.tree_leaves(jac_custom)):
      self.assertArraysAllClose(array_num, array_custom, atol=1e-2)


if __name__ == '__main__':
  # Uncomment the line below in order to run in float64.
  # jax.config.update("jax_enable_x64", True)
  absltest.main(testLoader=jtu.JaxTestLoader())
