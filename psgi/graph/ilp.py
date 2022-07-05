"""Inductive Logic Programming (ILP) module implementation."""

from typing import Optional, Dict, List

import os
import numpy as np
import math
import time
from psgi.utils import log_utils

import dm_env
from acme import specs

import psgi
from psgi.utils import graph_utils
from psgi.utils.graph_utils import SubtaskGraph, dotdict, GraphVisualizer

DUMMY_ACTION = 0
MAX_GINI = 2.0

class ILP:

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      num_adapt_steps: int,
      branch_neccessary_first: bool = True,
      bonus_mode: int = 0,  # TODO: Change this to string
      visualize: bool = False,
      directory: str = 'visualize',
      environment_id: Optional[str] = None,
      verbose: bool = False,
      tr_epi: int = 20):  # TODO: replace this with iteration
    # Create ILP parameters.
    self._action_spec = environment_spec.actions
    self._total_timesteps = num_adapt_steps * 2 # since we store T+1 states for episode of length T, it can be up to * 2
    self._bonus_mode = bonus_mode
    self._verbose = verbose

    # TODO: Remove this.
    # these are shared among batch
    self._tr_epi = tr_epi
    self._environment_id = environment_id

    # Extra features
    ### duplication filtering
    self._duplication_filter_flag = True
    self._unique_indices_by_batch_ind = None # Initialized in reset()
    self._hash_table_by_batch = None # Initialized in reset()
    ### neccessary-first
    self._necessary_first = branch_neccessary_first


    # Create subtask graph visualizer.
    self._visualize = visualize
    if self._visualize:
      assert environment_id is not None, \
        'Please specify environment id for graph visualization.'
      self._graph_visualizer = GraphVisualizer()
      self._subtask_label = psgi.envs.get_subtask_label(environment_id)
      self._visualize_directory = directory

    # Count steps and trials.
    self._step_count = 0
    self._trial_count = 0
    self._graphs = None
    self._kmap_sets = None

  @property
  def graphs(self):
    return self._graphs

  @property
  def trajectory(self) -> List[Dict]:
    trajectory = []
    for i in range(self._batch_size):
      unique_indices = self._unique_indices_by_batch_ind[i]
      data = dict(
          reward_sum=self.reward_sum[i],
          reward_count=self.reward_count[i],
          step_count=self._step_count,
          completion=self._completion_buffer[unique_indices, i, :],
          eligibility=self._eligibility_buffer[unique_indices, i, :],
      )
      trajectory.append(data)
    return trajectory

  @property
  def kmaps(self):
    return self._kmap_sets

  def reset(self, environment: dm_env.Environment):
    # Reset/increment counts.
    self._batch_size = environment.batch_size
    self._step_count = 0
    self._trial_count += 1

    # Create ILP buffers.
    self.ntasks = max(environment.num_subtasks)  # XXX each env may have different # of subtasks
    self._completion_buffer = np.zeros((self._total_timesteps, self._batch_size, self.ntasks), dtype=np.bool)
    self._eligibility_buffer = np.zeros((self._total_timesteps, self._batch_size, self.ntasks), dtype=np.bool)
    self.reward_sum = np.zeros((self._batch_size, self.ntasks))
    self.reward_count = np.zeros((self._batch_size, self.ntasks), dtype=np.long)

    if self._visualize:
      # TODO: find a way to avoid using the setter below
      self._graph_visualizer.set_num_subtasks(self.ntasks)

    if self._bonus_mode > 0:
      self.hash_table = [set() for i in range(self._batch_size)]
      # TODO: Remove _tr_epi
      self.base_reward = min(10.0 / self._tr_epi / self.ntasks, 1.0)
      if self._bonus_mode == 2:
        self.pn_count = np.zeros((self._batch_size, self.ntasks, 2))

    # Reset from environment.
    self._pool_to_index = np.stack(environment.pool_to_index, axis=0)
    self._index_to_pool = np.stack(environment.index_to_pool, axis=0)

    # Get task index.
    #self._task_indices = [c.seed for c in environment.task] # XXX: not compatible with mining
    self._task_indices = [0] * self._batch_size

    # Reset hash table.
    if self._duplication_filter_flag:
      self._hash_table_by_batch = [set() for _ in range(self._batch_size)]
      self._unique_indices_by_batch_ind = [[] for _ in range(self._batch_size)]

  def insert(
      self,
      is_valid: np.ndarray,  # 1-dim (batch_size,)
      completion: np.ndarray,   # 2-dim (batch_size, num_subtasks)
      eligibility: np.ndarray,  # 2-dim (batch_size, num_subtasks)
      action_id: Optional[np.ndarray] = None,  # 2-dim (batch_size, 1)
      rewards: Optional[np.ndarray] = None      # 2-dim (batch_size, 1)
  ):
    """Store the experiences into ILP buffer."""

    # Get unique indices & update hash table
    if self._duplication_filter_flag:
      batch_codes = graph_utils.batch_bin_encode(completion.astype(np.bool))
      for batch_index in range(self._batch_size):
        code = batch_codes[batch_index]
        hash_table = self._hash_table_by_batch[batch_index]
        is_unique = code not in hash_table
        if is_unique:
          hash_table.add(code)
          self._unique_indices_by_batch_ind[batch_index].append(self._step_count)

    self._completion_buffer[self._step_count] = completion.copy()
    self._eligibility_buffer[self._step_count] = eligibility.copy()

    # reward
    if rewards is not None:
      active = np.expand_dims(is_valid.astype(dtype=np.int32), -1)
      assert active.ndim == 2 and active.shape[-1] == 1

      # For inactive batches, replace with DUMMY_ACTION to avoid error in _get_index_from_pool()
      action_id[active < 0.5] = DUMMY_ACTION
      act_inds = (self._get_index_from_pool(action_id) * active).astype(np.int32)

      mask = graph_utils.to_multi_hot(act_inds, self.ntasks).astype(dtype=np.int32) * active
      reward_mat = np.zeros_like(self.reward_sum)
      np.put_along_axis(reward_mat, act_inds, rewards, axis=1)

      self.reward_sum += reward_mat * (mask.astype(np.float))
      self.reward_count += mask.astype(np.long)

    # Increment step count.
    self._step_count += 1

  def infer_task(self) -> List[SubtaskGraph]:
    assert self._check_data(), "ilp data is invalid!"
    graphs, kmap_sets = [], []
    for i in range(self._batch_size):
      if self._duplication_filter_flag: # Filter-out duplicated result
        curr_idx = self._unique_indices_by_batch_ind[i]
      else:
        curr_idx = list(range(self._step_count))

      #0. initialize graph
      graph = SubtaskGraph(
          env_id=self._environment_id,
          task_index=self._task_indices[i],
          num_data=len(curr_idx),
          index_to_pool=self._index_to_pool[i],
          pool_to_index=self._pool_to_index[i])

      #1. Infer reward (prior reward is already merged in "update_prior()")
      graph.subtask_reward = self._infer_reward(self.reward_count[i], self.reward_sum[i]) #mean-reward-tracking

      #2. Infer precondition
      kmap_set, tind_by_layer, subtask_layer = self._infer_precondition(
          completion=self._completion_buffer[curr_idx, i, :],
          eligibility=self._eligibility_buffer[curr_idx, i, :]
      )
      if self._visualize and i == 0:
        dot = self._graph_visualizer.visualize(kmap_set, subtask_layer, self._subtask_label)
        filepath = f"{self._visualize_directory}/graph_step{self._step_count}"
        self._graph_visualizer.render_and_save(dot, path=filepath)

      #3. Fill-out graph edges from precondition kmap
      if len(tind_by_layer) > 1:
        matrices = self._kmap_to_mat(kmap_set, tind_by_layer)
      else:
        matrices = dict(
          W_a=[],
          W_o=[],
          ANDmat=np.zeros(shape=(0)),
          ORmat=np.zeros(shape=(0)),
          tind_by_layer=tind_by_layer,
        )

      graph.fill_edges(**matrices)
      kmap_sets.append(kmap_set)
      graphs.append(graph)
    self._graphs = graphs
    self._kmap_sets = kmap_sets
    return graphs

  def _check_data(self):
    for batch_index in range(self._batch_size):
      completion = self._completion_buffer[:self._step_count, batch_index, :]
      eligibility = self._eligibility_buffer[:self._step_count, batch_index, :]
      for i in range(self.ntasks):
        comp = np.concatenate((completion[:, :i], completion[:, i+1:]), axis=1)
        elig = eligibility[:, i]
        if not self._check_validity(comp, elig):
          return False
    return True

  def _infer_precondition(
      self,
      completion: np.ndarray,   # (T, num_subtasks), dtype=np.bool
      eligibility: np.ndarray):  # (T, num_subtasks), dtype=np.bool
    # 0. assertions
    assert completion.shape == eligibility.shape and completion.ndim == 2, "shape error"
    assert completion.dtype == eligibility.dtype and completion.dtype == np.bool, "completion, eligibility should be np.bool type"

    # 1. Find first layer: either always elig or never elig, while ignoring DC
    always_elig = np.all(eligibility, axis=0)
    never_elig = np.all(np.logical_not(eligibility), axis=0)
    first_layer_mask = always_elig | never_elig

    # Layer-wise ILP with prior
    curr_layer_ind_list = np.flatnonzero(first_layer_mask).tolist()
    tind_by_layer = [curr_layer_ind_list]
    cand_ind_list = curr_layer_ind_list.copy()
    subtask_layer = np.full((self.ntasks), fill_value=-1, dtype=np.int16)
    subtask_layer[first_layer_mask] = 0 # assign first layer

    kmap_set = [None] * self.ntasks
    for layer_ind in range(1, self.ntasks):
      # Inference done.
      if len(cand_ind_list) == self.ntasks:
        break

      curr_layer_ind_list = []
      comp = completion[:, cand_ind_list]
      _, unique_idx = np.unique(completion[:, cand_ind_list], return_index=True, axis=0)

      for ind in range(self.ntasks):
        if subtask_layer[ind] >= 0: # already assigned
          continue
        elig = eligibility[:, ind]

        # 1. check if curr is valid
        if self._check_validity(comp, elig): # NOTE: MUST use non-compact-ver for checking validity
          feat_dim = comp.shape[-1]
          mask = np.ones(feat_dim, dtype=np.bool)
          root = self.cart_train(mask, comp[unique_idx], elig[unique_idx]) # 88%
          kmap_tensor_org = self.decode_cart(root, feat_dim) # 0.5%
          kmap_tensor = self.simplify_Kmap(kmap_tensor_org) # 0.6%
          assert kmap_tensor.ndim == 2, "kmap_tensor should be 2 dimension"
          kmap_set[ind] = kmap_tensor
          #
          curr_layer_ind_list.append(ind)
          subtask_layer[ind] = layer_ind

      assert len(curr_layer_ind_list) > 0, f"Error: layer-wise ILP failed to find precondition for subtasks {np.nonzero(subtask_layer==-1)}"
      cand_ind_list.extend(curr_layer_ind_list)
      tind_by_layer.append(curr_layer_ind_list)

    return kmap_set, tind_by_layer, subtask_layer

  def _check_validity(self, inputs, targets):
    # inputs: np.arr((T, ntasks), dtype=bool) (only 0/1)
    # targets: np.arr((T,), dtype=bool) (only 0/1)
    assert inputs.dtype == np.bool and targets.dtype == np.bool, "type error"
    assert inputs.ndim == 2 and targets.ndim == 1, "shape errror"

    # check if there exists any i1 and i2 such that inputs[i1]==inputs[i2] and targets[i1]!=targets[i2]
    # if there exists, it means the node is not valid, and it should be in the higher layer in the graph.
    code_batch = np.asarray(graph_utils.batch_bin_encode(inputs))
    eligible_code_set = set(code_batch[targets])
    ineligible_code_set = set(code_batch[np.logical_not(targets)])
    return eligible_code_set.isdisjoint(ineligible_code_set)

  def _infer_reward(self, reward_count, reward_tensor): #. mean-reward
    # reward_count: Ntasks
    # reward_tensor: Ntasks
    subtask_reward = np.zeros(self.ntasks)
    nonzero_mask = reward_count > 0
    subtask_reward[nonzero_mask] = reward_tensor[nonzero_mask] / reward_count[nonzero_mask]
    rmean = subtask_reward[nonzero_mask].sum() / len(nonzero_mask)
    # Optimism in the face of uncertainty
    rmean = rmean + abs(rmean)*0.1 if abs(rmean) > 0. else rmean + 0.1
    #subtask_reward[np.logical_not(nonzero_mask)] = rmean
    subtask_reward[np.logical_not(nonzero_mask)] = 0.0  # default to zero

    return subtask_reward

  # CART functions. Don't touch unless necessary
  def decode_cart(self, root, nFeature):
    Kmap = []
    stack = []
    instance = dotdict()
    instance.lexp = np.zeros((nFeature), dtype=np.int8)
    instance.node = root
    stack.append(instance)

    while len(stack) > 0:
      node = stack[0].node
      lexp = stack[0].lexp
      stack.pop(0)
      featId = node.best_ind

      if node.gini > 0:  # leaf node && positive sample
        # TODO: add helpful assert message
        assert featId >= 0

        if node.left.best_ind >= 0:  # negation
          instance = dotdict()
          instance.lexp = lexp.copy()
          instance.lexp[featId] = -1  # negative
          instance.node = node.left
          stack.append(instance)

        if node.right.best_ind >= 0:  # positive
          instance = dotdict()
          instance.lexp = lexp.copy()
          instance.lexp[featId] = 1  # positive
          instance.node = node.right
          stack.append(instance)

      elif node.sign == 0:
        lexp[featId] = -1
        Kmap.append(lexp[None, :])
      else:
        lexp[featId] = 1
        Kmap.append(lexp[None, :])

    kmap_tensor = np.concatenate(Kmap, axis=0)

    return kmap_tensor

  def cart_train(self, mask, inputs, targets):
    assert inputs.dtype == np.bool and targets.dtype == np.bool, 'type error'
    assert inputs.ndim == 2, 'inputs should be 2 dim'
    assert inputs.shape[0] > 0, "Error: data is empty!"
    nstep, ncand = inputs.shape
    root = dotdict()
    minval = MAX_GINI + 0.1  # gini range: 0~2.0
    assert (mask.sum() > 0), 'Error: No feature left but classification is incomplete. Data is not separable'

    best_ind = None
    for i in range(ncand):
      if mask[i] == True:
        # Compute gini
        left, right = self.compute_gini(inputs[:, i], targets)
        gini = left.gini + right.gini
        # Test whether neccessary
        bias = 0.
        if self._necessary_first:
          test_input = np.concatenate((inputs[:, :i], inputs[:, i+1:]), axis=1) # form data without target subtask
          if not self._check_validity(test_input.astype(np.bool), targets.astype(np.bool)): # not valid without target subtask -> target subtask is neccessary!
            bias = -1 * (MAX_GINI + 0.1) # higher priority than unneccessary ones
        noise = 1e-5 * np.random.random() # to break ties

        if minval > gini + bias + noise and left.gini < 1.0 and right.gini < 1.0:
          minval = gini + bias + noise
          best_gini = gini
          best_ind = i
          best_left = left
          best_right = right
    assert best_ind is not None, 'Error! no feature has been chosen!'

    root.best_ind = best_ind
    root.gini = best_gini
    mask[best_ind] = False

    assert  best_left.gini < 1.0 and best_right.gini < 1.0, "Error! Left or Right branch is empty. It means best split has no gain"

    if best_gini > 0: # means somethings left for further branch expansion
      if best_left.gini > 0:  # means there exists both 0 and 1
        left_mask = inputs[:, best_ind] == 0
        left_input = inputs[left_mask, :]
        left_targets = targets[left_mask]
        root.left = self.cart_train(mask, left_input, left_targets)
      else:
        root.left = dotdict()
        root.left.gini = 0
        root.left.sign = best_left.p1
        root.left.best_ind = -1

      if best_right.gini > 0:
        right_mask = inputs[:, best_ind] != 0
        right_input = inputs[right_mask, :]
        right_targets = targets[right_mask]
        root.right = self.cart_train(mask, right_input, right_targets)
      else:
        root.right = dotdict()
        root.right.gini = 0
        root.right.sign = best_right.p1
        root.right.best_ind = -1
    else:
      root.sign = best_right.p1  # if right is all True,: sign=1

    mask[best_ind] = True
    return root

  def compute_gini(self, input_feat, targets):
    assert input_feat.dtype == np.bool and targets.dtype == np.bool, "type error!"
    neg_input = np.logical_not(input_feat)
    neg_target = np.logical_not(targets)
    nn_count = np.logical_and(neg_input, neg_target).sum()   # count[0]
    np_count = np.logical_and(neg_input, targets).sum()      # count[1]
    pn_count = np.logical_and(input_feat, neg_target).sum()  # count[2]
    pp_count = np.logical_and(input_feat, targets).sum()     # count[3]
    assert (nn_count + np_count + pn_count + pp_count == input_feat.shape[0]), "error in computing nn_count, np_count, pn_count, pp_count"

    left, right = dotdict(), dotdict()
    if nn_count + np_count > 0:
      p0_left = nn_count / (nn_count + np_count)
      p1_left = np_count / (nn_count + np_count)
      left.gini = 1 - pow(p0_left, 2) - pow(p1_left, 2)
      assert left.gini >= 0 and left.gini <= 0.5, "Error: Gini value is out of range [0, 0.5]"
      left.p0 = p0_left
      left.p1 = p1_left
    else:
      left.gini = 1
      left.p0 = 1
      left.p1 = 1

    if pn_count + pp_count > 0:
      p0_right = pn_count / (pn_count + pp_count)
      p1_right = pp_count / (pn_count + pp_count)
      right.gini = 1 - pow(p0_right, 2) - pow(p1_right, 2)
      assert right.gini >= 0 and right.gini <= 0.5, "Error: Gini value is out of range [0, 0.5]"
      right.p0 = p0_right
      right.p1 = p1_right
    else:
      right.gini = 1
      right.p0 = 1
      right.p1 = 1

    return left, right

  def _kmap_to_mat(self, kmap_set, tind_by_layer):
    W_a, W_o, cand_tind = [], [], []
    num_prev_or = 0
    numA_all = 0

    #1. fillout W_a/W_o
    for layer_ind in range(1, len(tind_by_layer)):
      num_prev_or = num_prev_or + len(tind_by_layer[layer_ind - 1])
      num_cur_or = len(tind_by_layer[layer_ind])
      W_a_layer, W_a_layer_padded = [], []
      cand_tind += tind_by_layer[layer_ind - 1]
      OR_table = [None] * self.ntasks
      numA = 0

      # fill out 'W_a_layer' and 'OR_table'
      for ind in tind_by_layer[layer_ind]:
        Kmap = kmap_set[ind]
        if Kmap is None:
          print('ind=', ind)
          print('tind_by_layer', tind_by_layer[layer_ind] )
          assert False, "Kmap should not be None"

        if len(Kmap) > 0:  # len(Kmap)==0 if no positive sample exists
          OR_table[ind] = []
          for j in range(Kmap.shape[0]):
            ANDnode = Kmap[j,:].astype(np.float)

            # see if duplicated
            duplicated_flag = False
            for row in range(numA):
              if np.all(np.equal(W_a_layer[row],ANDnode)):
                duplicated_flag = True
                and_node_index = row
                break

            if duplicated_flag == False:
              W_a_layer.append(ANDnode)
              cand_tind_tensor = np.array(cand_tind).astype(np.int32)
              assert cand_tind_tensor.shape[0] == ANDnode.shape[0]

              padded_ANDnode = np.zeros((self.ntasks))
              np.put_along_axis(padded_ANDnode, cand_tind_tensor, ANDnode, axis=0)
              #padded_ANDnode = np.zeros( (self.ntasks) ).scatter_(0, cand_tind_tensor, ANDnode)
              W_a_layer_padded.append(padded_ANDnode)
              OR_table[ind].append(numA) #add the last one
              numA = numA + 1
            else:
              OR_table[ind].append(and_node_index) #add the AND node

      if numA > 0:
        numA_all = numA_all + numA
        W_a_tensor = np.stack(W_a_layer_padded, axis=0)
        W_a.append(W_a_tensor)

      # fill out 'W_o_layer' from 'OR_table'
      W_o_layer = np.zeros((self.ntasks, numA))
      for ind in tind_by_layer[layer_ind]:
        OR_table_row = OR_table[ind]
        for j in range(len(OR_table_row)):
          and_node_index = OR_table_row[j]
          W_o_layer[ind][and_node_index] = 1

      W_o.append(W_o_layer)

    if self._verbose:
      print('W_a')
      for i in range(len(W_a)): print(W_a[i])
      print('W_o')
      for i in range(len(W_o)): print(W_o[i])

    #2. fillout ANDmat/ORmat
    assert len(W_a) > 0
    ANDmat = np.concatenate(W_a, axis=0)
    ORmat = np.concatenate(W_o, axis=1)

    if numA_all == 0 or self.ntasks == 0:
      print('self.ntasks=', self.ntasks)
      print('numA_all=', numA_all)
      print('kmap_set=', kmap_set)
      print('tind_by_layer=', tind_by_layer)
      # TODO: add helpful assert message
      assert False

    if self._verbose:
      print('Inference result:')
      print('ANDmat=', ANDmat)
      print('ORmat=', ORmat)
    matrices = dict(
      W_a=W_a,
      W_o=W_o,
      ANDmat=ANDmat,
      ORmat=ORmat,
      tind_by_layer=tind_by_layer,
    )
    return matrices

  def simplify_Kmap(self, kmap_tensor):
    """
      This function performs the following two reductions
      A + AB  -> A
      A + A'B -> A + B

      Kmap_bin: binarized Kmap. (i.e., +1 -> +1, 0 -> 0, -1 -> +1)
    """
    numAND = kmap_tensor.shape[0]
    mask = np.ones(numAND)
    max_iter = 20

    for jj in range(max_iter):
      done = True
      remove_list = []
      Kmap_bin = np.not_equal(kmap_tensor,0).astype(np.uint8)

      for i in range(numAND):
        if mask[i] == 1:
          kbin_i = Kmap_bin[i]

          for j in range(i + 1, numAND):
            if mask[j] == 1:
              kbin_j = Kmap_bin[j]
              kbin_mul = kbin_i * kbin_j

              if np.all(kbin_mul == kbin_i):  # i subsumes j. Either 1) remove j or 2) reduce j.
                done = False
                Kmap_common_j = kmap_tensor[j] * kbin_i  # common parts in j.
                difference_tensor = Kmap_common_j != kmap_tensor[i]  # (A,~B)!=(A,B) -> 'B'
                num_diff_bits = np.sum(difference_tensor)

                if num_diff_bits == 0:  # completely subsumes --> remove j.
                  mask[j] = 0
                else:  # turn off the different bits
                  dim_ind = np.nonzero(difference_tensor)[0]
                  kmap_tensor[j][dim_ind] = 0

              elif np.all(kbin_mul == kbin_j):  # j subsumes i. Either 1) remove i or 2) reduce i.
                done = False
                Kmap_common_i = kmap_tensor[i] * kbin_j
                difference_tensor = Kmap_common_i != kmap_tensor[j]
                num_diff_bits = np.sum(difference_tensor)

                if num_diff_bits == 0:  # completely subsumes--> remove i.
                  mask[i] = 0
                else:  # turn off the different bit.
                  dim_ind = np.nonzero(difference_tensor)[0]
                  kmap_tensor[i][dim_ind] = 0

      if done:
        break

    if mask.sum() < numAND:
      kmap_tensor = kmap_tensor[mask.nonzero()[0],:]
      #kmap_tensor = kmap_tensor.index_select(0,mask.nonzero().view(-1))

    return kmap_tensor

  # TODO: Remove all the dependencies to prev_active variable.
  def compute_bonus(
      self,
      prev_active: np.ndarray,
      step_done: np.ndarray,
      completion: np.ndarray,
      eligibility: np.ndarray):
    rewards = np.zeros(self._batch_size)

    if self._bonus_mode == 0:
      pass
    elif self._bonus_mode == 1:  # novel completion
      completion_code = graph_utils.batch_bin_encode(completion)

      for i in range(self._batch_size):
        if prev_active[i] == 1 and step_done[i] == 0:
          code = completion_code[i].item()
          if not code in self.hash_table[i]:
            rewards[i] = self.base_reward
            self.hash_table[i].add(code)
    elif self._bonus_mode == 2:  # novel completion & UCB weight ( branch: pos/neg )
      completion_code = graph_utils.batch_bin_encode(completion)

      for i in range(self._batch_size):
        if prev_active[i] == 1 and step_done[i] == 0:
          elig = eligibility[i]
          pn_count = self.pn_count[i] #referencing
          num_task = elig.shape[0]
          code = completion_code[i].item()

          if code not in self.hash_table[i]:
            self.hash_table[i].add(code)

            # 1. add shifted-ones
            shifted_ones = np.zeros_like(pn_count)
            np.put_along_axis(shifted_ones, np.expand_dims(elig, 1).astype(dtype=np.int32), 1, axis=1)
            pn_count += shifted_ones

            # 2. compute weight
            N_all = pn_count.sum(1)
            n_current = pn_count.gather(dim=1, index=np.expand_dims(elig.astype(dtype=np.int32), 1)).squeeze()
            #UCB_weight = (np.log(N_all)/n_current).sqrt().sum() / num_task
            UCB_weight = (25 / n_current).sqrt().sum() / num_task   # version 2 (removed log(N_all) since it will be the same for all subtasks)
            rewards[i] = self.base_reward * UCB_weight
            """ # slower version
            reward_test = 0.
            for ind in range(num_task):
              pn_ind = elig[ind].item() # 0: neg, 1: pos
              pn_count[ind][pn_ind] += 1
              N_all = pn_count[ind].sum()
              n_current = pn_count[ind][pn_ind]
              UCB_weight = math.sqrt(math.log(N_all)/n_current) / num_task
              reward_test += self.base_reward * UCB_weight"""

    elif self._bonus_mode == 3:
      # update in hash table. (if the value of updated element is different from the previous (presumed-dont-care) value, give reward)
      # TODO: update k-map every time (is too slow...)
      pass

    return np.expand_dims(rewards, 1)

  ### Util
  def _get_index_from_pool(self, input_ids):
    if input_ids.ndim == 1:
      input_ids = np.expand_dims(input_ids, -1)
    assert self.pool_to_index is not None
    return graph_utils.get_index_from_pool(
      pool_ids=input_ids,
      pool_to_index=self.pool_to_index,
    )

  @property
  def environment_id(self) -> str:
    return self._environment_id

  @property
  def pool_to_index(self) -> np.ndarray:
    return self._pool_to_index

  @property
  def index_to_pool(self) -> np.ndarray:
    return self._index_to_pool

  ####
  def save(self, filename):
    data = (
      self._completion_buffer,
      self._eligibility_buffer,
      self.reward_sum,
      self.reward_count,
      self._step_count,
    )
    np.save(filename, data)

  def load(self, filename):
    self._completion_buffer, \
    self._eligibility_buffer, \
    self.reward_sum, \
    self.reward_count, \
    self._step_count = np.load(filename, allow_pickle=True)
    assert self.ntasks == self.reward_count.shape[-1]
