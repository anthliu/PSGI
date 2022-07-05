from typing import Optional

from psgi import envs


class ToyWoBEnv(envs.BaseWoBEnv):
  '''Implementation of ToyWoB environment.
  '''
  environment_id = 'toywob'

  def __init__(self, rank: Optional[int] = None):
    super().__init__(config_factory=envs.ToyWoB, rank=rank)

  def _check_environment_done(self, action: str):
    """Checks if the action completes any of the terminal subtasks."""
    return action in self.config.terminal_subtasks

  def _update_eligibility(self, completion: dict):
    eligibility = self.graph.compute_eligibility(completion=completion)
    return eligibility

  def _update_mask(self, completion: dict):
    mask = {k: not v for k, v in completion.items()}
    return mask

  def _update_completion(self, action: str):
    next_completion = self.completion.copy()

    # TODO: Can we remove this check?
    if action in next_completion:
      next_completion[action] = True

    # Process extra outcome
    if action in self.config.option_extra_outcome:
      option_outcome = self.config.option_extra_outcome[action]
      for subtask, val in option_outcome.items():
        next_completion[subtask] = val
    return next_completion
