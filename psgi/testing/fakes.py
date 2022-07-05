"""Minimal implementation of fake Meta agent components."""

import acme
from acme import specs
from acme.testing import fakes
from psgi import agents
from psgi.agents import meta_agent

class FakeMetaAgent(meta_agent.MetaAgent):
  """Fake meta agent with fake actor and learner."""

  def __init__(self, spec: specs.EnvironmentSpec):
    self._spec = spec
    self.num_updates = 0

  def instantiate_adapt_agent(self) -> agents.Agent:
    self.agent = agents.Agent(
        actor=fakes.Actor(self._spec),
        learner=FakeLearner(),
        min_observations=1,
        observations_per_step=1,
    )
    return self.agent

  def instantiate_test_actor(self) -> acme.Actor:
    self.test_actor = fakes.Actor(self._spec)
    return self.test_actor

  def update(self):
    self.num_updates += 1

  def reset_agent(self, environment):
    """Reset the 'fast' agent upon samping a new task."""
    pass  # do nothing.


class FakeLearner(acme.Learner):
  """Fake learner which does nothing."""

  def __init__(self):
    self.num_steps = 0

  def step(self):
    self.num_steps += 1

  def get_variables(self):
    pass  # Do nothing
