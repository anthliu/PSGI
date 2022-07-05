

class MazeConfig:

  def __repr__(self):
    return (f"<{type(self).__name__} at {hex(id(self))}, "
            f"{len(self.subtasks)} subtasks>")
