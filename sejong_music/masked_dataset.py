from .yeominrak_processing import AlignedScore


class MaskedTokenDataset(AlignedScore):
  def __init__(self) -> None:
    super().__init__()
    self.masked_tokens = []
    