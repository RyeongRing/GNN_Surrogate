"""Fail-closed loading of complete, trusted ensemble checkpoints."""

from pathlib import Path


def load_members(model, directory, device):
    """Load every member or raise before evaluating an incomplete ensemble.

    Checkpoints use pickle; only load files from a trusted source.
    """
    import torch

    directory = Path(directory)
    paths = [directory / f"member_{i:02d}.pt" for i in range(len(model.members))]
    if not paths:
        raise ValueError("An ensemble must have at least one member")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Incomplete ensemble; missing: " + ", ".join(missing))
    for member, path in zip(model.members, paths):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        state = checkpoint.get("model_state", checkpoint)
        member.load_state_dict(state, strict=True)
    model.eval()
    return model
