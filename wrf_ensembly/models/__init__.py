from typing import TYPE_CHECKING

from .base import Model, ModelRun

if TYPE_CHECKING:
    from wrf_ensembly.experiment import Experiment


def create(exp: "Experiment") -> Model:
    """The model of an experiment, by `metadata.model`"""

    # Imported here: each model pulls in what only it needs
    if exp.cfg.metadata.model == "icon":
        from .icon import IconModel

        return IconModel(exp)

    from .wrf import WrfModel

    return WrfModel(exp)


__all__ = ["Model", "ModelRun", "create"]
