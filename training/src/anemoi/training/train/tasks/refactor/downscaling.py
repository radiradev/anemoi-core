import torch

from anemoi.training.train.tasks.refactor.forecasting import ForecastingPLModule


class DownscalingPLModule(ForecastingPLModule):
    def build_model(self, model_config, sample_static_info, metadata) -> torch.nn.Module:
        from anemoi.models.models.downscaling import AnemoiDownscalingModel

        return AnemoiDownscalingModel(
            model_config=model_config,
            sample_static_info=sample_static_info,
            metadata=metadata,
        )

        # if we don't need to subclass the model, we can import AnemoiMultiModel directly
        # we could also add a config entry to choose the model class
        # from anemoi.models.models import AnemoiMultiModel
        # return AnemoiMultiModel
