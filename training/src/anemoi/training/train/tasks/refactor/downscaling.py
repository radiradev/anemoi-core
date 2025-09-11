from anemoi.training.train.tasks.refactor.forecasting import ForecastingModule


class DownscalingModule(ForecastingModule):
    @property
    def model_class(self):
        from anemoi.models.models import AnemoiDownscalingModel

        return AnemoiDownscalingModel

        # if we don't need to subclass the model, we can import AnemoiMultiModel directly
        # we could also add a config entry to choose the model class
        # from anemoi.models.models import AnemoiMultiModel
        # return AnemoiMultiModel
