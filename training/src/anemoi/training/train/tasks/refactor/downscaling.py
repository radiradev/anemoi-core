from anemoi.training.train.tasks.refactor.forecasting import ForecastingModule


class DownscalingModule(ForecastingModule):
    @property
    def model_class(self):
        from anemoi.models.models import AnemoiMultiModel

        return AnemoiMultiModel
