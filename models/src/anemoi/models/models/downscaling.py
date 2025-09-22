# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from anemoi.models.models import AnemoiMultiModel


class AnemoiDownscalingModel(AnemoiMultiModel):
    name = "downscaling"


class ToyAnemoiDownscalingModel(AnemoiMultiModel):
    name = "downscaling"

    def forward(self, x, *args, **kwargs):
        print(f"----- Start of Forward pass of {self.__class__.__name__} -----")
        print(f"Ignoring args: {args}")
        print(f"Ignoring kwargs: {kwargs}")
        import einops

        print(self.sample_static_info.to_str("sample_static_info"))

        print(x.to_str("input x"))

        output = self.sample_static_info["target"].new_empty()
        for k, value in self.sample_static_info["target"].items():
            res = value.copy()
            module = self.linear[k]  # get the linear layer for this key
            data = x[k]["data"]  # get the input data for this key
            data = einops.rearrange(data, "1 a b -> 1 b a")
            res["data"] = module(data)  # apply the linear layer
            output[k] = res  # save in output for this key

        print(output.to_str("output after linear"))
        assert len(output), "ouput must not be empty"
        print("❤️🆗----- End of Forward pass of AnemoiMultiModel -----")
        return output
