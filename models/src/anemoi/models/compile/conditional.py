# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import functools
import logging

import torch

LOG = logging.getLogger(__name__)


class ConditionalCompile:
    def __init__(self, method):
        self.method = method
        self.compiled_methods = {}

    def __get__(self, instance, owner):
        if instance is None:
            return self.method

        # Return a wrapper that chooses compiled or not
        @functools.wraps(self.method)
        def wrapper(*args, **kwargs):
            if instance.compile:
                if instance not in self.compiled_methods:
                    LOG.debug(f"Compiling {instance.__class__.__name__} using ConditionalCompile")
                    self.compiled_methods[instance] = torch.compile(self.method.__get__(instance, owner))
                return self.compiled_methods[instance](*args, **kwargs)
            else:
                return self.method.__get__(instance, owner)(*args, **kwargs)

        return wrapper
