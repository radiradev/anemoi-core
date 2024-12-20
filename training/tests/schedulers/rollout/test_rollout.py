# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from anemoi.training.schedulers.rollout import RolloutScheduler, Static

class DebugScheduler(RolloutScheduler):
    @property
    def rollout(self):
        return self._epoch
    
    @property
    def maximum_rollout(self):
        return self._epoch
    
    def description(self):
        return "DebugScheduler"

def test_static():
    sched = Static(1)
    assert sched.rollout == 1
    assert sched.maximum_rollout == 1
    assert sched.current_maximum == 1


def test_at():
    sched = DebugScheduler()

    with sched.at(epoch = 1):
        assert sched.rollout == 1
        assert sched.maximum_rollout == 1
        assert sched.current_maximum == 1
    
    assert sched.rollout == 0

def test_sync():
    sched = DebugScheduler()
    sched.sync(epoch = 10)
    assert sched.rollout == 10


def test_count():
    sched = DebugScheduler()
    sched.sync(epoch = 10)
    assert sched.count(n_epochs=5) == 2
    assert sched.count(n_epochs=3) == 3

def test_int_conversion():
    sched = DebugScheduler()
    sched.sync(epoch = 10)
    assert int(sched) == 10


