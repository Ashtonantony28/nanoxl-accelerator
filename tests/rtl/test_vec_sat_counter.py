"""pytest wrapper: rtl/sat_counter.sv (narrow width so saturation is reachable)."""
from simrun import run


def test_vec_sat_counter():
    run("sat_counter", ["rtl/sat_counter.sv"], "sat_counter_cocotb", parameters={"N": 4, "W": 6},
        extra_env={"N": "4", "W": "6"})
