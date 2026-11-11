SIM ?= icarus
.PHONY: test-fast test-slow test-py test-rtl test-audit test-verilator-fast lint clean
test-py:
	python3 -m pytest -q tests/py -m "not slow"
test-rtl:
	SIM=$(SIM) python3 -m pytest -q tests/rtl -m "not slow"
test-fast: test-py test-rtl
# CI job verilator-fast (every push): the cocotb fast tier on Verilator, which adds what Icarus skips or
# keeps short (L = 16 VEC engine, top-level SAT/OREG/ch1-PBUF smoke tests with a decode step, ...)
test-verilator-fast:
	SIM=verilator python3 -m pytest -q tests/rtl -m "not slow"
test-audit:
	python3 -m pytest -q tests/audit -m "not slow" -rxX
test-slow:
	SIM=$(SIM) python3 -m pytest -q -m slow
lint:
	verilator --lint-only -Wall -Wno-DECLFILENAME -Wno-UNUSEDPARAM rtl/*.sv
clean:
	rm -rf sim_build results.xml
