SIM ?= icarus
.PHONY: test-fast test-slow test-py test-rtl lint clean
test-py:
	python3 -m pytest -q tests/py -m "not slow"
test-rtl:
	SIM=$(SIM) python3 -m pytest -q tests/rtl -m "not slow"
test-fast: test-py test-rtl
test-slow:
	SIM=$(SIM) python3 -m pytest -q -m slow
lint:
	verilator --lint-only -Wall -Wno-DECLFILENAME -Wno-UNUSEDPARAM rtl/*.sv
clean:
	rm -rf sim_build results.xml
