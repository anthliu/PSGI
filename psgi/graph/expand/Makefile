.PHONY: test clean

# TODO: Use python's built-in build system (setup.py) for building native extensions.
TARGET_MODULE = expand_and_filter$(shell python3-config --extension-suffix)
INCLUDE_FLAG = $(shell python3 -m pybind11 --includes)

test: $(TARGET_MODULE)
	python -m mtsgi.graph.expand.expand_and_filter_test

$(TARGET_MODULE): expand_and_filter.cc
	c++ -O3 -Wall -shared -std=c++11 -fPIC $(INCLUDE_FLAG) $< -o $@

clean:
	rm -rf *.so
