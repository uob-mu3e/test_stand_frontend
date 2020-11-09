#!/bin/bash
echo "Run this in the install folder"
cd ../common/tests/vhdl_tests/example_test
python3 run.py
python3 run.py --export-json example_test.json
cd -

