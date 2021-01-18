#!/bin/bash
echo "Run this in the install folder"
cd ../common/tests/vhdl_tests/example_test
python run.py
python run.py --export-json example_test.json
cd -

