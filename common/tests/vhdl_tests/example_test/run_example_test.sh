#!/bin/bash
echo "Run this in the install folder"

cd ../common/tests/vhdl_tests/example_test
python3 run.py
python3 run.py --export-json example_test.json
cd ..
python3 json_to_html.py example_test.json example_test

