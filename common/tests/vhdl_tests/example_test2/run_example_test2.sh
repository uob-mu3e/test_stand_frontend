#!/bin/bash
echo "Run this in the install folder"

cd ../common/tests/vhdl_tests/example_test2
python3 run.py
python3 run.py --export-json example_test2.json
cd ..
python3 json_to_html.py example_test2.json example_test2

