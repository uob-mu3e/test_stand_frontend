#!/bin/bash
echo "Run this in the install folder"

cd ../common/tests/vhdl_tests/example_test2
python3 run.py |& tee output.txt
python3 run.py --export-json example_test2.json
grep "All passed!" output.txt > /tmp/output.txt
cd ..
if [ -s /tmp/output.txt ]
then
	python3 json_to_html.py example_test2.json example_test2 0
else
 	python3 json_to_html.py example_test2.json example_test2 1
fi

