#!/bin/bash

cd ../common/tests/vhdl_tests/example_test
./run_example_test.sh

cd ../example_test2
./run_example_test2.sh

# at the last test combine all html pages
cd ..
rm vhdl_tests.html
for d in */*.html ; do
    cat "$d" >> vhdl_tests.html
done



