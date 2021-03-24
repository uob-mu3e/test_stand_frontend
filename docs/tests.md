# Tests

## VHDL Tests

All tests should be located in `common/tests/vhdl_tests`. An example test is placed in the folder example_test.
The needed software is [GHLD](https://github.com/ghdl/ghdl) and [VUnit](https://vunit.github.io/).

For using the tests on the build server bash scripts are used. Therefor each tests needs a run_TESTNAME.sh file like the example_test.
Also the test needs to be added to the `run_tests.sh` file. To test the automatic testing one needs to do

```
mkdir build
cd build
./../common/tests/vhdl_tests/run_tests.sh
```

Than a html file should be generated in the folder `common/tests/vhdl_tests/`. Therefor another python library called [json2html](https://pypi.org/project/json2html/) is needed.

## Midas/Software tests

All tests should be located in `common/tests/` they are build cmake and use [googletest](https://github.com/google/googletest) for testing. For running the test without hardware one needs to do:

```
mkdir build
cd build
cmake .. --DEBUG=1
make 
source set_env.sh
make install
cd -
make build_and_test
```

For using the test with hardware only do `cmake ..`. The individual tests are also installed and can be executed on there own.
