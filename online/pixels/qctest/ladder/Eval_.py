import matplotlib.pyplot as plt
import numpy as np
import midas.client

"""
A simple example program that connects to a midas experiment,
reads an ODB value, then sets an ODB value.

Expected output is:

```
The experiment is currently stopped
The new value of /pyexample/eg_float is 5.670000
```
"""

if __name__ == "__main__":
    client = midas.client.MidasClient("pytest")

    # Read a value from the ODB. The return value is a normal python
    # type (an int in this case, but could also be a float, string, bool,
    # list or dict).
    state = client.odb_get("/Runinfo/State")

    if state == midas.STATE_RUNNING:
        print("The experiment is currently running")
    elif state == midas.STATE_PAUSED:
        print("The experiment is currently paused")
    elif state == midas.STATE_STOPPED:
        print("The experiment is currently stopped")
    else:
        print("The experiment is in an unexpected run state")

    
    '''USEFUL VARIABLES'''

    # Paths to MIDAS input/output variables obtained during ladder tests
    main_path = "/Equipment/Mupix/QCTests/Ladder"
    # List of chips
    chips = ['0', '1', '2', '3', '4', '5']
    # Maximum operational voltage of the ladder in Volts
    Max_operationalV =20
    # Overall test result: 0 on failed test, 1 on passed test
    TEST = 0
    # Test for IV
    test_IV =0
    # Test for LV
    test_LV=0
    # Test for VPDAC
    test_VPDAC=0
    # Test for ref_Vss
    test_refVss=0
    '''END OF USEFUL VARIABLES'''



    # Creates subdirectories for each chip where to print the output of the evaluation 
    main_directory = "Equipment/Mupix/QCTests/Ladder/Eval/"
    for chip in chips:
        directory = main_directory + chip + "/evaluation" 
        client.odb_set(directory, True)

    
    ''' LADDER TEST: characteristic I Vs V curve'''

    #BREAKDOWN VOLTAGE
    IVinput  = client.odb_get(main_path + "/IV/Input")
    IVoutput = client.odb_get(main_path + "/IV/Output")
    current=IVoutput[('Current')]
    voltage=IVoutput[('Voltage')]

    # Find breakdown voltage
    breakdownV = IVinput['stop_voltage']
    for i,v in zip(current,voltage):
        if i>=IVinput['current_limit']:
            breakdownV = v

    # Breakdown voltage test is passed 
    if breakdownV < Max_operationalV:
        test_IV=1
    print("TEST 1")
    print("Ladder breakdown voltage = ",breakdownV, "[V]. Did it pass the test ", test_IV )
    # plt.scatter(current, voltage)
    # plt.xlabel('Voltage [V]')
    # plt.ylabel('Current [A]')
    # plt.title('Characteristic curve of the Mupix ladder')
    # plt.show()


    '''INDIVIDUAL CHIP TEST'''

    # CHIP CONFIGURE TEST
    expected_increase =0.4
    chip_LV_test = np.ones(len(chips))
    current_increase = client.odb_get(main_path + "/LVPowerOn/Output/current_increase")
    for chip in range(len(chips)):
        if  current_increase[chip] < expected_increase:
            chip_LV_test[chip] = 0
    
    # The config test is passed 
    if np.sum(chip_LV_test)==len(chips):
        test_LV=1
    print("TEST 2")
    print("Chip current increase from chip configuration, test results:", chip_LV_test)
    


    # CHIP VPDAC AND REF_VSS scan
    chip_nr=0
    for chip in chips:

        VPDAC = client.odb_get(main_path + "/DACScan/Output/VPDAC/" + chip)
        VPDAC_values = VPDAC[('VPDAC_values')]
        VPDAC_current = VPDAC[('VPDAC_current')]

        ref_VSS = client.odb_get(main_path + "/DACScan/Output/ref_Vss/" + chip)
        ref_VSS_values = VPDAC[('ref_VSS_values')]
        ref_VSScurrent = VPDAC[('ref_VSS_current')]

        if chip_nr ==3:
            print(VPDAC_values)
            print(VPDAC_current)
            plt.scatter(VPDAC_values,VPDAC_current)
            plt.xlabel('Values ')
            plt.ylabel('Current [A]')
            plt.title('VPDAC scan ladder: chip nr: ')
            plt.show()
        chip_nr +=1





    

    
