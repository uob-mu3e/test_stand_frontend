
// Variables
const mupixState = {
	ON: 5,
	OFF: 0
}

const channelState = {
	ON: 'y',
	OFF: 'n'
}


//**************************************************//
// UTILS CONTROLS
//**************************************************//

/**
 * Function to pause execution for a specific amount of tme
 * @author Luigi
 * @param {int} ms time to pause execution
 */
 function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}


//**************************************************//
// POWER CONTROLS
//**************************************************//

//// HIGH VOLTAGE POWER SUPPLIER CONTROLS

/**
 * Function to set the output voltage in the HV supplier (Keithley 2611B)
 * @author Luigi, Pepe
 */
async function set_hv_voltage(volt) {
	try{
		mjsonrpc_db_paste(["/Equipment/KEITHLEY0/Variables/Demand Voltage[0]"], [volt]);
		return true;
	} catch (err) {
		mjsonrpc_error_alert(err);
		console.log("Could not set output voltage in HV supplier:", err)
		return false;
	}
}


/**
 * Function to get the actual voltage output in HV supplier (Keithley 2611B)
 * @author Pepe
 * @returns {float} Returns actual voltage of HV supplier in Amps
 */
 async function get_hv_voltage(){
	try{
		let result_rpc = await mjsonrpc_db_get_values(["/Equipment/KEITHLEY0/Variables/Voltage[0]]"]);
		return result_rpc.result.data[0];
	}catch (err){
		mjsonrpc_error_alert(err);
		console.log("Could not get hv voltage from ODB:", err)
		return 0;
	}
}


/**
 * Function to get actual current output in HV supplier (Keithley 2611B)
 * @author Luigi, Pepe
 * @returns {float} Returns actual current of HV supplier in Amps
 */
async function get_hv_current(){
	try{
		let result_rpc = await mjsonrpc_db_get_values(["/Equipment/KEITHLEY0/Variables/Current[0]]"]);
		return result_rpc.result.data[0];
	}catch (err){
		mjsonrpc_error_alert(err);
		console.log("Could not get hv current from ODB:", err)
		return 0;
	}
}


/**
 * Function to control the output state of the LV power supply (Rhode & Schwarz HMP 4040)
 * @author Pepe
 * @param {int} channel Channel to be activated or deactivated. 
 * @param {int} state Accepts only channelState.ON or channelState.OFF
 */
 async function hv_change_channel_output_state(state){
	if (state != channelState.ON && state != channelState.OFF) {
		mjsonrpc_error_alert("Trying to change channels state with an invalid petition. ON: y. OFF: n")
		state = channelState.OFF;
	}

	try {
		await mjsonrpc_db_paste(["/Equipment/HAMEG0/Variables/Set State"], [state]);
		return true;
	} catch (err) {
		mjsonrpc_error_alert(err);
		console.log("Could not change channels state in HV supplier:", err);
		return false;
	}
}


//// LOW VOLTAGE POWER SUPPLIER CONTROLS

/**
 * Function to set the output voltage in the LV supplier (Rohde & Schwarz HMP 4040)
 * @param {int} channel Channel output from 1 to 4: 1, 2, 3, 4
 * @param {float} volt Desired demanded output voltage in Volts
 * @author Pepe
 */
 async function set_lv_voltage(channel, volt) {
	try{
		mjsonrpc_db_paste(["/Equipment/HAMEG0/Variables/Demand Voltage["+(channel-1).toString()+"]"], [volt]);
		return true;
	} catch (err) {
		mjsonrpc_error_alert(err);
		console.log("Could not set output voltage in HV supplier:", err)
		return false;
	}
}


/**
 * Function to get actual current output of LV supplier (Rohde & Schwarz HMP4040)
 * 
 * Function should be called using the await keyword in front. For example:
 *	alert (await get_lv_current(4)) --> this will display an alert dialog with the current 
 * 										output from channel 4. 
 * 
 * @author Luigi, Pepe
 * @param {int} channel Desired channel output 1 to 4: 1, 2, 3, 4
 * @returns {float} Returns actual current of LV supplier in Amps
 */
async function get_lv_current(channel) {
	try {
		let result_rpc = await mjsonrpc_db_get_values(["/Equipment/HAMEG0/Variables/Current[" + (channel-1).toString() + "]"]);
		return result_rpc.result.data[0];
	}catch (err){
		mjsonrpc_error_alert(err);
		console.log("Could not get lv current from ODB:", err)
		return 0;
	}	
}


/**
 * Function to control the output state of the LV power supply (Rhode & Schwarz HMP 4040)
 * @author Pepe
 * @param {int} channel Channel to be activated or deactivated. 
 * @param {int} state Accepts only channelState.ON or channelState.OFF
 */
async function lv_change_channel_output_state(channel, state){
	if (state != channelState.ON && state != channelState.OFF) {
		mjsonrpc_error_alert("Trying to change channel "+channel.toString()+" with an invalid petition. ON: y. OFF: n")
		tate = channelState.OFF;
	}

	try {
		await mjsonrpc_db_paste(["/Equipment/HAMEG0/Variables/Set State["+(channel-1).toString()+"]"], [state]);
		return true;
	} catch (err) {
		mjsonrpc_error_alert(err);
		console.log("Could not change channel "+channel.toString()+" state:", err);
		return false;
	}
}


//**************************************************//
// MUPIX CONTROLS
//**************************************************//

/**
 * Function to turn on or off the chip. TODO TRY OUT
 * @author Pepe
 * @param {int} number Chip number to configure. Input 999 to configure all available
 * @param {int: state} nState Can only input either mupixState.ON or mupixState.OFF 
 */
 async function turnOnOffMupix(number, nState){
	if (nState != mupixState.ON && nState != mupixState.OFF){
		mjsonrpc_error_alert("Trying to turn on/off mupix sensor with an invalid petition. ON: 5. OFF: 0")
		nState = mupixState.OFF;
	}

	try {
		// Checked that these three petitions will execute in sequence, waiting for the other to finish. 
		await mjsonrpc_db_paste("/Equipment/Mupix/Settings/BIASDACS/"+number.toString()+"/BiasBlock_on", nState)
		await mjsonrpc_db_paste("/Equipment/Switching/Settings/MupixChipToConfigure", number);
		await mjsonrpc_db_paste("/Equipment/Switching/Settings/MupixConfig", "y")
	} catch (err) {
		mjsonrpc_error_alert("Could not configure chip:", err)
	}
}


/**
 * Function to load the current configuration in the ODB onto the chip. 
 * @author Pepe
 * @param {int} number Chip number to load configuration to. Input 999 to configure all available
 */
async function loadConfigToPixel(number){
	turnOnOff(number, mupixState.ON);
}
 

//**************************************************//
// TEST DEFINITIONS
//**************************************************//


//// IV CURVE TEST

/**
 * Second version of lv_iv_curve using for loops and taking into account update response time
 */
async function lv_iv_curve2(channel, start_voltage, final_voltage, step, whenToIncrementRate, final_step, time_step, current_limit){
	//Variables taken from GUI
	//let start_voltage = 0; // BEWARE OF THE SIGNS. Input will be negative --> work will be done with positive values --> final voltage input in keithley will be negative
	//let final_voltage = 10; // Needs to be device specific. Can be changed, but should be set as default option after changing between devices
	//let step = 1; // this will result in final_voltage/step+1 iterations, unless stopped by current limit or user
	//let final_step = 1; // step for final round
	//let time_step = 2; // seconds between operation
	//let current_limit = 0.015; //in amperes

	/* 
	* Helper function to make arrays between start and stop values with a separation of step between its elements. 
	* If the step is not chosen so that it ends on stopValue, it will output until the last possible value. Example:
	* makeArr(0, 10, 7) --: [0, 7]
	*/
	function makeArr(startValue, stopValue, step) {
		let arr = [startValue]
		for (let i = 0; arr[i]+step <= stopValue; i++){
			arr.push(startValue+step*(i+1))
		}

		return arr
	}

	/** 
	* Function that inputs the interval at which you want to decrease the step and returns all volts of the test, included the step 
	* @param {int} whenToIncrementRate is interval at which you want to decrease the step in voltage supplied to chip with HV supplier
	*/
	function getRun(whenToIncrementRate) { 
		// Generate initial run
		let runs = makeArr(start_voltage, final_voltage, step);
		// Extract info of desired last interval
		let fin_int = runs.slice(-(whenToIncrementRate+1));
		runs = runs.slice(0, -(whenToIncrementRate+1));
		// Calculate last interval
		fin_int = makeArr(fin_int[0], fin_int[fin_int.length-1], final_step)
		// Add last interval
		runs.push(fin_int)
		// Flatten
		runs = runs.flat()
		// Change sign
		/*runs.forEach(function(e, index){
			runs[index] *= -1;
		}, array);*/

		return runs;
	}

	let curve = [];
	let voltage_run = getRun(whenToIncrementRate);
	let curr;

	for (let j = 0; j < voltage_run.length; j++){
		// Set voltage and return curve if this fails
		if(!(await set_lv_voltage(channel, voltage_run[j]))){
			return curve;
		}
		
		//Sleep
		await sleep(time_step*1e3)

		// Get current read from power supple
		curr = await get_lv_current(channel);
		// Save result
		curve.push([voltage_run[j], curr]);
		// Check current limit
		if (curr >= current_limit){ //TODO check units
			dlgAlert('Current limit exceeded')
			return curve;
		}
	}

	console.log(curve)
	return curve;
}

/**
 * Function to perform the iv_curve test on the ladder. See function comments for more details. Remember to turn on relevant channels
 * before calling iv_curve
 * 
 * @author Pepe, Luigi
 * @param {int} start_voltage Start voltage in Volts
 * @param {int} final_voltage Target voltage in Volts
 * @param {int} step Step in initial (fast) stage in Volts
 * @param {int} final_step Step in final (slow) stage in Volts
 * @param {int} time_step Time step between runs in secs.
 * @param {int} current_limit Current limit in Amps
 * @returns {array} Returns an array of 2-vector elements. First element corresponds to input voltage 
 * 					whereas second element corresponds to measured current. Output array will be of 
 * 					desired length unless current limit is reached
 */
async function lv_iv_curve(){
	let aux, curve = [], i = 0; // control through this variable --> control through config?

	//Variables taken from GUI
	let start_voltage = 0; // BEWARE OF THE SIGNS. Input will be negative --> work will be done with positive values --> final voltage input in keithley will be negative
	let final_voltage = 10; // Needs to be device specific. Can be changed, but should be set as default option after changing between devices
	let step = 1; // this will result in final_voltage/step+1 iterations, unless stopped by current limit or user
	let final_step = 1; // step for final round
	let time_step = 2; // seconds between operation
	let current_limit = 0.015; //in amperes

	/* 
	* Helper function to make arrays between start and stop values with a separation of step between its elements. 
	* If the step is not chosen so that it ends on stopValue, it will output until the last possible value. Example:
	* makeArr(0, 10, 7) --: [0, 7]
	*/
	function makeArr(startValue, stopValue, step) {
		let arr = [startValue]
		for (let i = 0; arr[i]+step <= stopValue; i++){
			arr.push(startValue+step*(i+1))
		}

		return arr
	}

	/** 
	* Function that inputs the interval at which you want to decrease the step and returns all volts of the test, included the step 
	* @param {int} whenToIncrementRate is interval at which you want to decrease the step in voltage supplied to chip with HV supplier
	*/
	function getRun(whenToIncrementRate = 1) { 
		// Generate initial run
		let runs = makeArr(start_voltage, final_voltage, step);
		// Extract info of desired last interval
		let fin_int = runs.slice(-(whenToIncrementRate+1));
		runs = runs.slice(0, -(whenToIncrementRate+1));
		// Calculate last interval
		fin_int = makeArr(fin_int[0], fin_int[fin_int.length-1], final_step)
		// Add last interval
		runs.push(fin_int)
		// Flatten
		runs = runs.flat()
		// Change sign
		/*runs.forEach(function(e, index){
			runs[index] *= -1;
		}, array);*/

		return runs;
	}


	voltage_run = getRun();


	setTimeout(async function run(){
		///////////
		/// The block below will run every time_step seconds + operation time for getRun.length times
		///////////

		/*
		* During each operation we need to:
		* - set_hv_voltage OK
		* - read_hv_current OK
		* - store values OK (returns)
		* - this has to be done in steps of 5 V until last steps where we will go every 1 V OK
		* - process will stop if we exceed the current limit OK
		* - needs to be chip specific: 20 \ohm \cdot cm --> 70V
		* 							   200 \ohm \cdot cm --> 25V
		* - conigurable so that we can choose what type of test we want: test the limit, standard procedure => make it fully flexible in GUI but also make different default settings
		*/

		b = setTimeout(run, time_step*1e3);
		// voltage_run[i] is the voltage that needs to be applied on each run


		await set_lv_voltage(4, voltage_run[i]); // TODO TRY OUT: este await es necesario? Yo lo que quiero es que hasta que no se asegure que lo otro ha cambiado no siga. LIMITACION: una vez que se cambie en la ODB yo voy a tirar. Cómo si si el keitley está lista para medir? Luigi en la func inicial lo hacía así. Por ahora tiramos
		aux = await get_lv_current(4);

		curve.push([voltage_run[i], aux])

		if(aux >= current_limit){
			dlgAlert('Current limit exceeded')
			clearTimeout(b);
			return curve;
		}

		i++;
		if(i == voltage_run.length+1) {
			//\\\\\\
			//\ This will be executed after all runs are completed
			//\\\\\\
			clearTimeout(b);
			//heading.innerText = 'Done';
			console.log(curve)
			return curve;
			//\\\\\\
		}
		///////////
	}, time_step*1e3);
}


/**
 * 3rd version of function to perform the iv_curve test on the ladder. See function comments for more details.
 * Improvements:
 *   - for loop structure
 *   - update time taken into account
 * 
 * @author Pepe, Luigi
 * @param {int} start_voltage Start voltage in Volts
 * @param {int} final_voltage Target voltage in Volts
 * @param {int} step Step in initial (fast) stage in Volts
 * @param {int} final_step Step in final (slow) stage in Volts
 * @param {int} time_step Time step between runs in secs.
 * @param {int} current_limit Current limit in Amps. Need to make sure that 
 * @returns {array} Returns an array of 2-vector elements. First element corresponds to input voltage 
 * 					whereas second element corresponds to measured current. Output array will be of 
 * 					desired length unless current limit is reached
 */
 async function iv_curve4(){//start_voltage, final_voltage, step, final_step, time_step, current_limit
	//Variables taken from GUI
	let start_voltage = 0; // BEWARE OF THE SIGNS. Input will be negative --> work will be done with positive values --> final voltage input in keithley will be negative
	let final_voltage = 20; // Needs to be device specific. Can be changed, but should be set as default option after changing between devices
	let step = 5; // this will result in final_voltage/step+1 iterations, unless stopped by current limit or user
	let final_step = 1; // step for final round
	let time_step = 2; // seconds between operation
	let current_limit = 0.00001; //in amperes. The current limit in the Keithley needs to be set as well!!!!! But this current limit from should be slightly lower to operate as expected

	/* 
	* Helper function to make arrays between start and stop values with a separation of step between its elements. 
	* If the step is not chosen so that it ends on stopValue, it will output until the last possible value. Example:
	* makeArr(0, 10, 7) --: [0, 7]
	*/
	function makeArr(startValue, stopValue, step) {
		let arr = [startValue]
		for (let i = 0; arr[i]+step <= stopValue; i++){
			arr.push(startValue+step*(i+1))
		}

		return arr
	}

	/** 
	* Function that inputs the interval at which you want to decrease the step and returns all volts of the test, included the step 
	* @param {int} whenToIncrementRate is interval at which you want to decrease the step in voltage supplied to chip with HV supplier
	*/
	function getRun(whenToIncrementRate = 4) { 
		// Generate initial run
		let runs = makeArr(start_voltage, final_voltage, step);
		// Extract info of desired last interval
		let fin_int = runs.slice(-(whenToIncrementRate+1));
		runs = runs.slice(0, -(whenToIncrementRate+1));
		// Calculate last interval
		fin_int = makeArr(fin_int[0], fin_int[fin_int.length-1], final_step)
		// Add last interval
		runs.push(fin_int)
		// Flatten
		runs = runs.flat()
		// Change sign
		runs.forEach(function(e, index){
			runs[index] *= -1;
		}, []);

		return runs;
	}


	voltage_run = getRun();
	let curr, curve = [];

	for (let j = 0; j < voltage_run.length; j++){
		// Set voltage and return curve if this fails
		if(!(await set_hv_voltage(voltage_run[j]))){
			return curve;
		}

		//Sleep
		await sleep(time_step*1e3)

		// Get current read from power supple
		curr = await get_hv_current();

		console.log('j='+j+'; volt='+voltage_run[j]+'; curr='+curr);

		// Check current limit
		if (Math.abs(curr) >= current_limit){ //TODO CURRENT
			curve.push([await get_hv_voltage(), curr]);
			dlgAlert('Current limit exceeded')
			await set_hv_voltage(0);
			break;
		} else {
			curve.push([voltage_run[j], curr]);
		}		
	}

	console.log(curve);
	return curve;
}


/**
 * Function to perform the iv_curve test on the ladder. See function comments for more details.
 * @author Pepe, Luigi
 * @param {int} start_voltage Start voltage in Volts
 * @param {int} final_voltage Target voltage in Volts
 * @param {int} step Step in initial (fast) stage in Volts
 * @param {int} final_step Step in final (slow) stage in Volts
 * @param {int} time_step Time step between runs in secs.
 * @param {int} current_limit Current limit in Amps
 * @returns {array} Returns an array of 2-vector elements. First element corresponds to input voltage 
 * 					whereas second element corresponds to measured current. Output array will be of 
 * 					desired length unless current limit is reached
 */
async function iv_curve2(){
	let aux, curve = [], i = 0; // control through this variable --> control through config?

	//Variables taken from GUI
	let start_voltage = 0; // BEWARE OF THE SIGNS. Input will be negative --> work will be done with positive values --> final voltage input in keithley will be negative
	let final_voltage = 20; // Needs to be device specific. Can be changed, but should be set as default option after changing between devices
	let step = 5; // this will result in final_voltage/step+1 iterations, unless stopped by current limit or user
	let final_step = 1; // step for final round
	let time_step = 0.5; // seconds between operation
	let current_limit = 10e-6; //in amperes

	/* 
	* Helper function to make arrays between start and stop values with a separation of step between its elements. 
	* If the step is not chosen so that it ends on stopValue, it will output until the last possible value. Example:
	* makeArr(0, 10, 7) --: [0, 7]
	*/
	function makeArr(startValue, stopValue, step) {
		let arr = [startValue]
		for (let i = 0; arr[i]+step <= stopValue; i++){
			arr.push(startValue+step*(i+1))
		}

		return arr
	}

	/** 
	* Function that inputs the interval at which you want to decrease the step and returns all volts of the test, included the step 
	* @param {int} whenToIncrementRate is interval at which you want to decrease the step in voltage supplied to chip with HV supplier
	*/
	function getRun(whenToIncrementRate = 1) { 
		// Generate initial run
		let runs = makeArr(start_voltage, final_voltage, step);
		// Extract info of desired last interval
		let fin_int = runs.slice(-(whenToIncrementRate+1));
		runs = runs.slice(0, -(whenToIncrementRate+1));
		// Calculate last interval
		fin_int = makeArr(fin_int[0], fin_int[fin_int.length-1], final_step)
		// Add last interval
		runs.push(fin_int)
		// Flatten
		runs = runs.flat()
		// Change sign
		runs.forEach(function(e, index){
			runs[index] *= -1;
		}, array);

		return runs;
	}


	voltage_run = getRun();


	setTimeout(async function run(){
		///////////
		/// The block below will run every time_step seconds + operation time for getRun.length times
		///////////

		/*
		* During each operation we need to:
		* - set_hv_voltage OK
		* - read_hv_current OK
		* - store values OK (returns)
		* - this has to be done in steps of 5 V until last steps where we will go every 1 V OK
		* - process will stop if we exceed the current limit OK
		* - needs to be chip specific: 20 \ohm \cdot cm --> 70V
		* 							   200 \ohm \cdot cm --> 25V
		* - conigurable so that we can choose what type of test we want: test the limit, standard procedure => make it fully flexible in GUI but also make different default settings
		*/

		b = setTimeout(run, time_step*1e3);
		// voltage_run[i] is the voltage that needs to be applied on each run


		await set_hv_voltage(voltage_run[i]); // TODO TRY OUT: este await es necesario? Yo lo que quiero es que hasta que no se asegure que lo otro ha cambiado no siga. LIMITACION: una vez que se cambie en la ODB yo voy a tirar. Cómo si si el keitley está lista para medir? Luigi en la func inicial lo hacía así. Por ahora tiramos
		aux = await get_hv_current();

		curve.push([voltage_run[i], aux])

		if(aux >= current_limit){
			dlgAlert('Current limit exceeded')
			clearTimeout(b);
			return curve;
		}

		i++;
		if(i == voltage_run.length+1) {
			//\\\\\\
			//\ This will be executed after all runs are completed
			//\\\\\\
			clearTimeout(b);
			heading.innerText = 'Done';
			console.log(curve)
			return curve;
			//\\\\\\
		}
		///////////
	}, time_step*1e3);
}


function iv_curve() {
	let funcArray = [];

	//These 4 should be taken from html
	volt_step = 10;
	start_volt = 0;
	volt_max = 20;
	max_current = 1e-5;

	max_iteration = parseInt((volt_max-start_volt)/volt_step);
	iteration = 0;

	test_result["IV"] = {}

	for (volt = 0; volt <= volt_max; volt = volt+volt_step) {
		funcArray.push(function(iteration) {
			console.log("Iteration = " + iteration + ", voltage = " + (start_volt - iteration*volt_step));
			set_hv_voltage(start_volt - iteration*volt_step).then( function(volt) {
				var current = 0
				get_hv_current().then( function (val) {
					current = val[0];
					console.log("Current = " + current)
					test_result["IV"][parseFloat(start_volt - iteration*volt_step)] = current;
					sleep(2000).then( () => {
						iteration +=1;
						if (iteration <= max_iteration && Math.abs(current) < max_current)
							return funcArray[iteration](iteration);
						else {
							console.log(test_result)
							return;
						}
					})		
				})
			})
		});
	}

funcArray[0](0);

}


//// LV POWER ON TEST: 
/**
 * Function that carries out the 'LV Power ON test' (name subject to change) on a ladder of 6 mupix Description of test:
 *  0. Setup: HV => -5V and LV => 2V
 * 	1. Turn off all mupix of ladder
 *  2. Turn on each mupix of ladder independently. If current jumps ~ 400-600 mA, mupix passes the test. If not, chip doesn't pass the test
 * 
 * @author Pepe
 * @returns {intArray} Returns an array of 6 elements with 0/1 flags. 0 flag means test not passed. 1 means test passed.
 */
async function lv_power_on_test(channel, time_step){
	//0. Setup and turning off of chips
	await turnOnOffMupix(999, mupixState.OFF); // Turn off all chips
	await set_hv_voltage(-5); // Set HV voltage to -5
	await set_lv_voltage(channel, 2); // Set LV voltage to 2

	//Sleep time to wait for update before reading current
	// TODO change by specific time --> ask luigi for it
	await sleep(500);

	//1. Turn on each pixel and read change in current
	let init_curr, curr, delta, result = [], ladder_size = 6;

	for (let i = 0; i < ladder_size; i++){
		// Measure init current
		init_curr = await get_lv_current();
		// Turn on
		await turnOnOffMupix(i, mupixState.ON);
		// Wait to measure
		await sleep(500);
		// Measure final current
		curr = await get_lv_current();
		
		//Check current increase
		let delta = Math.abs(init_curr - curr);
		if (400 < delta && delta < 600){
			result[i] = 1;
		} else {
			result[i] = 0;
		}

		//Turn off
		await turnOnOffMupix(i, mupixState.OFF);

		// Wait time
		await sleep(time_step);
	}
}


//// LV TEST 0

function lv_test_0_2(){

}


function lv_test_0() {
	
	//These should be taken from html
	channel = "1"

	console.log("LV test 0")

	get_lv_current(channel).then(function(val) {
		let current_before = val[0]
		configure_chip().then(function() {
			sleep(2000).then(function () {
				get_lv_current(channel).then(function(val2) {
					let current_after = val2[0];
					let current_diff = current_after-current_before;
					console.log("Currents: before config = " + current_before + ", after config = " + current_after + ", difference = " + current_diff)
				})
			})
		})
	})

}


/*function configure_chip () {
	return new Promise (function(resolve, reject) {
			mjsonrpc_db_paste(["/Equipment/Switching/Settings/MupixConfig"], [true]).catch(function(error) {
			mjsonrpc_error_alert(error);
		}).then (function (rpc) {
			resolve(rpc)
		});
	});
}

//var tmp_curr;

var test_result = {};

function get_hv_current() {
	return new Promise (function(resolve, reject) {
		mjsonrpc_db_get_values(["/Equipment/KEITHLEY0/Variables/Current[0]]"]).then(function(rpc) {
			console.log("Result = " + rpc.result.data)
			tmp_curr = rpc.result.data
			resolve(tmp_curr);
		})
		//return tmp_curr;
	})
}

function get_lv_current(channel) {
	return new Promise (function(resolve, reject) {
		mjsonrpc_db_get_values(["/Equipment/HAMEG0/Variables/Current[" + channel + "]"]).then(function(rpc) {
			console.log("Result = " + rpc.result.data)
			tmp_curr = rpc.result.data
			resolve(tmp_curr);
		})
		//return tmp_curr;
	})
}
*/