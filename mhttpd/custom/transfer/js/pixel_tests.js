/**
 * File created by @author Luigi
 * Edited by @author Luigi, Pepe
 * Created on the 5th of march of 2022
 * 
 * 
 * Future development ideas:
 * - Implement a generic function to change any value in the ODB through await/async structure
 * 	 recode everything to use this --> code-readability and coding efficiency would improve
 * 
 * 
 * BEWARE: 
 * - Trying to push the same config two times in a row results a switch_fe crashes
 * 
 */


//**************************************************//
/* GUI REQUIREMENTS
 *
 * - Specify ladder id
 * - Specify downstream or upstream --> code
 *
 *
//**************************************************/


//**************************************************//
/* CODE - TEST REQUIREMENTS
 *
 * - Qualification criteria --> grading system A, B, C, D, E, F at chip level
 * - Output format and storing:
 * 		+ Test number 1:
 * 			- Params
 * 			- Results
 * 		+ Test number 2:
 * 		.
 * 		.	
 * 		.
 * 		
 * 
 * 	- Have a separate
 * - Divide page in two: left, one test at a time, specify params
 * 						 right, button to run all
 *
//**************************************************/



// Variables
const mupixState = {
	ON: 5,
	OFF: 0
}


//**************************************************//
// UTILS
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
// MIDAS UTILS
//**************************************************//

/**
 * Function to change any value in the ODB. Needs to be called as a Promise with await keyword.
 * 
 * 
 * @author Pepe
 * @param pathToValue
 * @param value
 * @returns {Promise} Success?
 */
async function setODBValue(pathToValue, value, errorMsg){ // Remember that you can call a JS script with any number of params that you'd like. If a param is not present, its undefined. Function will manage	
	console.log(pathToValue)
	try{
		if (!Array.isArray(pathToValue)){
			return await mjsonrpc_db_paste([pathToValue], [value]);			
		} else {
			return await mjsonrpc_db_paste(pathToValue, value);
		}
		
	}catch (err){
		let errMsg = 'Couldn\'t change value at '+pathToValue+': '+err;
		if (errorMsg != 'undefined'){
			errMsg = arguments.slice(-1)[0]+': '+err;
		}

		console.log(errMsg);
		mjsonrpc_error_alert(errMsg);
		return 0;
	}
}


/**
 * Function to get current value at ODB of certain variable. 
 * 
 * @param {string} pathToValue Array of paths or individual path
 * @param {string} errorMsg Custom error message if necessary
 * @returns 
 */
async function getODBValue(pathToValue, errorMsg){
	try{
		if (!Array.isArray(pathToValue)){
			let result_rpc = await mjsonrpc_db_get_values([pathToValue]);
			console.log(pathToValue);
			return result_rpc.result.data[0];
		} else {
			let result_rpc = await mjsonrpc_db_get_values(pathToValue);
			console.log(pathToValue);
			return result_rpc.result.data;
		}
		
	} catch (err){
		let errMsg = 'Couldn\'t get value at '+pathToValue+': '+err;
		if (errorMsg != 'undefined'){
			errMsg = arguments.slice(-1)[0]+': '+err;
		}

		mjsonrpc_error_alert(errMsg);
		console.log(errMsg);
		return 'null';
	}
}


/**
 * Function that calls: 
 * 	setODBValue('/Equipment/Switching/Settings/MupixChipToConfigure', mupixID);
 *	setODBValue('/Equipment/Switching/Settings/MupixConfig', true);
 * 
 *  Load current configuration to desired chip
 * 
 * @author Pepe
 * @param {int} mupixID MupixID to set configuration to. Right now: values from 0 to 5
 */
async function pushConfig(mupixID){
	let a = await setODBValue('/Equipment/Switching/Settings/MupixChipToConfigure', mupixID);
	let b = await setODBValue('/Equipment/Switching/Settings/MupixConfig', true);

	return Boolean(a.result.status[0] & b.result.status[0]);
}


//**************************************************//
// POWER CONTROLS
//**************************************************//

//// HIGH VOLTAGE POWER SUPPLIER CONTROLS

/**
 * Function to set the output voltage in the HV supplier (Keithley 2611B)
 * 
 * @author Luigi, Pepe
 */
async function set_hv_voltage(volt) {
	let a = await setODBValue("/Equipment/KEITHLEY0/Variables/Demand Voltage[0]", volt, "Could not set output voltage in HV supplier:")
	return Boolean(a.result.status[0]);
}


/**
 * Function to get the voltage output readout in HV supplier (Keithley 2611B)
 * @author Pepe
 * @returns {float} Returns voltage readout of HV supplier in Amps
 */
 async function get_hv_voltage(){
	return await getODBValue("/Equipment/KEITHLEY0/Variables/Voltage", "Could not get HV voltage from ODB:");
}


/**
 * Function to get current output readout in HV supplier (Keithley 2611B)
 * @author Luigi, Pepe
 * @returns {float} Returns current output readout of HV supplier in Amps
 */
async function get_hv_current(){
	return await getODBValue("/Equipment/KEITHLEY0/Variables/Current", "Could not get HV current from ODB:")
}


/**
 * Function to modify the current limit in HV power supply (Keithley 2611B)
 * 
 * @author Pepe
 * @param {float} curr_lim Current limit in Amps
 * @returns 
 */
async function set_hv_curr_lim(curr_lim){
	let a = await setODBValue('/Equipment/KEITHLEY0/Variables/Current Limit', curr_lim);
	return Boolean(a.result.status[0]);;
}


/**
 * Function to control the output state of the HV power supply (Keithley 2611B)
 * 
 * @author Pepe
 * @param {int} state true -> turn ON. false -> turn OFF
 */
 async function hv_change_channel_output_state(state){
	let a = await setODBValue('/Equipment/KEITHLEY0/Variables/Set State', state, 'Could not change channels state in HV supplier:');
	return Boolean(a.result.status[0]);
}


/**
 * Function to get current state of LV power supplier (Rohde & Schwarz HMP4040)
 * 
 * @author Luigi, Pepe
 * @param {int} channel Desired channel state 1 to 4: 1, 2, 3, 4
 * @returns {float} Returns actual state of channel (true, false)
 */
 async function get_hv_channel_state(channel) {
	return await getODBValue("Equipment/KEITHLEY0/Variables/State", "Could not get HV state from ODB:");
}


//// LOW VOLTAGE POWER SUPPLIER CONTROLS

/**
 * Function to set the output voltage in the LV supplier (Rohde & Schwarz HMP 4040)
 * @param {int} channel Channel output from 1 to 4: 1, 2, 3, 4
 * @param {float} volt Desired demanded output voltage in Volts
 * @author Pepe
 */
async function set_lv_voltage(channel, volt) {
	let a = await setODBValue("/Equipment/HAMEG0/Variables/Demand Voltage["+(channel-1).toString()+"]", volt, "Could not set output voltage in LV supplier:");
	return Boolean(a.result.status[0]);;
}


/**
 * Function to get current output readout of LV power supplier (Rohde & Schwarz HMP4040)
 * 
 * @author Luigi, Pepe
 * @param {int} channel Desired channel output 1 to 4: 1, 2, 3, 4
 * @returns {float} Returns actual current of LV supplier in Amps
 */
async function get_lv_current(channel) {
	return await getODBValue("/Equipment/HAMEG0/Variables/Current[" + (channel-1).toString() + "]", "Could not get LV current from ODB:")
}


/**
 * Function to modify the current limit in LV power supplier (Rohde & Schwarz HMP4040)
 * 
 * @author Pepe
 * @param {float} curr_lim Current limit in Amps
 * @returns 
 */
async function set_lv_curr_limit(channel, curr_lim){
	let a = await setODBValue("/Equipment/HAMEG0/Variables/Current Limit["+(channel-1).toString()+"]", curr_lim, "Could not set current limit in LV supplier:");
	return Boolean(a.result.status[0]);;
}


/**
 * Function to control the output state of the LV power supply (Rhode & Schwarz HMP 4040)
 * @author Pepe
 * @param {int} channel Channel to be activated or deactivated. 
 * @param {int} state Accepts only true or false
 */
async function lv_change_channel_output_state(channel, state){
	let a = await setODBValue("/Equipment/HAMEG0/Variables/Set State["+(channel-1).toString()+"]", state, "Could not change channel "+channel.toString()+" state:");
	return Boolean(a.result.status[0]);;
}


/**
 * Function to get current state of LV power supplier (Rohde & Schwarz HMP4040)
 * 
 * @author Luigi, Pepe
 * @param {int} channel Desired channel state 1 to 4: 1, 2, 3, 4
 * @returns {float} Returns actual state of channel (true, false)
 */
 async function get_lv_channel_state(channel) {
	return await getODBValue("Equipment/HAMEG0/Variables/State["+(channel-1).toString()+"]", "Could not get LV state from ODB:");
}


//// POWER UTILS
function configurePowerControls(lv, hv, hv_curr_lim){
	return;
}


//**************************************************//
// MUPIX CONTROLS
//**************************************************//

/**
 * Function to turn off the chip. It will call the following:
 * - setODBValue("/Equipment/Mupix/Settings/BIASDACS/"+mupixID+"/BiasBlock_on", channelState.OFF);
 * - pushConfig(mupixID);
 * 
 * @author Pepe
 * @param numberID Chip numberID to configure starting at 0. 
 */
async function turnOffMupix(mupixID){
	let a = await setODBValue("/Equipment/Mupix/Settings/BIASDACS/"+mupixID+"/BiasBlock_on", mupixState.OFF);
	let b = await setODBValue('/Equipment/Switching/Settings/MupixChipToConfigure', mupixID);
	let c = await setODBValue('/Equipment/Switching/Settings/MupixConfig', true);

	return Boolean(a.result.status[0] & b.result.status[0] & c.result.status[0]);
}


/**
 * Function to turn on the chip. It will call the following:
 * - setODBValue("/Equipment/Mupix/Settings/BIASDACS/"+mupixID+"/BiasBlock_on", channelState.ON);
 * - pushConfig(mupixID);
 * 
 * @author Pepe
 * @param numberID Chip numberID to configure starting at 0
 */
async function turnOnMupix(mupixID){
	let a = await setODBValue("/Equipment/Mupix/Settings/BIASDACS/" + mupixID + "/BiasBlock_on", mupixState.ON);
	let b = await setODBValue('/Equipment/Switching/Settings/MupixChipToConfigure', mupixID);
	let c = await setODBValue('/Equipment/Switching/Settings/MupixConfig', true);

	return Boolean(a.result.status[0] & b.result.status[0] & c.result.status[0]);
}
 

//**************************************************//
// TEST DEFINITIONS
//**************************************************//


//// IV CURVE TEST

/**
 * Function that performs the IV curve test on the ladder. See function comments for detailing working.
 * 
 * Important configuration previous to execution:
 * Keithley setting:
 * 	- Set current limit ~ 10 uA
 *  - Set MEAS to mA and range to AUTO
 *  - OUTPUT --> ON
 * 
 * 4th version of function. Implementing:
 * 	 - await/async structure
 *   - for loop structure
 *   - tested on ladder
 * 
 * @author Pepe, Luigi
 * @param {int} start_voltage Start voltage in Volts. Indepent of sign, value will be converted to positive, 
 * 							  then back to negative again as input for the Keithley.
 * @param {int} final_voltage Target voltage in Volts. Usually will be given by chip type: 
 * 							  200 \Ohm \cdot m => 20-25V or 20 \Ohm \cdot m => 70-120V
 * @param {int} step Step in initial (fast) stage in Volts
 * @param {int} final_step Step in final (slow) stage in Volts
 * @param {int} time_step Time step between runs in secs.
 * @param {int} current_limit Current limit in Amps. If Keithley limit and curernt_limit are set to be the same, 
 * 							  the output current will never surpass the limit set in Keithley and therefore the 
 *                            loop will not be exited and chip will not be correctly qualified. TODO Make sure
 * 							  loop is exited at some point
 * @returns {array} Returns an array of 2-vector elements. First element corresponds to input voltage 
 * 					whereas second element corresponds to measured current. Output array will be of 
 * 					desired length unless current limit is reached
 */
 async function iv_curve(start_voltage, final_voltage, step, final_step, time_step, current_limit){
	//Variables taken from GUI
	//let start_voltage = 0; 
	//let final_voltage = 20; 
	//let step = 5; 
	//let final_step = 1; 
	//let time_step = 2; 
	//let current_limit = 0.00001; //in amperes. The current limit in the Keithley needs to be set as well!!!!! But this current limit from should be slightly lower to operate as expected

	start_voltage = Math.abs(start_voltage)
	final_voltage = Math.abs(final_voltage)

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
		/*
		if(!(await set_hv_voltage(voltage_run[j]))){
			return curve;
		}
		*/

		// Set voltage
		await set_hv_voltage(voltage_run[i])

		//Sleep
		await sleep(time_step*1e3);

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


//// LV POWER ON TEST: 

/**
 * Function that carries out the 'LV Power ON test' (name subject to change) on a ladder of 6 mupix Description of test:
 *  0. Setup: HV => -5V and LV => 2V
 * 	1. Turn off all mupix of ladder
 *  2. Turn on each mupix of ladder independently. If current jumps ~ 400-600 mA, mupix passes the test. If not, chip doesn't pass the test
 *  3. Returns working chips
 * 
 * TODO: implement upstream, downstream configuration
 * 
 * @author Pepe
 * @param {int} lvChannel LV channel with ladder connected to it
 * @param {int} time_step Time step between tests in seconds
 * @returns {intArray} Returns an array of 6 elements with true/false flags. 0 flag means test not passed. 1 means test passed.
 */
async function lv_power_on_test(lvChannel, time_step){
	//0. Setup and turning off of chips

	let ladder_size = 1; // BEWARE: upstream and downstream configuration

	console.log(await set_hv_curr_lim(10e-6)); // TODO Values should be configured from GUI?
	console.log(await set_hv_voltage(-5)); // Set HV voltage to -5

	console.log(await set_lv_voltage(lvChannel, 2)); // Set LV voltage to 2
	console.log(await set_lv_curr_limit(lvChannel, 2));

	console.log(await hv_change_channel_output_state(true));
	console.log(await lv_change_channel_output_state(lvChannel, true));

	// DEBUG
	console.log('At 1\n\n\n\n\n');

	// Wait until everythig is on
	let current_state_hv = await get_hv_channel_state();
	let current_state_lv = await get_lv_channel_state(lvChannel);
	while(!current_state_hv && ! current_state_lv){
		current_state_hv = await get_hv_channel_state();
		current_state_lv = await get_lv_channel_state(lvChannel);
	}

	await sleep(2000); // Just to make sure everything is ready
	// DEBUG
	console.log('At 2\n\n\n\n\n');

	// Turn off all chips
	for (let i = 0; i < ladder_size; i++){
		console.log(await turnOffMupix(i));
	}

	await sleep(50);

	console.log('At 3\n\n\n\n\n');
	//1. Turn on each pixel and read change in current
	let init_curr, curr, result = [], delta; // TODO implement upstream, downstream

	for (let i = 0; i < ladder_size; i++){
		console.log('\n\n\n'+'At 4:'+i);
		// Measure init current
		init_curr = await get_lv_current(lvChannel);
		// Turn on
		console.log(await turnOnMupix(i));
		// Wait to measure
		console.log("Waiting start")
		await sleep(5000);
		console.log("Waiting end")
		// Measure final current
		curr = await get_lv_current(lvChannel);
		
		//Check current increase
		delta = Math.abs(init_curr - curr);

		console.log("Log: "+init_curr+";"+curr);

		if (400 < delta && delta < 600){ // Increase must be of 500 mA more or less
			result[i] = true;
		} else {
			result[i] = false; 
		}

		await sleep(5000);	
		//Turn off
		await turnOffMupix(i);

		// Wait time
		await sleep(time_step*1e3);
	}

	await hv_change_channel_output_state(false)

	console.log(result);
	return result; // TODO think about upstream/downstream output format
}



// TESTING

async function lv_power_on_test2(){
	let init_curr, curr; // TODO implement upstream, downstream

	init_curr = await get_lv_current(2);
	await turnOnMupix(0);
	console.log("\nTurned on");
	alert('Waiting for turning on');
	
	curr = await get_lv_current(2);

	let delta = Math.abs(init_curr - curr);

	console.log("Current difference:"+delta);

	await turnOffMupix(0);
	alert("Current difference was: "+delta+". Waiting for chip to turn off");

	return "finished"; // TODO think about upstream/downstream output format
}

//////////



////  DAC SCAN TEST

/**
 * Function to perform an individual DAC scan over the specified values
 * 
 * TODO: 
 * 		- choose how to format input dac query. Dropdown list with DACs in GUI
 * 		- time_step should asked?	
 * 		- Are all the values updated when we read
 * 
 * @param {string} dac DAC to be modified
 * @param {array} values DAV values to test
 * @param {intArray} mupixID Individual pixel ID to perform dac scan test at
 */
async function dac_scan_test(dac, values, mupixID){
	// Values should be taken from GUI
	let time_step = 1;
	// Configure

	// Loop over dac values and read current
	let result = [];

	for (let i = 0; i < values.length; i++){
		setODBValue(dac, values[i]);
		sleep(time_step*1e3);
		result.push([values[i], await get_lv_current(2)]) ;
		sleep(100);
	}

	return result;

}


//// QUALIFICATION








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