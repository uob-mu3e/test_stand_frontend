

function setup_listeners() {
	document.getElementById("keithley_curr").addEventListener("DOMSubtreeModified", function () {
		value = parseFloat(document.getElementById("keithley_curr").innerHTML);
		document.getElementById("keithley_curr_display").innerHTML = value*1000000;
	});
}



function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
 
function set_hv_voltage(volt) {
	return new Promise (function(resolve, reject) {
			mjsonrpc_db_paste(["/Equipment/KEITHLEY0/Variables/Demand Voltage[0]"], [volt]).catch(function(error) {
			mjsonrpc_error_alert(error);
		}).then(function(rpc) {
			resolve(volt);
		});
	})
}

function configure_chip () {
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

function iv_curve() {
	var funcArray = [];

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


setup_listeners();