/*
javascript content:

Difference between const, let and var in JS:
	- var: can be redeclared and updated
	- let: can be updated but not redeclared
	- const: cannot be updated nor redeclared
*/


//TODO Add error handling
//TODO add comments

var full_config = {};
var dac_library = {};
var _temp = 0;
var _mupixChipToConfigureValue;

/**
* Function called on loading of page. It loads mhttpd and populates dac_library
* with list of DACs avaliable for change with mjsonrpc_db_ls.
*/
function load(){
	mhttpd_init('Import from GUI')

	mjsonrpc_db_ls(["/Equipment/Mupix/Settings/BIASDACS/1",
				   "/Equipment/Mupix/Settings/CONFDACS/1",
				   "/Equipment/Mupix/Settings/VDACS/1"]).then(function(rpc) {

		   // Populate dac_library with all DACs at /Equipment/Mupix/Settings/
		   var result = rpc.result;
		   dac_library = {'BIASDACS': result.data[0],
		   				  'CONFDACS': result.data[1],
						  'VDACS':result.data[2]};

		   //console.log(dac_library);

	   }).catch(function(error) {
		   console.log("Error getting info on DACs avaliable for change in ODB:", error);
	   });

   mjsonrpc_db_get_values(["/Equipment/Switching/Settings/MupixChipToConfigure"]).then(function(rpc) {
  		   _mupixChipToConfigureValue = rpc.result.data[0];
  	   }).catch(function(error) {
  		   console.log("Error getting info on DACs avaliable for change in ODB:", error);
  	   });


	//  Setup event listeners	
}

console.log("Before listeenr")
document.addEventListener('DOMContentLoaded', function(event) {
	console.log("Executed");
	document.getElementById("keithley_curr").addEventListener("DOMSubtreeModified", function () {
		console.log("Executed 21:"+event)
		value = parseFloat(document.getElementById("keithley_curr").innerHTML);
		document.getElementById("keithley_curr_display").innerHTML = value*1000000;
		});
})
console.log("After listener")

/**
* Function that handles the load-file request coming from user. It will dump all
* content of JSON file into an object: full_config (TODO What's the name of this
* GUI? )
*/
function handleFile() {
	const [file] = document.querySelector('input[type=file]').files;
	const reader = new FileReader();

	reader.addEventListener("load", () => {
		// TODO implement async json parsing
		 full_config = JSON.parse(reader.result);
	}, false);

	if (file) {
		reader.readAsText(file);
	}

	console.log("File loaded")
}


function uploadConfigToODB(){
	// Algo: (preprocess) filter out empty dacs from full_config. Iterate
	// over each dac defined in full_config to check if it is in dac_library.
	// If it is, set order to ODB. TODO check performance
	// Filter out empty named dacs for full_config:
	if (full_config != {}){
		const content = document.querySelector(".content");

		if (_temp == 0){
			_temp = {};
			for (var i in full_config.zdacs){
				let _name = full_config.zdacs[i].name;
				let _value = full_config.zdacs[i].value;

				if (_name != ''){
					_temp[_name] = _value;
				}
			}
			// Copy result to full_config
			full_config = _temp;
		} else {
			console.log("Not first time")
		}

		// Iterate over each full_config dac, check if its in dac_library and
		// set order to ODB
		var _paths = []
		var _values = []

		for (var dac in full_config){
			if (Object.keys(dac_library.BIASDACS).includes(dac)){
				_paths[_paths.length] = "/Equipment/Mupix/Settings/BIASDACS/1/"+dac
				_values[_values.length] = full_config[dac]
				console.log("Entered BIAS")
			} else if (Object.keys(dac_library.CONFDACS).includes(dac)){
				_paths[_paths.length] = "/Equipment/Mupix/Settings/CONFDACS/1/"+dac
				_values[_values.length] = full_config[dac]
				console.log("Entered CONF")
			} else if (Object.keys(dac_library.VDACS).includes(dac)){
				_paths[_paths.length] = "/Equipment/Mupix/Settings/VDACS/1/"+dac
				_values[_values.length] = full_config[dac]
				console.log("Entered VDACS")
			} else {
				console.log('DAC', dac ,' in input JSON file not present in ODB');
			}
		}

		modbset(_paths, _values);


		var now = new Date();
		content.innerText = "UPLOADED! " + now.getHours() + ":" + now.getMinutes() + ":" + now.getSeconds();
	} else {
		dlgAlert("Upload file, please")
	}
}


function checkAndConfig(){
	if (_mupixChipToConfigureValue != 1){
	   modbset("/Equipment/Switching/Settings/MupixChipToConfigure", 1)
	   _mupixChipToConfigureValue = 1
	   console.log("MupxChipToConfigureValue set to 1") //TODO add error handling
	}

	modbset("/Equipment/Switching/Settings/MupixConfig", "y")
}


function uA_listeners() {
	document.getElementById("keithley_curr").addEventListener("DOMSubtreeModified", function () {
		value = parseFloat(document.getElementById("keithley_curr").innerHTML);
		document.getElementById("keithley_curr_display").innerHTML = value*1000000;
	});
}
