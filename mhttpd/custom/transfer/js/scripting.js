//TODO Add error handling
//TODO add comments

var full_config = {};
var dac_library = {};
var temp = 0;




/**
* Function called on loading of page. It loads mhttpd and populates dac_library
* with list of DACs avaliable for change with mjsonrpc_db_ls.
*
* @author Pepe
*/
function load(){
	mhttpd_init('Import from GUI', 100)

	mjsonrpc_db_ls(["/Equipment/Mupix/Settings/BIASDACS/1",
				   "/Equipment/Mupix/Settings/CONFDACS/1",
				   "/Equipment/Mupix/Settings/VDACS/1"]).then(function(rpc) {

		   // Populate dac_library with all DACs at /Equipment/Mupix/Settings/
		   var result = rpc.result;
		   dac_library = {'BIASDACS': result.data[0],
		   				  'CONFDACS': result.data[1],
						  'VDACS':result.data[2]};
	   }).catch(function(error) {
		   console.log("Error getting info on DACs avaliable for change in ODB:", error);
	   });

	document.getElementById("autotest_IV").addEventListener('DOMSubtreeModified', function () {
		if (document.getElementById("autotest_IV").innerHTML == "y")
	            document.getElementById("status_report").innerHTML = "IV test required!";
		else
	            document.getElementById("status_report").innerHTML = "Nothing to do";
    	});
}



/**
* Function that handles the load-file request coming from user. It will dump all
* content of JSON file into an object: full_config
*
* @author Pepe
*/
function handleFile() {
	const [file] = document.querySelector('input[type=file]').files;
	const reader = new FileReader();

	reader.addEventListener("load", async function() {
		// TODO implement async json parsing
		try {
			full_config = await JSON.parse(reader.result); //TODO TRY OUT
		} catch (err) {
			console.log("Could not read input config file:", err)
		}

	}, false);

	if (file) {
		reader.readAsText(file);
	}

	console.log("File loaded")
}


/**
 * Function that will upload the input config in JSON format into the ODB. It will preprocess 
 * and filter out empty dacs from the inputted list. It will iterate over each remaining dac 
 * defined in the input config to check if it is in the current ODB library. If it is, the DAC 
 * will be selected to later on update the ODB value with the new one. 
 * 
 * @author Pepe
 */
async function uploadConfigToODB(){ //TODO TRY OUT
	// Filter out empty named dacs for full_config:
	if (full_config != {}){
		const content = document.querySelector(".content");

		if (temp == 0){
			temp = {};
			for (var i in full_config.zdacs){
				let name = full_config.zdacs[i].name;
				let value = full_config.zdacs[i].value;

				if (name != ''){
					temp[name] = value;
				}
			}
			// Copy result to full_config
			full_config = temp;
		} else {
			console.log("Not first time")
		}

		// Iterate over each full_config dac, check if its in dac_library and
		// set order to ODB
		var paths = []
		var values = []

		for (var dac in full_config){
			if (Object.keys(dac_library.BIASDACS).includes(dac)){
				paths[paths.length] = "/Equipment/Mupix/Settings/BIASDACS/1/"+dac
				values[values.length] = full_config[dac]
				console.log("Entered BIAS")
			} else if (Object.keys(dac_library.CONFDACS).includes(dac)){
				paths[paths.length] = "/Equipment/Mupix/Settings/CONFDACS/1/"+dac
				values[values.length] = full_config[dac]
				console.log("Entered CONF")
			} else if (Object.keys(dac_library.VDACS).includes(dac)){
				paths[paths.length] = "/Equipment/Mupix/Settings/VDACS/1/"+dac
				values[values.length] = full_config[dac]
				console.log("Entered VDACS")
			} else {
				console.log('DAC', dac ,' in input JSON file not present in ODB');
			}
		}

		try{
			await mjsonrpc_db_paste(paths, values);
		} catch (err) {
			dlgAlert("Error while uploading config to ODB:", err)
		}
		


		var now = new Date();
		content.innerText = "UPLOADED! " + now.getHours() + ":" + now.getMinutes() + ":" + now.getSeconds();
	} else {
		dlgAlert("Upload file, please")
	}
}


