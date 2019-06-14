function handleFiles(files) {
    // Check for the various File API support.
    if (window.FileReader) {
        // FileReader are supported.
        getAsText(files[0]);
    } else {
        alert('FileReader are not supported in this browser.');
    }
}

function getAsText(fileToRead) {
    var reader = new FileReader();
    // Handle errors load
    reader.onload = loadHandler;
    reader.onerror = errorHandler;
    // Read file into memory as UTF-8
    reader.readAsText(fileToRead);
}

function loadHandler(event) {
    var csv = event.target.result;
    processData(csv);
}

function processData(csv) {
    var allTextLines = csv.split(/\r\n|\n/);
    var lines = [];
    while (allTextLines.length) {
        lines.push(allTextLines.shift().split(','));
    }
    console.log(lines);
    // modbset("/Equipment/Switching/Variables/DATA_WRITE", lines);
    //

    mjsonrpc_db_resize(["/Equipment/Switching/Variables/DATA_WRITE"], [lines.length]);

    var i;
    for (i = 0; i < lines.length; i++) {
        modbset("/Equipment/Switching/Variables/DATA_WRITE[" + String(i) + "]", parseInt(lines[i])); // TODO: why is it possible to address this as an array in js?
    }
    modbset("/Equipment/Switching/Variables/DATA_WRITE_SIZE", parseInt(lines.length));
}

function errorHandler(evt) {
    if(evt.target.error.name == "NotReadableError") {
        alert("Canno't read file !");
    }
}
