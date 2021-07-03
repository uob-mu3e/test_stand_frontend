// Function to disable updating.
gIsPaused = false;
function pause_resume(){

 if(gIsPaused){
   gIsPaused = false;
   document.getElementById("pause_button").value = "Pause"; 
   $(".displayControlEl").prop('disabled', false);

 }else{
   gIsPaused = true;
   document.getElementById("pause_button").value = "Resume"; 
   $(".displayControlEl").prop('disabled', true);
 }

}


gRunOngoing = true;
var gStatusInflight = false;
function midasStatus(){

  // don't make another request if the last request is still outstanding.
   if(gStatusInflight){
      return;
   }

  gStatusInflight = true;

  mjsonrpc_db_get_values(["/Runinfo"]).then(function(rpc) {    
    var runinfo= rpc.result.data[0];
    if(runinfo.state == 3){
      gRunOngoing = true;
    }else{
      gRunOngoing = false;
    }
    gStatusInflight = false; // finished async routine
  }).catch(function(error) {
    if (error.request) {
      var s = mjsonrpc_decode_error(error);
      console.log("mjsonrpc_error_alert: " + s);
    } else {
       console.log("mjsonroc_error_alert: " + error);
    }
  });
}


// This function sets the directory at which to find the ROOT HTTP server, in form <myhost>/webdir_end/
function setRootanaDirectory(webdir_end){
  // TODO: get port from ROOTANA dynamic
  // TODO:webdir_end should be "manalyzer" not rootana
  webdir_end = "";
   var webdir = location.protocol + '//' + location.hostname + ":8088" + "/" + webdir_end + "/";
   console.log("Setting to " +webdir);
   rootana_dir = webdir;
   document.getElementById('rootanaLink').href = webdir;
}


// clean div for new display
function cleanDivs(){
   console.log("clean divs!!!");
   deleteOldPlots();   
   cleanJSDivs();   
   $("#graphdiv").html("");
   $("#graphdiv21").html("");
   $("#graphdiv22").html("");
}

var histogramsSetupInfo;
var currentHistogramInfo;
invalidSetup = true;


// configure the display for current settings; return true if successful
function configureDisplay(histogramsSetupInfo){
  var histogramID = $(":button.displayControlEl.select_buttons.selected").attr('id')

  cleanDivs();
  if (!(histogramID in histogramsSetupInfo)){
    console.log("Failure; value " + histogramID + " is not in histogram Setup JSON dump ");
    invalidSetup = true;
    return false;
  } 

  // cache the current configuration; use this to get Webname
  currentHistogramInfo = histogramsSetupInfo[histogramID];


  // set channel button; if MaxChannel <= 1 or MaxChannel doesn't exist, then handle individual histograms
  if(("MaxChannels" in currentHistogramInfo)
     &&  currentHistogramInfo["MaxChannels"] > 1){
    document.getElementById("DisplaySelectionDiv").style.display = "block"; 
    document.getElementById("ChannelSelectionDiv").style.display = "block"; 
    document.getElementById("channel_number").max = currentHistogramInfo["MaxChannels"] -1;
    document.getElementById("listedHistoDisplayDiv").style.display = "none";
    document.getElementById("channel_number").value = 0;  
    // set module button
    if(!("MaxModules" in currentHistogramInfo) || ( currentHistogramInfo["MaxModules"] <= 1 ) ){
      document.getElementById("ModuleSelectionDiv").style.display = "none"; 
    } else {
      document.getElementById("ModuleSelectionDiv").style.display = "block"; 
      document.getElementById("module_number").max = currentHistogramInfo["MaxModules"] -1;
      document.getElementById("module_number").value = 0;  
    }
  } else if ("MaxLayers" in currentHistogramInfo) {
    document.getElementById("DisplaySelectionDiv").style.display = "none"; 
    document.getElementById("LayerSelectionDiv").style.display = "block";
    document.getElementById("layer_number").max = currentHistogramInfo["MaxLayers"] -1;
    if( typeof currentHistogramInfo["WebserverName"] == 'string'){
      document.getElementById("listedHistoDisplayDiv").style.display = "none";
    }else{
      document.getElementById("listedHistoDisplayDiv").style.display = "block";
      $("#listedHistoMenu").empty();
      for (var key in currentHistogramInfo["WebserverName"]){
        var option = document.createElement("option");
        console.log("Test key : " + key);
        option.text = key;
        document.getElementById("listedHistoMenu").add(option); 
      }
    }
    document.getElementById("layer_number").value = 0;
    // set Pixel button
    if(!("MaxPixels" in currentHistogramInfo) || ( currentHistogramInfo["MaxModules"] <= 1 ) ){
      document.getElementById("PixelSelectionDiv").style.display = "none"; 
    } else {
      document.getElementById("PixelSelectionDiv").style.display = "block"; 
      document.getElementById("pixel_number").max = currentHistogramInfo["MaxPixels"] -1;
      document.getElementById("pixel_number").value = 0;  
    }
  } else{
    document.getElementById("ChannelSelectionDiv").style.display = "none";
    document.getElementById("LayerSelectionDiv").style.display = "none";
    document.getElementById("DisplaySelectionDiv").style.display = "none";
    // Treat it differently if it is a single histogram or a list of histograms
    if( typeof currentHistogramInfo["WebserverName"] == 'string'){
      document.getElementById("listedHistoDisplayDiv").style.display = "none";
    }else{
      document.getElementById("listedHistoDisplayDiv").style.display = "block";
      $("#listedHistoMenu").empty();
      for (var key in currentHistogramInfo["WebserverName"]){
        var option = document.createElement("option");
        console.log("Test key : " + key);
        option.text = key;
        document.getElementById("listedHistoMenu").add(option); 
      }
    }
  }
  invalidSetup = false;
  return true;
}


// main method to setup which histograms to display, using JSON formatted data
function selectHistogramsFromJSON(){
  // get the data used to do setup
  var request = new XMLHttpRequest();
  request.open('GET', "online_ana_setup.json", false);
  request.send(null);

  if(request.status != 200){ 
    document.getElementById("readstatus").innerHTML = "Couldn't get data describing set of histograms";
    document.getElementById("readstatus").style.color = 'red'; 
    console.log("Couldn't get data describing set of histograms ");
    return;                                                  
  }

  // read json file
  console.log("return: " + request.responseText);
  histogramsSetupInfo = JSON.parse(request.responseText);
   
  // create button for each element in json
  var buttonString = "";
  var first = true;
  var hasSettings = false;
  for (var key in histogramsSetupInfo){
    if(key == "DisplaySettings"){
       hasSettings = true;
       continue;
    }
    //console.log("key " + key);
    buttonString += "<button class='displayControlEl select_buttons";
    if(first){
       buttonString += " selected";
       first = false;
    }
    buttonString += "' id='"+key+"' > "+key+" </button>\n";
  }

  // Disable the ROOT vs PLOT-LY option, if specified in options file
  var styleSet = false;
  if(hasSettings && "DisplayStyle" in histogramsSetupInfo["DisplaySettings"]){
    console.log("Have display style");
    if( histogramsSetupInfo["DisplaySettings"]["DisplayStyle"] == "root" || histogramsSetupInfo["DisplaySettings"]["DisplayStyle"] == "plotly"){
       $(":button.styleControlEl.select_buttons.selected").removeClass("selected");
       $("#"+histogramsSetupInfo["DisplaySettings"]["DisplayStyle"]+"_style").addClass("selected");
       document.getElementById("root_style").style.display = "none";   
       document.getElementById("plotly_style").style.display = "none";
       // For reasons unclear I need to manually set the div size if I use only plotly
       // This won't work cleanly on all displays.
       if(histogramsSetupInfo["DisplaySettings"]["DisplayStyle"] == "plotly"){            
          document.getElementById("graphdiv").style.width="600px";
          document.getElementById("graphdiv").style.height="500px";
       }
       styleSet = true;
       console.log("Making display style buttons invisible");
    }
  }

  if(!styleSet){
    console.log("Making display style buttons visible");
    document.getElementById("root_style").style.display = "inline";   
    document.getElementById("plotly_style").style.display = "inline";
  }  

  // Configure the displays
  document.getElementById("displayControlButtons").innerHTML = buttonString;
  configureDisplay(histogramsSetupInfo);

  // set up what happens for display button clicks
  $(":button.displayControlEl,select_buttons").click(function(){
    $(":button.displayControlEl.select_buttons.selected").removeClass("selected");
    $(this).addClass("selected");
    
    // find the configuration for this button
    configureDisplay(histogramsSetupInfo);
    
    ShowParams();
    updatePlots(true);
  });

  // set up what happens for plotting style button clicks
  $(":button.styleControlEl,select_buttons").click(function(){
    $(":button.styleControlEl.select_buttons.selected").removeClass("selected");
    $(this).addClass("selected");
    
    // find the configuration for this button
    configureDisplay(histogramsSetupInfo);
    
    updatePlots(true);
  });
}


function getNumericString(channel){
   var tmpstr = "";  
   if(currentHistogramInfo["MaxModules"] > 1){
      tmpstr += "_" + document.getElementById("module_number").value;
   }
   if(currentHistogramInfo["MaxChannels"] > 1){
      tmpstr += "_" + String(channel);
   }
   return tmpstr;
}


function getLayerString(channel){
   var tmpstr = "";  
   if(currentHistogramInfo["MaxLayers"] > 1){
      tmpstr += "_Layer_" + String(channel);
   }
   return tmpstr;
}


function getPixelString(channel){
   var tmpstr = "";  
   if(currentHistogramInfo["MaxPixels"] > 1){
      tmpstr += "_Pixel_" + String(channel);
   }
   return tmpstr;
}


// Function will call either JSROOT plotting or Plot-ly/dygraph plotting, depending on what user chooses
function plotAllHistograms(plotType, divNames, histogramNameList, deleteDygraph){
  console.log("plotAllHistograms");
  console.log(histogramNameList);
   if($(":button.styleControlEl.select_buttons.selected").attr('id') == "root_style"){
      plotAllHistogramsJSROOT(plotType, divNames, histogramNameList, deleteDygraph);
   }else{
      plotAllHistogramsNonROOT(plotType, divNames, histogramNameList, deleteDygraph);
   }

   
}


function updatePlots(deleteDygraph){

   
  // don't bother updating if run is not ongoing.
  if(gRunOngoing == false){ 
    document.getElementById("readstatus").innerHTML = "No updates when run stopped.";
    document.getElementById("readstatus").style.color = "black";
    document.getElementById("datetime").innerHTML = Date();
    return;
  }

  // don't update if paused.
  if(gIsPaused){
    return;
  }
   
  // Check for single histogram; if so, just plot
  if (("MaxChannels" in currentHistogramInfo) && currentHistogramInfo["MaxChannels"] == 1 ) {
  var channel = Number(document.getElementById("channel_number").value);
  var scan_type = document.getElementById("HistoDisplay").value;

  switch (scan_type) {
    case "Single":
       plotAllHistograms("single",["graphdiv"],[currentHistogramInfo["WebserverName"]+getNumericString(channel)],deleteDygraph);
       break;

    case "Overlay":
       var numberOverlay = Number(document.getElementById("overlay_number").value);

       var histograms = [];
       for(var ii = 0; ii < numberOverlay; ii++){
          var index = channel + ii;
          if(index < currentHistogramInfo["MaxChannels"]){
             histograms.push(currentHistogramInfo["WebserverName"]+getNumericString(index));
          }
       }          
       plotAllHistograms("overlay",["graphdiv"],histograms,deleteDygraph); 
       break;
    case "Multiple":
       var histograms = []; var divnames = [];
       histograms.push(currentHistogramInfo["WebserverName"]+getNumericString(channel)); divnames.push("graphdiv21")
       histograms.push(currentHistogramInfo["WebserverName"]+getNumericString(channel+1)); divnames.push("graphdiv22")
       plotAllHistograms("multiple",divnames,histograms,deleteDygraph);
       break;
  }
  } else if ("MaxLayers" in currentHistogramInfo) {
    var layer = Number(document.getElementById("layer_number").value);
    var pixel = Number(document.getElementById("pixel_number").value);
    var scan_type = document.getElementById("HistoDisplay").value;
    if(typeof currentHistogramInfo["WebserverName"] == 'string'){
      plotAllHistograms("single",["graphdiv"],[currentHistogramInfo["WebserverName"]+getLayerString(layer)+getPixelString(pixel)],deleteDygraph);
    }else{
       // find the histogram name using drop-down menu       
       var menukey = document.getElementById("listedHistoMenu").value;
       plotAllHistograms("single",["graphdiv"],[currentHistogramInfo["WebserverName"][menukey]+getLayerString(layer)+getPixelString(pixel)],deleteDygraph);
    }
    console.log(currentHistogramInfo["WebserverName"]+getLayerString(layer)+getPixelString(pixel));
  } else {
    if(typeof currentHistogramInfo["WebserverName"] == 'string'){   
       plotAllHistograms("single",["graphdiv"],[currentHistogramInfo["WebserverName"]],deleteDygraph);
    }else{
       // find the histogram name using drop-down menu       
       var menukey = document.getElementById("listedHistoMenu").value;
       plotAllHistograms("single",["graphdiv"],[currentHistogramInfo["WebserverName"][menukey]],deleteDygraph);
    }
  }
  document.getElementById("datetime").innerHTML = Date();
  return;
}


function ShowParams() {
   var scan_type = document.getElementById("HistoDisplay").value;
   // document.getElementById("params").innerHTML = "You chose: " + scan_type;

   switch (scan_type) {
      case "Single":
         document.getElementById("OverlayParams").style.display = "none";
         document.getElementById("MultipleParams").style.display ="none";
         document.getElementById("graphdiv").style.display ="block";
         document.getElementById("graphdiv21").style.display ="none";
         document.getElementById("graphdiv22").style.display ="none";
         break;
      case "Overlay":
         document.getElementById("OverlayParams").style.display = "block";
         document.getElementById("MultipleParams").style.display ="none";
         document.getElementById("graphdiv").style.display ="block";
         document.getElementById("graphdiv21").style.display ="none";
         document.getElementById("graphdiv22").style.display ="none";
         break;
      case "Multiple":
         document.getElementById("OverlayParams").style.display = "none";
         document.getElementById("MultipleParams").style.display ="block";
         document.getElementById("graphdiv").style.display ="none";
         document.getElementById("graphdiv21").style.display ="block";
         document.getElementById("graphdiv22").style.display ="block";
         break;
      default:
         break;
   }
   updatePlots(true);
}
var gIntervalID = 0; 
gIntervalID = setInterval(updatePlots,5000);
setInterval(midasStatus,5000);   
function changeUpdateTime() {
   // Delete old update interval
   clearInterval(gIntervalID);

   // set new interval
   var currentUpdateTime = Number(document.getElementById("updatePeriod").value);
   gIntervalID = setInterval(updatePlots,currentUpdateTime);
}
