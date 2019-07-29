

/* Various and sundry helper functions */
/* Niklaus Berger 3.5.2017  --  niberger@uni-mainz.de */
/* Simon Corrodi November 2017 */

function mu3e_init(current_page, interval, callback) {
    mhttpd_init(current_page, interval, callback);
    /*n = document.getElementById('mheader_expt_name');
    n.innerHTML = "<img src=\"img/logo_drawing_50.png\" alt=\"Mu3e Logo\">";*/
    init_dragable();
}


/* Handle Dragging of Elements */
function init_dragable() {
    var cboxes = document.querySelectorAll('.contentbox');
    [].forEach.call(cboxes, function(cbox) {
        cbox.addEventListener('dragstart', handleDragStart, false);
        cbox.addEventListener('dragenter', handleDragEnter, false);
        cbox.addEventListener('dragover',  handleDragOver,  false);
        cbox.addEventListener('dragleave', handleDragLeave, false);
        cbox.addEventListener('drop',      handleDrop,      false);
        cbox.addEventListener('dragend',   handleDragEnd,   false);
    });
}

var dragSrc = null;

function handleDragStart(e) {
    this.style.opacity = '0.4';

    dragSrc = this;
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/html', this.innerHTML);
}

function handleDragOver(e) {
    if (e.preventDefault) {
        e.preventDefault(); // Necessary. Allows us to drop.
    }
    e.dataTransfer.dropEffect = 'move';  // See the section on the DataTransfer object.
    return false;
}

function handleDragEnter(e) {
    this.classList.add('over'); // add the over class 
}

function handleDragLeave(e) {
    this.classList.remove('over');  
}

function handleDrop(e) {
    //this.style.opacity = '1.0';
    if (e.stopPropagation) {
        e.stopPropagation(); // stops the browser from redirecting.
    }

    if(dragSrc != this && dragSrc != null) {
        var classes = dragSrc.className;
        dragSrc.innerHTML = this.innerHTML;
        dragSrc.className = this.className;
        this.innerHTML = e.dataTransfer.getData("text/html");
        this.className = classes;
    }

    return false;
}

function handleDragEnd(e) {
    this.style.opacity = '1.0';
    var cboxes = document.querySelectorAll('.contentbox');
    [].forEach.call(cboxes, function (cbox) {
        cbox.classList.remove('over');
    });
}




String.prototype.endsWith = function(suffix) {
    return this.indexOf(suffix, this.length - suffix.length) !== -1;
};


function populate_nav() {
    var paths = [ "/Experiment", "/Custom"]; 
    var data_odb=ODBMCopy(paths, populate_nav_callback, "json");
}

function populate_nav_callback(data){
    var obj= JSON.parse(data);
    // This is the part for the generic MIDAS buttons
    var experiment=obj[0];
    var buttonsstring = experiment["Menu Buttons"];
    var buttons = buttonsstring.split(',');
    var list ='';
    for(var i =0; i < buttons.length; i++){
        buttons[i] = buttons[i].trim();
        list += '<li id=\"' + buttons[i] +'\"> <a href=\'./?cmd=' + buttons[i] +'\'>' +buttons[i] + '</a></li>\n';
    }
    
    
    
    // And now we go for the Custom buttons
    var custom = obj[1];
    //document.getElementById('message1').innerHTML = data;
    for(var prop in custom){
        if(prop.endsWith('&')){
                var name = prop.substr(0, prop.length - 1);
                list += '<li id=\"' + name +'\"> <a href=\'./' + name +'\'>' + name + '</a></li>\n';
        }
           
    }
    document.getElementById('navigationlist').innerHTML = list;
    
    set_current();
}

function toggle(x) {
	//var x = document.getElementById(name);
	//alert(x);
	if (x.style.display === "none") {
		x.style.display = "block";
		//n.innerHTML = n.innerHTML.replace('+', '-');
	} else {
		x.style.display = "none";
		//n.innerHTML = n.innerHTML.replace('-', '+');
	}
}

