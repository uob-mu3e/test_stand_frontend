import json
import sys
from json2html import *

json_file = sys.argv[1]
test_folder = sys.argv[2]
check_pass = sys.argv[3]

html_file = "{}/{}.html".format(test_folder, json_file.split(".")[0])
json_file_path = "{}/{}".format(test_folder, json_file)

html_page = "<link rel=\"stylesheet\" href=\"https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css\" integrity=\"sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T\" crossorigin=\"anonymous\">"

if check_pass == '0':
    html_page += "\
    <button onclick=\"{}Function()\", style=\"background-color:#4CAF50;color:white;\">Open {}</button>\
    ".format(json_file.split(".")[0], json_file.split(".")[0])
else:
    html_page += "\
    <button onclick=\"{}Function()\", style=\"background-color:#f44336;color:white;\">Open {}</button>\
    ".format(json_file.split(".")[0], json_file.split(".")[0])

with open(json_file_path) as file:
    json_dict = json.load(file)
    html_page += json2html.convert(
        json=json_dict,
        table_attributes = "id=\"{}\", style=\"display: none\"".format(json_file.split(".")[0])
    )

html_page += "\n<script>\n"
html_page += "function {}Function()\n".format(json_file.split(".")[0])
html_page += "{\n"
html_page += "var x = document.querySelectorAll(\"[id=\'{}\']\");\n".format(json_file.split(".")[0])
html_page += "for(var i = 0; i < x.length; i++) {\n"
html_page += "if (x[i].style.display == \"none\") {\n"
html_page += "x[i].style.display = \"block\";\n"
html_page += "} else {\n"
html_page += "x[i].style.display = \"none\";\n"
html_page += "}\n"
html_page += "}\n"
html_page += "}\n"
html_page += "</script>"

with open(html_file, "w") as file:
    file.write(html_page)
