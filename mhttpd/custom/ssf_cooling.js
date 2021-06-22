var convert_v_to_c = function (volt) {
    return (79.5 * Math.exp(-0.00011 * (10000 * (volt / (5 - volt)))) - 1.76).toFixed(2);;
}

var convert_v_to_bar = function (volt) {
    return (volt * 9 / 4).toFixed(2);;
}

document.getElementById("1_temp_v").addEventListener('DOMSubtreeModified', function () {
    document.getElementById("1_temp_c").textContent = convert_v_to_c(document.getElementById("1_temp_v").textContent)
})
document.getElementById("1_pres_v").addEventListener('DOMSubtreeModified', function () {
    document.getElementById("1_press_bar").textContent = convert_v_to_bar(document.getElementById("1_pres_v").textContent)
})

document.getElementById("2_temp_v").addEventListener('DOMSubtreeModified', function () {
    document.getElementById("2_temp_c").textContent = convert_v_to_c(document.getElementById("2_temp_v").textContent)
})
document.getElementById("2_pres_v").addEventListener('DOMSubtreeModified', function () {
    document.getElementById("2_press_bar").textContent = convert_v_to_bar(document.getElementById("2_pres_v").textContent)
})

document.getElementById("3_temp_v").addEventListener('DOMSubtreeModified', function () {
    document.getElementById("3_temp_c").textContent = convert_v_to_c(document.getElementById("3_temp_v").textContent)
})
document.getElementById("3_pres_v").addEventListener('DOMSubtreeModified', function () {
    document.getElementById("3_press_bar").textContent = convert_v_to_bar(document.getElementById("3_pres_v").textContent)
})

document.getElementById("4_temp_v").addEventListener('DOMSubtreeModified', function () {
    document.getElementById("4_temp_c").textContent = convert_v_to_c(document.getElementById("4_temp_v").textContent)
})
document.getElementById("4_pres_v").addEventListener('DOMSubtreeModified', function () {
    document.getElementById("4_press_bar").textContent = convert_v_to_bar(document.getElementById("4_pres_v").textContent)
})