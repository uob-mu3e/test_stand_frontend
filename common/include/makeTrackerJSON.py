# NOTE: This has been made for cosmic run (needs to be changed for scaling up to outer layers!)
from ctypes import alignment
import json
import argparse
import numpy as np
from collections import OrderedDict

parser = argparse.ArgumentParser(description='Read alignment tree from file and produce json for event display',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--filename', type=str, required=False, help='Input sensor alignment json file', default="./sensors_cosRun2022.json")
parser.add_argument('--outputDir', type=str, default="./", help="path/to/outputfile.json (not including filename")
args = parser.parse_args()

tracker = OrderedDict() # to preserve the order

# add definitions
description = [
        "Tracker configuration chip-by-chip",
        "'Chips' is a dictionary with a unique chip identifier as key. This is NOT related to any ID used in the experiment (configuration, analysis,...).",
        "Every value of the dictionary then corresponds to a chip, and it is a dictionary itself. The element of the dictionary are the following:",
        "- confChip: the chip ID used for configuring the chip with MIDAS and switch_fe. Unique for all chips. from 0 to 119 in Cosmic Run",
        "- runChip: the chip ID from the data stream. (Same as confChip at the moment)",
        "- ladderChip: the chip ID on the physical ladder (from 0 to 5)",
        "- simChip: the chip ID as in the simulation and specbook",
        "- layer: layer to which the chip belongs (0 or 1 for cosmic run)",
        "- direction: upstream (US) or downstream (DS)",
        "- localLadder: ladder to which the chip belongs, as seen inside the layer (from 0 to 7 for Layer 0, from 0 to 9 for layer 1)",
        "- confLadder: ladder to which the chip belongs, using the ladder unique ID for configuration (global ID, from 0 to 39)",
        "- simLadder: ladder to which the chip belongs, using the ladder simulation and specbook ID",
        "- phi: the phi position of the ladder to which the chip belongs",
        "- dataLinkA_onFEB: the FEB data link number to which the chip's link A is connected",
        "- dataLinkB_onFEB: the FEB data link number to which the chip's link B is connected",
        "- dataLinkC_onFEB: the FEB data link number to which the chip's link C is connected",
        "- dataLinkD_onFEB: the FEB data link number to which the chip's link D is connected (-1 if not connected)"
    ]

tracker["description"] = description
tracker["chips"] = {}

n_layers = 2
ladders_per_layer = { "layer0" : 8, "layer1" : 10}

def getSensorDataFromJSON(filename):
    data = []
    with open(filename) as file: data = json.load(file)
    return data

def calculateConfLadder(layer, ladder, localChip):
    side = "US" if localChip <= 2 else "DS"
    confLadder = 0
    DS_GAP = 2
    # tidy up (make recursive?)
    if (layer==1):
        if (side == "US"): confLadder = ladder + ladders_per_layer["layer" + str(layer-1)]
        else: confLadder = ladder + 2*(ladders_per_layer["layer" + str(layer-1)]) + ladders_per_layer["layer" + str(layer)] + DS_GAP
    else:
        if (side == "US"): confLadder = ladder
        else: confLadder = ladder + ladders_per_layer["layer" + str(layer+1)] + ladders_per_layer["layer" + str(layer)] + DS_GAP
    return confLadder

def calculateConfChip(layer, ladder, localChip):
    pattern = [0, 1, 2, 2, 1, 0]
    return calculateConfLadder(layer, ladder, localChip) * 3 + pattern[localChip]

def makeRunChipsForLayer(layer):
    layerChips = []
    for ladder in range(ladders_per_layer["layer" + str(layer)]):
        for chip_in_ladder in range(6):
            layerChips.append(calculateConfChip(layer, ladder, chip_in_ladder))
    return layerChips

layer0 = makeRunChipsForLayer(0)
layer1 = makeRunChipsForLayer(1)

alignmentData = getSensorDataFromJSON(args.filename)

unique_ctr = 0
for layer in range(n_layers):
    print("layer {}".format(layer))

    for ladder in range(ladders_per_layer["layer" + str(layer)]):

        # From Mu3eInnerSiliconLayer.cpp in mu3eSim
        simLadder = 0
        ladderOffset = 5
        layerOffset  = ladderOffset + 5

        if(layer == 0):
            simLadder = int(ladder/4)*8 + ladder%4
        else:
            if(ladder < 3):
                simLadder = ladder
            elif(ladder < 5):
                simLadder = ladder+1
            elif(ladder < 8):
                simLadder = ladder+3
            else:
                simLadder = ladder+4

        recoLadder = simLadder
        # From SiDet.cpp in mu3eTrirec
        if (layer == 0): recoLadder = int(recoLadder / 8) * 4 + recoLadder % 8 
        if (layer == 1): recoLadder = int(recoLadder / 8) * 6 + recoLadder % 8

        print("  ladder {}, simLadder {}, recoLadder {}".format(ladder, simLadder, recoLadder))
        for chip_in_ladder in range(6):

            if (layer == 0): runChip = layer0[chip_in_ladder+6*recoLadder]
            if (layer == 1): 
                if (recoLadder>=4 and recoLadder < 10): runChip = layer1[chip_in_ladder+6*(recoLadder-1)]
                elif (recoLadder>=10): runChip = layer1[chip_in_ladder+6*(recoLadder-2)]
                else: runChip = layer1[chip_in_ladder+6*recoLadder]

            simChip = 0
            simChip |= layer<<10
            simChip |= simLadder<<5
            simChip |= (chip_in_ladder+1)

            confChip = calculateConfChip(layer, ladder, chip_in_ladder)
            dataLinks = [(confChip % 12) * 3 + i for i in range(3)]

            print("    chip {}, simChip {}, runChip {}, confLadder {}, confChip {}, data link A {}, B {}, C {}".format(chip_in_ladder, simChip, runChip, \
                calculateConfLadder(layer, ladder, chip_in_ladder), confChip, dataLinks[0], dataLinks[1], dataLinks[2]))

            assert(runChip == confChip) # for now
            chipAlignment = alignmentData[str(confChip)]

            tracker["chips"][str(unique_ctr)] = {
                "confChip": confChip,
                "runChip": runChip,
                "ladderChip": chip_in_ladder,
                "simChip": simChip,
                "layer": layer,
                "direction": "US" if chip_in_ladder <= 2 else "DS",
                "localLadder": ladder,
                "confLadder": calculateConfLadder(layer, ladder, chip_in_ladder),
                "simLadder": recoLadder,
                "dataLinkA_onFEB": dataLinks[0],
                "dataLinkB_onFEB": dataLinks[1],
                "dataLinkC_onFEB": dataLinks[2],
                "dataLinkD_onFEB": -1,
                "phi": np.arctan2(chipAlignment["v"]["y"], chipAlignment["v"]["x"]),
                "alignment" : {
                    "v": {
                        "x": chipAlignment["v"]["x"],
                        "y": chipAlignment["v"]["y"],
                        "z": chipAlignment["v"]["z"]
                    },
                    "drow": {
                        "x": chipAlignment["drow"]["x"],
                        "y": chipAlignment["drow"]["y"],
                        "z": chipAlignment["drow"]["z"]
                    },
                    "dcol": {
                        "x": chipAlignment["dcol"]["x"],
                        "y": chipAlignment["dcol"]["y"],
                        "z": chipAlignment["dcol"]["z"]
                    },
                    "nrow": chipAlignment["nrow"],
                    "ncol": chipAlignment["ncol"],
                    "width": chipAlignment["width"],
                    "length": chipAlignment["length"],
                    "thickness": chipAlignment["thickness"],
                    "pixelSize": chipAlignment["pixelSize"]
                }
            }
            unique_ctr += 1

with open('tracker_chip_configuration.json', 'w') as outfile:
    json.dump(dict(tracker), outfile, indent=4)
