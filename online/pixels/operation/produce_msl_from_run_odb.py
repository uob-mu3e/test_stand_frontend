import json
import sys

run_number = 1047

if len(sys.argv) == 2:
    run_number = int(sys.argv[1])

with open('/data1/run2022/data/run0' + str(run_number) + '.json') as infile:
    odb = json.load(infile)


outfile = open("setup_odb_links.msl", 'w')
outfile.write("COMMENT File to reproduce VPVCO, VNVCO and FEB links as in run " + str(run_number) + "\n")
outfile.write("\n")

bias_dacs = odb["Equipment"]["PixelsCentral"]["Settings"]["BIASDACS"]

for chip in range(120):
    #print(bias_dacs[str(chip)])
    outfile.write("ODBSET /Equipment/PixelsCentral/Settings/BIASDACS/" + str(chip) + "/VPVCO, " + str(bias_dacs[str(chip)]["VPVCO"]) + "\n")
    outfile.write("ODBSET /Equipment/PixelsCentral/Settings/BIASDACS/" + str(chip) + "/VNVCO, " + str(bias_dacs[str(chip)]["VNVCO"]) + "\n")

outfile.write("\n")


febs =  odb["Equipment"]["PixelsCentral"]["Settings"]["FEBS"]

#print(febs)

for feb in range(10):
    outfile.write("ODBSET /Equipment/PixelsCentral/Settings/FEBS/" + str(feb) + "/MP_LVDS_LINK_MASK, " + str(febs[str(feb)]["MP_LVDS_LINK_MASK"]) + "\n")
    outfile.write("ODBSET /Equipment/PixelsCentral/Settings/FEBS/" + str(feb) + "/MP_LVDS_LINK_MASK2, " + str(febs[str(feb)]["MP_LVDS_LINK_MASK2"]) + "\n")

outfile.write("\n")
outfile.write("ODBSET /Equipment/PixelsCentral/Settings/CONFDACS/*/AlwaysEnable, 0")
outfile.write("ODBSET /Equipment/PixelsCentral/Settings/VDACS/*/ThLow, 179")

outfile.close()
