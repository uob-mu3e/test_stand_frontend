import midas.client
import sys


if len(sys.argv) != 3:
    print("Usage: python3 mask_single_link.py [chip number] [link: 0=A, 1=B, 2=C]")

chip = int(sys.argv[1])
link = int(sys.argv[2])

if chip > 120:
    print("Chip number (argument 1) must be between 0 and 119")
if link > 2:
    print("Chip link (argument 2) must be between 0 and 2")

client = midas.client.MidasClient("px_Mask_Links")

feb = int(chip/12)

#if chip >= 60:
#    link = 2 - link

link_pos = (chip - feb*12)*3 + link

print("The position of the link is", link_pos)

additional_string = ""

if link_pos < 32:
    curr_mask = client.odb_get("/Equipment/PixelsCentral/Settings/FEBS/" + str(feb) + "/MP_LVDS_LINK_MASK")
else:
    curr_mask = client.odb_get("/Equipment/PixelsCentral/Settings/FEBS/" + str(feb) + "/MP_LVDS_LINK_MASK2")
    link_pos = link_pos - 32
    additional_string = "2"

new_mask =  curr_mask & (~(1 << link_pos))

print("Mask of FEB", feb, "is going to be changed from", hex(curr_mask), "to", hex(new_mask))

client.odb_set("/Equipment/PixelsCentral/Settings/FEBS/" + str(feb) + "/MP_LVDS_LINK_MASK" + str(additional_string), new_mask)
