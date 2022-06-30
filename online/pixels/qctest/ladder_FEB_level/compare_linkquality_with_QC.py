import json
import matplotlib.pyplot as plt
import midas.client
import time
import numpy

def plot_error_rate(x, a, b, c):
    plt.yscale("log")  

    plt.plot(x, a)
    plt.plot(x, b)
    plt.plot(x, c)

    plt.legend(["Error Rate Link A", "Error Rate Link B", "Error Rate Link C"])
    plt.show()

def get_errors_from_json(json_node, chip_number, format_file = "ladder"):

    x, a, b, c = [], [], [], []
    if format_file == "ladder":
        x = json_node["LINKQUALIcheck"]["Output"][str(chip_number)]["Scan"]["VPVCO"]
        a = json_node["LINKQUALIcheck"]["Output"][str(chip_number)]["Scan"]["error_rate_linkA"]
        b = json_node["LINKQUALIcheck"]["Output"][str(chip_number)]["Scan"]["error_rate_linkB"]
        c = json_node["LINKQUALIcheck"]["Output"][str(chip_number)]["Scan"]["error_rate_linkC"]
    elif format_file == "feb":
        x = json_node["Output"][str(chip_number)]["Scan"]["VPVCO"]
        a = json_node["Output"][str(chip_number)]["Scan"]["error_rate_linkA"]
        b = json_node["Output"][str(chip_number)]["Scan"]["error_rate_linkB"]
        c = json_node["Output"][str(chip_number)]["Scan"]["error_rate_linkC"]
    else:
        print("get_errors_from_json: File format", format_file, "not recognized, returning empty vectors")
        
    return x, a, b, c

def correct_errors(a, b, c):
    return  list(map(abs, a)), list(map(abs, b)), list(map(abs, c))

def get_link_quality(a, b, c, limit = 0):
    quality = []
    for i, a0 in enumerate(a):
        count = 0
        if a[i] <= limit:
            count += 1
        if b[i] <= limit:
            count += 1
        if c[i] <= limit:
            count += 1
        quality.append(count)
    return quality
    
def find_opt_index(q):
    max_link_working = max(q)
    if max_link_working == 0:
        print("find_opt_value: The chip has always 0 link working, it can not be operated")
        return -1
    
    pl_length = []
    pl_liminf = []
    pl_limsup = []
    on_pl = False
    for i, v in enumerate(q):
        if v == max_link_working and on_pl == False: #first value of a new plateau
            pl_length.append(1)
            pl_liminf.append(i)
            on_pl = True
        if v == max_link_working and on_pl == True: #prosecuting the last plateau
            pl_length[-1] += 1
        if v < max_link_working and on_pl == True: #exited the last plateau
            pl_limsup.append(i-1)
            on_pl = False
    if on_pl == True: #still on a plateau
        pl_length[-1] += 1
        pl_limsup.append(len(q)-1)
        on_pl = False

    print ("find_opt_value parsed with results: pl_liminf =", pl_liminf, ", pl_limsup =", pl_limsup, ", pl_length =", pl_length)
    max_value = max(pl_length)
    max_index = pl_length.index(max_value)
    liminf = pl_liminf[max_index]
    limsup = pl_limsup[max_index]
    return int((limsup + liminf)/2)

def produce_link_mask(nwl):
    mask = 0
    for l in nwl:
        mask = mask | (1 << l)
    return (mask & 0xffffffff), ((mask & 0xf00000000) >> 32)

client = midas.client.MidasClient("px_QC_analysis")

plt_legend = []

feb_number = 0
file_number = 7

with open('Output/Link_quality_output_file' + str(file_number) + '.json') as infile:
    out_feb_0 = json.load(infile)

n_working_links = 0

not_working_links = []

error_rate = []

tmp_err = client.odb_get("/Equipment/PixelsCentral/Variables/PCLS")
print("Lenght of PCLS =", len(tmp_err))
#print(tmp_err)
for feb_id in range(10):
    for linkn in range(36):
        lvds = (linkn)*6 + 2*(feb_id+1) + 216*feb_id
        error_rate.append(numpy.int64(tmp_err[lvds]))

print("Error inital:")
print(error_rate)
print("Sleeping 10 seconds")
time.sleep(10)

tmp_err2 = client.odb_get("/Equipment/PixelsCentral/Variables/PCLS")
for feb_id2 in range(10):
    for linkn2 in range(36):
        lvds2 = (linkn2)*6 + 2*(feb_id2+1) + 216*feb_id2
        err_prev = error_rate[feb_id2*36 + linkn2]
        error_rate[feb_id2*36 + linkn2] = numpy.int64(tmp_err2[lvds2]) - err_prev

print("Error final:")
print(error_rate)

gg = 0
gb = 0
bg = 0
bb = 0

for chip_number in range(120):

    if chip_number >= 54 and chip_number < 60:
        continue
    if chip_number >= 114:
        continue


    g_chip_number = chip_number + feb_number*12
    xf0, af0_, bf0_, cf0_ = get_errors_from_json(out_feb_0, chip_number, "feb")
    af0, bf0, cf0 = correct_errors(af0_, bf0_, cf0_)
    #plot_error_rate(xf0, af0, bf0, cf0)

    qf0 = get_link_quality(af0, bf0, cf0, 10)

    ax = plt.gca()
    plt.yscale("linear")  
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(xf0, qf0, color = color)
    plt_legend.append("# Good Links " + str(g_chip_number))
    opt_index = find_opt_index(qf0)
    n_working_links += max(qf0)

    if chip_number % 12 == 0:
        not_working_links.append([])


    if af0[opt_index] > 10:
        if abs(error_rate[(chip_number)*3 + 0]) > 10:
            print("Chip", chip_number, " link A : bad and confirmed")
            bb += 1
            not_working_links[-1].append((chip_number%12)*3 + 0)
        else:
            print("Chip", chip_number, " link A : bad and now good")
            bg += 1
    else:
        if abs(error_rate[(chip_number)*3 + 0]) > 10:
            print("Chip", chip_number, " link A : good and now bad")
            gb += 1
            not_working_links[-1].append((chip_number%12)*3 + 0)
        else:
            print("Chip", chip_number, " link A : good and confirmed")
            gg += 1

    if bf0[opt_index] > 10:
        if abs(error_rate[(chip_number)*3 + 1]) > 10:
            print("Chip", chip_number, " link B : bad and confirmed")
            bb += 1
            not_working_links[-1].append((chip_number%12)*3 + 1)
        else:
            print("Chip", chip_number, " link B : bad and now good")
            bg += 1
    else:
        if abs(error_rate[(chip_number)*3 + 1]) > 10:
            print("Chip", chip_number, " link B : good and now bad")
            gb += 1
            not_working_links[-1].append((chip_number%12)*3 + 1)
        else:
            print("Chip", chip_number, " link B : good and confirmed")
            gg += 1

    if cf0[opt_index] > 10:
        if abs(error_rate[(chip_number)*3 + 2]) > 10:
            print("Chip", chip_number, " link C : bad and confirmed")
            bb += 1
            not_working_links[-1].append((chip_number%12)*3 + 2)
        else:
            print("Chip", chip_number, " link C : bad and now good")
            bg += 1
    else:
        if abs(error_rate[(chip_number)*3 + 2]) > 10:
            print("Chip", chip_number, " link C : good and now bad")
            gb += 1
            not_working_links[-1].append((chip_number%12)*3 + 2)
        else:
            print("Chip", chip_number, " link C : good and confirmed")
            gg += 1

    opt_value = xf0[opt_index]
    #print("Chip Number", g_chip_number, " : optimal VPVCO =", opt_value)
    max_value = max(qf0)

print("Summary:")
print(gg, "links worked on QC and work now               (", gg*100/324., "%)")
print(gb, "links worked on QC and do not work now        (", gb*100/324., "%)")
print(bg, "links did not work on QC and work now,        (", bg*100/324., "%)")
print(bb, "links did not work on QC and do not work now, (", bb*100/324., "%)")


print("Not working liks:", not_working_links)

for feb, nwl in enumerate(not_working_links):
    print("Feb", feb, ": not working links:", nwl)
    mask, mask2 = produce_link_mask(nwl)
    print("---> mask =", mask, "= 0x", hex(mask), "= 0b", "{0:b}".format(mask))
    client.odb_set("/Equipment/PixelsCentral/Settings/FEBS/" + str(feb) + "/MP_LVDS_LINK_MASK", mask)
    print("---> mask2 =", mask2, "= 0x", hex(mask2), "= 0b", "{0:b}".format(mask2))
    client.odb_set("/Equipment/PixelsCentral/Settings/FEBS/" + str(feb) + "/MP_LVDS_LINK_MASK2", mask2)

