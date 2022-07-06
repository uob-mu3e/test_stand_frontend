/* Constants and definitions for MIDAS banks */


#ifndef MU3EBANKS_H
#define MU3EBANKS_H

#include <array>
#include <string>
#include <string>

#include "odbxx.h"

#include "link_constants.h"

using std::array;
using std::string;
using std::to_string;

using midas::odb;


namespace mu3ebanks{
////////////// Switching board

//// SSFE
constexpr int per_fe_SSFE_size = 26;
const array<const string, MAX_N_SWITCHINGBOARDS> ssfe = {"SCFE","SUFE","SDFE","SFFE"};
const array<const string, MAX_N_SWITCHINGBOARDS> ssfenames = {"Names SCFE","Names SUFE","Names SDFE","Names SFFE"};

void create_ssfe_names_in_odb(odb & settings, int switch_id);

//// SCFC
constexpr int per_crate_SCFC_size = 21;

void create_scfc_names_in_odb(odb crate_settings);

//// SSSO
constexpr int max_sorter_inputs_per_feb = 12;
constexpr int num_sorter_counters_per_feb = 3*max_sorter_inputs_per_feb +2;
constexpr int per_fe_SSSO_size = num_sorter_counters_per_feb + 1;
const array<const string, MAX_N_SWITCHINGBOARDS> ssso = {"SCSO","SUSO","SDSO","SFSO"};
const array<const string, MAX_N_SWITCHINGBOARDS> sssonames = {"Names SCSO","Names SUSO","Names SDSO","Names SFSO"};

void create_ssso_names_in_odb(odb & settings, int switch_id);

//// SSCN
constexpr int num_cosmic_febs = 10; // for the cosmic run we only readout the 10 pixel febs
constexpr int num_swb_counters_global = 6;
constexpr int num_swb_counters_data_path = 8;
constexpr int num_swb_counters_per_feb = 9;
constexpr int num_swb_counters_per_tree_layer = 1;
constexpr int num_swb_tree_layers = 3;
constexpr int out_nodes_per_tree_layer[num_swb_tree_layers] = { 4, 2, 1 };
constexpr int num_swb_counters_per_node = 4;
constexpr int num_swb_counters_node_total = 4 * 4 + 2 * 4 + 1 * 4;
 // TODO: the detector parts need to be added to the read_counters function
constexpr int num_swb_detector_types = 2;
const array<const string, MAX_N_SWITCHINGBOARDS> sscn = {"SCCN","SUCN","SDCN","SFCN"};
const array<const string, MAX_N_SWITCHINGBOARDS> sscnnames = {"Names SCCN","Names SUCN","Names SDCN","Names SFCN"};

void create_sscn_names_in_odb(odb & settings, int switch_id);

constexpr int ssplsize = 4;

const array<const string, MAX_N_SWITCHINGBOARDS> sspl = {"SCPL","SUPL","SDPL","SFPL"};
const array<const string, MAX_N_SWITCHINGBOARDS> ssplnames = {"Names SCPL","Names SUPL","Names SDPL","Names SFPL"};

void create_sspl_names_in_odb(odb & settings, int switch_id);

constexpr uint32_t per_fe_PSLS_size = MAX_LVDS_LINKS_PER_FEB*6 +2;

// The fibre switching board should not produce these pixel banks, thus XXXX
const array <const string, MAX_N_SWITCHINGBOARDS> psls = {"PCLS","PULS","PDLS","XXXX"};
const array <const string, MAX_N_SWITCHINGBOARDS> pslsnames = {"Names PCLS","Names PULS","Names PDLS","Names XXXX"};

void create_psls_names_in_odb(odb & settings, int switch_id, uint32_t n_febs_mupix);
}

#endif // MU3EBANKS_H
