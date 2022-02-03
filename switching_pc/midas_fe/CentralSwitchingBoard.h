// Definitions for the all-powerful integration switching board (all subdetectors)

constexpr int switch_id = 0;
// 0 - Central
// 1 - Recurl US
// 2 - Recurl DS
// 3 - Fibres

constexpr bool has_scifi = false;
constexpr bool has_tiles = false;
constexpr bool has_pixels = true;

const string fe_name = "SW Central";
const string eq_name = "SwitchingC";
const string link_eq_name = "LinksC";
const string scifi_eq_name = "Njet";
const string tile_eq_name = "Njet";
const string pixel_eq_name = "PixelsC";

/*-- Equipment list ------------------------------------------------*/
enum EQUIPMENT_ID {Switching=0,Links, Mupix};
constexpr int NEQUIPMENT = 3;
EQUIPMENT equipment[NEQUIPMENT+1] = {
   {"SwitchingC",                /* equipment name */
    {110, 0,                    /* event ID, trigger mask */
     "SYSTEM",                  /* event buffer */
     EQ_PERIODIC,               /* equipment type */
     0,                         /* event source */
     "MIDAS",                   /* format */
     TRUE,                      /* enabled */
     RO_ALWAYS | RO_ODB,        /* read always and update ODB */
     1000,                      /* read every 1 sec */
     0,                         /* stop run after this event limit */
     0,                         /* number of sub events */
     1,                         /* log history every event */
     "", "", ""},
     read_sc_event,             /* readout routine */
   },
   {"LinksC",                /* equipment name */
    {110, 0,                    /* event ID, trigger mask */
     "SYSTEM",                  /* event buffer */
     EQ_PERIODIC,               /* equipment type */
     0,                         /* event source */
     "MIDAS",                   /* format */
     TRUE,                      /* enabled */
     RO_ALWAYS | RO_ODB,        /* read always and update ODB */
     1000,                      /* read every 1 sec */
     0,                         /* stop run after this event limit */
     0,                         /* number of sub events */
     1,                         /* log history every event */
     "", "", ""},
     read_sc_event,             /* readout routine */
   },
    {"MupixC",                    /* equipment name */
    {113, 0,                      /* event ID, trigger mask */
     "SYSTEM",                  /* event buffer */
     EQ_PERIODIC,                 /* equipment type */
     0,                         /* event source crate 0, all stations */
     "MIDAS",                   /* format */
     TRUE,                      /* enabled */
     RO_ALWAYS | RO_ODB,   /* read during run transitions and update ODB */
     1000,                      /* read every 1 sec */
     0,                         /* stop run after this event limit */
     0,                         /* number of sub events */
     1,                         /* log history every event */
     "", "", "",},
     read_mupix_sc_event,          /* readout routine */
    },
    {""}
};

