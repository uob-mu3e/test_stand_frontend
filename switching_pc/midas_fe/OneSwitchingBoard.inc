// Definitions for the all-powerful integration switching board (all subdetectors)

constexpr int switch_id = 0;
// 0 - Central
// 1 - Recurl US
// 2 - Recurl DS
// 3 - Fibres

constexpr bool has_scifi = true;
constexpr bool has_tiles = true;
constexpr bool has_pixels = true;

const char *frontend_name = "SW Central";
const string fe_name = "SW Central";
const string eq_name = "SwitchingCentral";
const string link_eq_name = "LinksCentral";
const string scifi_eq_name = "ScifiCentral";
const string tile_eq_name = "TilesCentral";
const string pixel_eq_name = "PixelsCentral";

/*-- Equipment list ------------------------------------------------*/
enum EQUIPMENT_ID {Switching=0,Links, SciFi,Tiles,Pixels};
constexpr int NEQUIPMENT = 5;
EQUIPMENT equipment[NEQUIPMENT+1] = {
   {"SwitchingCentral",     /* equipment name */
    {103, 0,                    /* event ID, trigger mask */
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
   {"LinksCentral",         /* equipment name */
    {107, 0,                    /* event ID, trigger mask */
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
   {"SciFiCentral",         /* equipment name */
    {112, 0,                    /* event ID, trigger mask */
     "SYSTEM",                  /* event buffer */
     EQ_PERIODIC,                 /* equipment type */
     0,                         /* event source crate 0, all stations */
     "MIDAS",                   /* format */
     FALSE,                      /* enabled */
     RO_ALWAYS | RO_ODB,        /* read always and update ODB */
     10000,                      /* read every 10 sec */
     0,                         /* stop run after this event limit */
     0,                         /* number of sub events */
     1,                         /* log history every event */
     "", "", "",},
     read_scifi_sc_event,          /* readout routine */
    },
   {"TilesCentral",       /* equipment name */
    {113, 0,                      /* event ID, trigger mask */
     "SYSTEM",                  /* event buffer */
     EQ_PERIODIC,                 /* equipment type */
     0,                         /* event source crate 0, all stations */
     "MIDAS",                   /* format */
     FALSE,                      /* enabled */
     RO_ALWAYS | RO_ODB,        /* read always and update ODB */
     10000,                      /* read every 10 sec */
     0,                         /* stop run after this event limit */
     0,                         /* number of sub events */
     1,                         /* log history every event */
     "", "", "",},
     read_scitiles_sc_event,          /* readout routine */
    },
    {"PixelsCentral",         /* equipment name */
    {111, 0,                      /* event ID, trigger mask */
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

