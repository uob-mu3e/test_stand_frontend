/* The frontend name (client name) as seen by other MIDAS clients   */
const char *frontend_name = "ROSWBCentral";

const string eq_name = "ROSWBCentral";

const uint16_t eventID = 0x1;

EQUIPMENT equipment[] = {

   {"ROSWBCentral",                /* equipment name */
    {1, 0,                   /* event ID, trigger mask */
     "SYSTEM",               /* event buffer */
     EQ_USER,                /* equipment type */
     0,                      /* event source crate 0, all stations */
     "MIDAS",                /* format */
     TRUE,                   /* enabled */
     RO_RUNNING,             /* read while running and stopped but not at transitions and update ODB */
     1000,                    /* poll for 1s */
     0,                      /* stop run after this event limit */
     0,                      /* number of sub events */
     0,                      /* don't log history */
     "", "", "",},
     NULL,                    /* readout routine */
    },
   {""}
};
