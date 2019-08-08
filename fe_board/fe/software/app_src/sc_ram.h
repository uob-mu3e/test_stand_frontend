
struct sc_ram_t {
    union {
        alt_u32 data[AVM_SC_SPAN/4 - 256];
        struct {
            alt_u16 len;
            alt_u16 cmd;
            alt_u32 offset;
            alt_u32 reserved[16 - 2];
        } ctrl;
    };

    struct {
        alt_u32 reserved0[16 - 0];

        alt_u32 reserved1[16 - 0];

        alt_u32 reserved2[16 - 0];

        alt_u32 reserved3[16 - 0];

        struct {
            struct {
                alt_u32 data;
                alt_u16 tag;
                alt_u16 status;
                alt_u32 reserved[2];
            } fifo;

            struct {
                alt_u32 pll_lock;
                alt_u32 ready;
                alt_u32 dpa_lock;
                alt_u32 reserved[1];
            } rx;

            struct {
                alt_u32 desync;
                alt_u32 reserved[3];
            } frame;

            alt_u32 reserved4[16 - 12];
        } malibu;

        struct {
            alt_u32 reserved5[16 - 0];
        } mupix;

        struct {
            alt_u32 reserved6[16 - 0];
        } scifi;

        alt_u32 reserved7[16 - 0];

        alt_u32 reserved8[16 - 0];

        alt_u32 reserved9[16 - 0];

        alt_u32 reserved10[16 - 0];

        alt_u32 reserved11[16 - 0];

        alt_u32 reserved12[16 - 0];

        alt_u32 reserved13[16 - 0];

        alt_u32 reserved14[16 - 0];

        alt_u32 reserved15[16 - 0];
    } regs;
};

static_assert(sizeof(sc_ram_t) == 65536 * 4, "");
static_assert(sizeof(sc_ram_t::ctrl) == 16 * 4, "");
static_assert(sizeof(sc_ram_t::regs) == 256 * 4, "");
static_assert(sizeof(sc_ram_t::regs.malibu) == 16 * 4, "");
static_assert(sizeof(sc_ram_t::regs.mupix) == 16 * 4, "");
static_assert(sizeof(sc_ram_t::regs.scifi) == 16 * 4, "");
static_assert(sizeof(sc_ram_t::regs.fe) == 16 * 4, "");
