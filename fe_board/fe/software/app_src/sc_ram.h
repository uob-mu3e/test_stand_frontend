#ifndef __FE_SC_RAM_H__
#define __FE_SC_RAM_H__

struct sc_ram_t {
    alt_u32 data[AVM_SC_SPAN/4 - 256];

    struct regs_t {
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
                alt_u32 status;
                alt_u32 rx_dpa_lock;
                alt_u32 rx_ready;
                alt_u32 reserved[1];
            } mon;

            struct {
                alt_u32 dummy;
                alt_u32 dp;
                alt_u32 reset;
                alt_u32 reserved[1];
            } ctrl;

            alt_u32 reserved4[16 - 12];
        } malibu;

        alt_u32 reserved5[16 - 0];

        struct {
            struct {
                alt_u32 data;
                alt_u16 tag;
                alt_u16 status;
                alt_u32 reserved[2];
            } fifo;

            struct {
                alt_u32 status;
                alt_u32 rx_dpa_lock;
                alt_u32 rx_ready;
                alt_u32 reserved[1];
            } mon;

            struct {
                alt_u32 dummy;
                alt_u32 dp;
                alt_u32 reset;
                alt_u32 reserved[1];
            } ctrl;

            alt_u32 reserved6[16 - 12];
        } scifi;

        alt_u32 reserved7[16 - 0];

        struct {
            alt_u32 reserved8[16 - 0];
        } mupix;

        alt_u32 reserved9[16 - 0];

        alt_u32 reserved10[16 - 0];

        alt_u32 reserved11[16 - 0];

        alt_u32 reserved12[16 - 0];

        alt_u32 reserved13[16 - 0];

        alt_u32 reserved14[16 - 0];

        struct {
            alt_u32 cmdlen;
            alt_u32 offset;
            alt_u32 reserved[2];
            alt_u32 reset_bypass;
            alt_u32 reserved15[16 - 5];
        } fe;
    } regs;
};

static_assert(sizeof(sc_ram_t) == 65536 * 4, "");

static_assert(sizeof(sc_ram_t::regs) == 256 * 4, "");
static_assert(sizeof(sc_ram_t::regs_t::malibu) == 16 * 4, "");
static_assert(sizeof(sc_ram_t::regs_t::mupix) == 16 * 4, "");
static_assert(sizeof(sc_ram_t::regs_t::scifi) == 16 * 4, "");
static_assert(sizeof(sc_ram_t::regs_t::fe) == 16 * 4, "");

static_assert(offsetof(sc_ram_t, regs) % 64 == 0, "");
static_assert(offsetof(sc_ram_t::regs_t, malibu) % 64 == 0, "");
static_assert(offsetof(sc_ram_t::regs_t, mupix) % 64 == 0, "");
static_assert(offsetof(sc_ram_t::regs_t, scifi) % 64 == 0, "");
static_assert(offsetof(sc_ram_t::regs_t, fe) % 64 == 0, "");

#endif // __FE_SC_RAM_H__
