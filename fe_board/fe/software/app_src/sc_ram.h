
struct sc_ram_t {
    union {
        alt_u32 __cmds[16];
        struct {
            alt_u32 length;
            alt_u32 offset;
//            ...
        } cmds;
    };
    alt_u32 data[256*256 - 16 - 256];
    union {
        alt_u32 __regs[256];
        struct {
            alt_u32 resets;
            alt_u32 reserved[256 - 1];
        } regs;
    };
};
