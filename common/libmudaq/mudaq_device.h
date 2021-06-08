/**
 * basic interface to a mudaq readout board (stratix 5 pcie dev board)
 *
 * @author      Moritz Kiehn <kiehn@physi.uni-heidelberg.de>
 * @author      Heiko Augustin <augustin.heiko@physi.uni-heidelberg.de>
 * @author      Lennart Huth <huth@physi.uni-heidelberg.de>
 * @date        2013-11-14
 */

#ifndef MUDAQ_DEVICE_HPP
#define MUDAQ_DEVICE_HPP

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <sys/ioctl.h>

#include <boost/dynamic_bitset.hpp>

#include "mudaq_circular_buffer.hpp"
#include "../include/mudaq_device_constants.h"
#include "../include/switching_constants.h"
#include "utils.hpp"
#include <stdlib.h>
#include <stdio.h>
#if __APPLE__
    #define REQUEST_INTERRUPT_COUNTER     _IOR(MUDAQ_IOC_TYPE, 1, int)
    #define MUDAQ_IOC_TYPE 102
    struct mesg {
        volatile void * address;
        size_t size;
     };
    #define MAP_DMA                       _IOW(MUDAQ_IOC_TYPE, 0, struct mesg)
#else
    #include "../kerneldriver/mudaq.h"
#endif

[[maybe_unused]] static size_t _pagesize(void) { return static_cast<size_t>(sysconf(_SC_PAGESIZE)); }
int physical_address_check( uint32_t * virtual_address, size_t size );
int is_page_aligned( void * pointer );
void * align_page( void * pointer );

namespace mudaq {

  class MudaqDevice
  {
  public:
    // a device can exist only once. forbid copying and assignment
    MudaqDevice() = delete;
    MudaqDevice(const MudaqDevice&) = delete;
    MudaqDevice& operator=(const MudaqDevice&) = delete;

    MudaqDevice(const std::string& path);
    virtual ~MudaqDevice() { close(); }

    virtual bool is_ok() const;
    virtual bool open();
    virtual void close();
    virtual bool operator!() const;

    virtual void write_register(unsigned idx, uint32_t value);
    virtual void write_register_wait(unsigned idx, uint32_t value, unsigned wait_ns);
    virtual void toggle_register(unsigned idx, uint32_t value, unsigned wait_ns);
    virtual uint32_t read_register_rw(unsigned idx) const;
    virtual uint32_t read_register_ro(unsigned idx) const;
    virtual uint32_t read_counters(uint32_t write_value, unsigned read_value) const;
    virtual uint32_t read_memory_ro(unsigned idx) const;
    virtual uint32_t read_memory_rw(unsigned idx) const;
    virtual void write_memory_rw(unsigned idx, uint32_t value);

    void enable_led(unsigned which);
    void enable_leds(uint8_t pattern);
    void disable_leds();

    void print_registers();

protected:
    volatile uint32_t* mmap_rw(unsigned idx, unsigned len);
    volatile uint32_t* mmap_ro(unsigned idx, unsigned len);
    void munmap_wrapper(uint32_t** addr, unsigned len,
                        const std::string& error_msg);
    void munmap_wrapper(volatile uint32_t** addr, unsigned len,
                        const std::string& error_msg);

    // needs to be accessible by the dma readout subtype
    int _fd;

private:
    const std::string       _path;
    volatile uint32_t*      _regs_rw;
    volatile uint32_t*      _regs_ro;
    volatile uint32_t*      _mem_ro;
    volatile uint32_t*      _mem_rw;  // added by DvB for rw mem

    uint16_t                _last_read_address; // for reading command of slow control

    friend std::ostream& operator<<(std::ostream&, const MudaqDevice&);
};


class DmaMudaqDevice : public MudaqDevice
{
public:
    typedef CircularBufferProxy<MUDAQ_DMABUF_DATA_ORDER_WORDS> DataBuffer;
    typedef CircularSubBufferProxy<MUDAQ_DMABUF_DATA_ORDER_WORDS> DataBlock;
    enum {
        READ_ERROR,
        READ_TIMEOUT,
        READ_NODATA,
        READ_SUCCESS
    };

    // a device can exist only once. forbid copying and assignment
    DmaMudaqDevice() = delete;
    DmaMudaqDevice(const DmaMudaqDevice &) = delete;
    DmaMudaqDevice & operator=(const DmaMudaqDevice &) = delete;

    DmaMudaqDevice(const std::string & path);

    int enable_continous_readout(int interTrue);


    void disable();

    virtual bool open();
    virtual void close();
    virtual bool operator!() const;

    int read_block(DataBlock& buffer, volatile uint32_t * pinned_data );
    volatile uint32_t* dma_buffer();
    int map_pinned_dma_mem( struct mesg user_message );
    int pinned_mem_is_mapped();
    int get_current_interrupt_number();  // via ioctl from driver (no need for read_block() function)

    uint32_t last_written_addr() const;
    uint32_t last_endofevent_addr() const;

  private:
    volatile uint32_t * _dmabuf_ctrl;

    uint32_t _last_interrupt;   // updated in read_block() function
    unsigned _last_end_of_buffer;
  };

} // namespace mudaq

#endif // __MUDAQ_DEVICE_HPP_WKZIQD9F__
