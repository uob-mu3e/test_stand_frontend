/**
 * basic interface to a mudaq readout board (stratix 5 pcie dev board)
 *
 * @author      Moritz Kiehn <kiehn@physi.uni-heidelberg.de>
 * @author      Heiko Augustin <augustin.heiko@physi.uni-heidelberg.de>
 * @author      Lennart Huth <huth@physi.uni-heidelberg.de>
 * @date        2013-11-14
 */

#ifndef __MUDAQ_DEVICE_HPP_WKZIQD9F__
#define __MUDAQ_DEVICE_HPP_WKZIQD9F__

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <sys/ioctl.h>		/* ioctl */

#include <boost/dynamic_bitset.hpp>

#include "mudaq_circular_buffer.hpp"
#include "../include/mudaq_device_constants.h"
#include "../include/mudaq_registers.h"
#include "utils.hpp"
#include "time.h"
#include <stdlib.h>
#include <stdio.h>
#include <list>
#include <vector>
/*??????*/
#include "../kerneldriver/mudaq.h"

static size_t _pagesize(void) { return static_cast<size_t>(sysconf(_SC_PAGESIZE)); }
int physical_address_check( uint32_t * virtual_address, size_t size );
int is_page_aligned( void * pointer );
void * align_page( void * pointer );

namespace mudaq {

  class MudaqDevice
  {
  public:
    union TimingConfig
    {
      uint8_t values[8];
      struct fields{
    uint8_t data_sampling_point;
    uint8_t priout_sampling_point;
    uint8_t rdcol_width;
    uint8_t rdcol_pulldown_delay;
    uint8_t rdcol_rdcol_delay;
    uint8_t ldcol_rdcol_delay;
    uint8_t pulldown_ldcol_delay;
    uint8_t ldpix_pulldown_delay;
      };
    };

    // a device can exist only once. forbid copying and assignment
    MudaqDevice() = delete;
    MudaqDevice(const MudaqDevice&) = delete;
    MudaqDevice& operator=(const MudaqDevice&) = delete;

    MudaqDevice(const std::string& path);
    virtual ~MudaqDevice() { close(); }

    bool is_ok() const;
    virtual bool open();
    virtual void close();
    virtual bool operator!() const;

    void write_register(unsigned idx, uint32_t value);
    void write_register_wait(unsigned idx, uint32_t value, unsigned wait_ns);
    void toggle_register(unsigned idx, uint32_t value, unsigned wait_ns);
    uint32_t read_register_rw(unsigned idx) const;
    uint32_t read_register_ro(unsigned idx) const;
    uint32_t read_memory_ro(unsigned idx) const;
    uint32_t read_memory_rw(unsigned idx) const;
    void write_memory_rw(unsigned idx, uint32_t value); // added by DvB for rw mem

    void FEBsc_resetMaster();
    void FEBsc_resetSlave();
    //write slow control packet with payload of length words in data. Returns 0 on success, -1 on error receiving acknowledge packet.
    //Disable any check of reply by setting the request_reply to false, then only the write request will be sent.
    int FEBsc_write(uint32_t FPGA_ID, uint32_t* data, uint16_t length, uint32_t startaddr, bool request_reply=true);
    //request write slow control read packet, with payload of length words saved in data. Returns length of packet returned, -1 on error receiving reply packet
    //Disable any check of reply by setting the request_reply to false, then only the read request will be sent.
    int FEBsc_read(uint32_t FPGA_ID, uint32_t* data, uint16_t length, uint32_t startaddr, bool request_reply=true);
    //write all packets received into a midas bank. clears internal packet fifo and should be called from time to time to avoid storing all replies
    int FEBsc_write_bank(char *pevent, int off);
    //get
    uint32_t FEBsc_get_packet();

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

    //FEB slow control
    uint32_t m_FEBsc_wmem_addr;
    uint32_t m_FEBsc_rmem_addr;
    struct SC_reply_packet : public std::vector<uint32_t>{
    public:
        bool Good(){
            //header+startaddr+length+trailer+[data]
            if(size()<4) return false;
            if(IsWR()&&IsResponse()) return size()==4; //No payload for write response
            if(size()!=GetLength()+4) return false;
            return true;
            };
        bool IsOOB(){return (this->at(0)&0x1f0000bc) == 0x1c0000bc;};
        bool IsRD() {return (this->at(0)&0x1f0000bc) == 0x1e0000bc;};
        bool IsWR() {return (this->at(0)&0x1f0000bc) == 0x1f0000bc;};
        uint16_t IsResponse(){return (this->at(2)&0x10000)!=0;};
        uint16_t GetFPGA_ID(){return (this->at(0)>>8)&0xffff;};
        uint16_t GetStartAddr(){return this->at(1);};
        size_t GetLength(){if(IsWR() && IsResponse()) return 0; else return this->at(2)&0xffff;};


    };
    std::list<SC_reply_packet> m_sc_packet_fifo; //storage of all received SC packets to be consumed by a MIDAS bank writer

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

  private:
    volatile uint32_t * _dmabuf_ctrl;

    uint32_t _last_interrupt;   // updated in read_block() function
    unsigned _last_end_of_buffer;
  };

} // namespace mudaq

#endif // __MUDAQ_DEVICE_HPP_WKZIQD9F__
