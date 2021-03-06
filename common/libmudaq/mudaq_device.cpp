/**
 * @author      Moritz Kiehn <kiehn@physi.uni-heidelberg.de>
 * @author      Heiko Augustin <augustin.heiko@physi.uni-heidelberg.de>
 * @author      Lennart Huth <huth@physi.uni-heidelberg.de>
 * @date        2013-11-14
 */

#include <atomic>
#include <cmath>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <time.h>
#include <cstdint>

#include <fcntl.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fstream>
#include <thread>
#include <chrono>

#include "mudaq_device.h"
#include "utils.hpp"


#define PAGEMAP_LENGTH 8 // each page table entry has 64 bits = 8 bytes
#define PAGE_SHIFT 12    // on x86_64: 4 kB pages, so shifts of 12 bits

using namespace std;

// ----------------------------------------------------------------------------
// additional local helper functions

/**
 * Check physical address of allocated memory
 * for 32 bits / 36 bits
 */
int physical_address_check( uint32_t * base_address, size_t size ) {

  /* Open binary file with page map table */
  FILE *pagemap = fopen("/proc/self/pagemap", "rb");
  void * virtual_address;
  unsigned long offset = 0, page_frame_number = 0, distance_from_page_boundary = 0;
  uint64_t offset_mem;

  for (uint i = 0; i < size / _pagesize(); i++ ) { // loop over all pages in allocated memory
    virtual_address = base_address + i * _pagesize() / 4; // uint32_t words
    /* go to entry for virtual_address  */
    offset = (unsigned long)virtual_address / _pagesize() * PAGEMAP_LENGTH;
    if( fseek( pagemap, (unsigned long)offset, SEEK_SET) != 0) {
      fprintf(stderr, "Failed to seek pagemap to proper location\n");
    }

    /* get page frame number and shift by page_shift to get physical address */
    fread(&page_frame_number, 1, PAGEMAP_LENGTH - 1, pagemap);
    page_frame_number &= 0x7FFFFFFFFFFFFF;  // clear bit 55: soft-dirty. clear this to indicate that nothing has been written yet
    distance_from_page_boundary = (unsigned long)virtual_address % _pagesize();
    offset_mem = (page_frame_number << PAGE_SHIFT) + distance_from_page_boundary;

    //cout << hex << "Physical address: " << offset_mem << endl;
    if ( offset_mem >> 32 == 0 ) {
      cout << dec << "Memory resides within 4 GB for page " << i << " at address " << hex << offset_mem << endl;
      fclose(pagemap);
      return -1;
    }
  }
  fclose(pagemap);
  return 0;
}

int is_page_aligned( void * pointer ) {
  DEBUG("diff to page: %lu", ( (uintptr_t)(const void *)(pointer) ) % _pagesize() );
  return !( ( (uintptr_t)(const void *)(pointer) ) % _pagesize()  == 0 );
}

void * align_page( void * pointer ) {
  void * aligned_pointer = (void *)( (uintptr_t)(const void *)pointer + _pagesize() - ((uintptr_t)(const void *)(pointer) ) % _pagesize() );
  return aligned_pointer;
}

void * get_next_aligned_page( void * base_address ) {

  int retval = is_page_aligned( base_address );
    if ( retval != 0 ) {
      //ERROR("Memory buffer is not page aligned");
      void * aligned_pointer = align_page( base_address );
      retval = is_page_aligned( aligned_pointer );
      return aligned_pointer;
    }

  return base_address;
}


static void _print_raw_buffer(volatile uint32_t* addr, unsigned len,
                              unsigned block_offset = 0)
{
    const unsigned BLOCK_SIZE = 8;
    const unsigned first = (block_offset * BLOCK_SIZE);
    const unsigned last = first + len - 1;

    cout << showbase << hex;

    unsigned i;
    for (i = first; i <= last; ++i) {
        if ((i % BLOCK_SIZE) == 0) {
            cout << setw(6) << i << " |";
        }
        cout << " " << setw(10) << addr[i];
        // add a newline after every complete block and after the last one.
        if ((i % BLOCK_SIZE) == (BLOCK_SIZE - 1) || (i == last)) {
            cout << endl;
        }
    }

    cout << noshowbase << dec;
}

namespace mudaq {

// MudaqDevice

  mudaq::MudaqDevice::MudaqDevice(const std::string& path) :
    _fd(-1),
    _path(path),
    _regs_rw(nullptr),
    _regs_ro(nullptr),
    _mem_ro(nullptr),
    _mem_rw(nullptr)
  {
      _last_read_address = 0;
}

  bool MudaqDevice::is_ok() const {

    bool error = (_fd < 0) ||
      (_regs_rw == nullptr) ||
      (_regs_ro == nullptr) ||
      (_mem_ro == nullptr)  ||
      (_mem_rw == nullptr);

    return !error;
  }

  bool MudaqDevice::open()
{
    // O_SYNC only affects 'write'. not really needed but doesnt hurt and makes
    // things safer if we later decide to use 'write'.
    _fd = ::open(_path.c_str(), O_RDWR | O_SYNC);
    if (_fd < 0) {
        ERROR("could not open device '%s': %s", _path, strerror(errno));
        return false;
    }
    _regs_rw = mmap_rw(MUDAQ_REGS_RW_INDEX, MUDAQ_REGS_RW_LEN);
    _regs_ro = mmap_ro(MUDAQ_REGS_RO_INDEX, MUDAQ_REGS_RO_LEN);
    _mem_rw =  mmap_rw(MUDAQ_MEM_RW_INDEX,  MUDAQ_MEM_RW_LEN);
    _mem_ro =  mmap_ro(MUDAQ_MEM_RO_INDEX,  MUDAQ_MEM_RO_LEN);
    return (_regs_rw != nullptr) && (_regs_ro != nullptr) && (_mem_rw != nullptr) && (_mem_ro != nullptr);
}

void MudaqDevice::close()
{
    munmap_wrapper(&_mem_ro,  MUDAQ_MEM_RO_LEN,  "could not unmap read-only memory");
    munmap_wrapper(&_mem_rw,  MUDAQ_MEM_RW_LEN,  "could not unmap read/write memory");  // added by DvB for rw mem
    munmap_wrapper(&_regs_ro, MUDAQ_REGS_RO_LEN, "could not unmap read-only registers");
    munmap_wrapper(&_regs_rw, MUDAQ_REGS_RW_LEN, "could not unmap read/write registers");
    if (_fd >= 0 && ::close(_fd) < 0) {
        ERROR("could not close '%s': %s", _path, strerror(errno));
    }
    // invalidate the file descriptor
    _fd = -1;
}

bool MudaqDevice::operator!() const
{
    return (_fd < 0) || (_regs_rw == nullptr)
                     || (_regs_ro == nullptr)
                     || (_mem_ro == nullptr)
                     || (_mem_rw == nullptr);
}

void MudaqDevice::write_memory_rw(unsigned idx, uint32_t value)
{
    if(idx > 64*1024){
        cout << "Invalid memory address " << idx << endl;
        exit (EXIT_FAILURE);
    }
    else {
        _mem_rw[idx & MUDAQ_MEM_RW_MASK] = value;
    }
}

void MudaqDevice::write_register(unsigned idx, uint32_t value)
{
    if(idx > 63){
        cout << "Invalid register address " << idx << endl;
        exit (EXIT_FAILURE);
    }
    else {
    _regs_rw[idx] = value;
    }
}

void MudaqDevice::write_register_wait(unsigned idx, uint32_t value,
                                      unsigned wait_ns)
{
    write_register(idx, value);
    std::this_thread::sleep_for(std::chrono::nanoseconds(wait_ns));
}

void MudaqDevice::toggle_register(unsigned idx, uint32_t value, unsigned wait_ns)
{
    uint32_t old_value = read_register_rw(idx);
    write_register_wait(idx, value, wait_ns);
    write_register(idx, old_value);
}

uint32_t MudaqDevice::read_register_rw(unsigned idx) const {
       if(idx > 63){
           cout << "Invalid register address " << idx << endl;
           exit (EXIT_FAILURE);
       }
       return _regs_rw[idx];
}


uint32_t MudaqDevice::read_register_ro(unsigned idx) const {
       if(idx > 63){
           cout << "Invalid register address " << idx << endl;
           exit (EXIT_FAILURE);
       }
       return _regs_ro[idx];
}

uint32_t MudaqDevice::read_memory_ro(unsigned idx) const {
       if(idx > 64*1024){
           cout << "Invalid memory address " << idx << endl;
           exit (EXIT_FAILURE);
       }
       return _mem_ro[idx & MUDAQ_MEM_RO_MASK];
}

uint32_t MudaqDevice::read_memory_rw(unsigned idx) const {
       if(idx > 64*1024-1){
           cout << "Invalid memory address " << idx << endl;
           exit (EXIT_FAILURE);
       }
       return _mem_rw[idx & MUDAQ_MEM_RW_MASK];
}

void MudaqDevice::enable_led(unsigned which)
{
    uint8_t pattern;
    // turn on a single led w/o changing the status of the remaining ones
    // since we only have 8 leds we need to wrap the led index
    pattern  = read_register_rw(LED_REGISTER_W);
    pattern |= (1 << (which % 8));
    write_register(LED_REGISTER_W, pattern);
}

void MudaqDevice::enable_leds(uint8_t pattern)
{
    write_register(LED_REGISTER_W, pattern);
}

void MudaqDevice::disable_leds()
{
    write_register(LED_REGISTER_W, 0x0);
}

void MudaqDevice::print_registers()
{
    cout << "offset + read/write registers" << endl;
    _print_raw_buffer(_regs_rw, MUDAQ_REGS_RW_LEN);
    cout << "offset + read-only registers" << endl;
    _print_raw_buffer(_regs_ro, MUDAQ_REGS_RO_LEN);
}

// ----------------------------------------------------------------------------
// mmap / munmap helper functions

volatile uint32_t * MudaqDevice::mmap_rw(unsigned idx, unsigned len)
{
    off_t offset = idx * _pagesize();
    size_t size = len * sizeof(uint32_t);
    // TODO what about | MAP_POPULATE
    volatile void * rv = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED , _fd, offset);
    if (rv == MAP_FAILED) {
        ERROR("could not mmap region %d in read/write mode: %s", idx, strerror(errno));
        return static_cast<volatile uint32_t *>(nullptr);
    } else {
        return static_cast<volatile uint32_t *>(rv);
    }
}

volatile uint32_t * MudaqDevice::mmap_ro(unsigned idx, unsigned len)
{
    off_t offset = idx * _pagesize();
    size_t size = len * sizeof(uint32_t);
    // TODO what about | MAP_POPULATE
    volatile void * rv = mmap(nullptr, size, PROT_READ, MAP_SHARED , _fd, offset);
    if (rv == MAP_FAILED) {
        ERROR("could not mmap region %d in read-only mode: %s", idx, strerror(errno));
        return static_cast<volatile uint32_t *>(nullptr);
    } else {
        return static_cast<volatile uint32_t *>(rv);
    }
}

void MudaqDevice::munmap_wrapper(uint32_t** addr, unsigned len,
                                 const std::string& error_msg)
{
    // i have to cast away volatile to allow to call munmap. using any of the
    // "correct" c++ versions, e.g. const_cast, reinterpret_cast, ... do not
    // seem to work. back to plain c-type cast
    if (munmap((*addr), len * sizeof(uint32_t)) < 0) {
        ERROR("%s: %s", error_msg, strerror(errno));
    }
    // invalidate the pointer
    (*addr) = nullptr;
}

void MudaqDevice::munmap_wrapper(volatile uint32_t** addr, unsigned len,
                                 const std::string& error_msg)
{
    // cast away volatility. not required for munmap
    uint32_t** tmp = (uint32_t**)(addr);
    munmap_wrapper(tmp, len, error_msg);
}

/*
 *  PCIe packet and software interface
 *  20b: N: packet length for following payload(in 32b words)

 *  N*32b: packet payload:
 *      0xBC, 4b type=0xC, 2b SC type = 0b11, 16b FPGA ID
 *      start addr(32b, user parameter)
 *      (N-2)*data(32b, user parameter)
 *
 *      1 word as dummy: 0x00000000
 *      Write length from 0xBC -> 0x9c to SC_MAIN_LENGTH_REGISTER_W
 *      Write enable to SC_MAIN_ENABLE_REGISTER_W
 */
/*int MudaqDevice::FEB_write(uint32_t FPGA_ID, uint32_t* data, uint16_t length, uint32_t startaddr) {

    // TODO: Check length, startaddr and FPGA ID
    // TODO: Instead of data, length, use a vector
    // TODO: Think of using meaningful return codes specifying the error condition


    int count = 0;
    while(count < 1000){
        // NB: TODO: Is this really necessary? Are there multiple threads talking to the same MuDaq Device?
        // If yes, use a mutex
        // If no, we know that this function (and the read) are blocking and will only return if the
        // transaction is done or there was a timeout.

        // Also: Refactor this ready-check into a separate function instead of repeating it
        // Also: I guess that bit 0 is the busy bit - check for the busy bit only

        if ( read_register_ro(SC_MAIN_STATUS_REGISTER_R) == 0x00000001 ) break;
        count++;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    if(count==1000){
        printf("MudaqDevice::FEB_write SC Main is not ready");
        return -1;
    }
    
    uint32_t FEB_PACKET_TYPE_SC = 0x7;
    uint32_t FEB_PACKET_TYPE_SC_WRITE = 0x3; // this is 11 in binary

    // two most significant bits are 0
    write_memory_rw(0, FEB_PACKET_TYPE_SC << 26 | FEB_PACKET_TYPE_SC_WRITE << 24 | (uint16_t) FPGA_ID << 8 | 0xBC); 
    write_memory_rw(1, startaddr);
    write_memory_rw(2, length);

    for (int i = 0; i < length; i++) {
        write_memory_rw(3 + i, data[i]);
    }
    write_memory_rw(3 + length, 0x0000009c);
    
    // SC_MAIN_LENGTH_REGISTER_W starts from 1
    // length for SC Main does not include preamble and trailer, thats why it is 2+length
    // TODO: Why wait?
    write_register_wait(SC_MAIN_LENGTH_REGISTER_W, 2 + length, 100);

    // TODO: There is a toggle function for this...
    write_register_wait(SC_MAIN_ENABLE_REGISTER_W, 0x1, 100);
    // firmware regs SC_MAIN_ENABLE_REGISTER_W so that it only starts on a 0->1 transition
    write_register_wait(SC_MAIN_ENABLE_REGISTER_W, 0x0, 100);
    
    // check again if SC Main is done
    count = 0;
    while(count < 1000){
        if ( read_register_ro(SC_MAIN_STATUS_REGISTER_R) == 0x00000001 ) break;
        count++;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    if(count==1000){
        printf("MudaqDevice::FEB_write Timeout for done reg");
        return -1;
    }
    
    return 0;
}

int MudaqDevice::FEB_read(uint32_t FPGA_ID, uint16_t length, uint32_t startaddr) {

    // TODO: See write

    int count = 0;
    while(count < 1000){
        if ( read_register_ro(SC_MAIN_STATUS_REGISTER_R) == 0x00000001 ) break;
        count++;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    if(count==1000){
        printf("MudaqDevice::FEB_write SC Main is not ready");
        return -1;
    }
    
    uint32_t FEB_PACKET_TYPE_SC = 0x7;
    uint32_t FEB_PACKET_TYPE_SC_READ = 0x2; // this is 10 in binary

    write_memory_rw(0, FEB_PACKET_TYPE_SC << 26 | FEB_PACKET_TYPE_SC_READ << 24 | (uint16_t) FPGA_ID << 8 | 0xBC);
    write_memory_rw(1, startaddr);
    write_memory_rw(2, length);
    write_memory_rw(3, 0x0000009c);
    
    // SC_MAIN_LENGTH_REGISTER_W starts from 1
    // length for SC Main does not include preamble and trailer, thats why it is 2
    write_register_wait(SC_MAIN_LENGTH_REGISTER_W, 2, 100);
    write_register_wait(SC_MAIN_ENABLE_REGISTER_W, 0x1, 100);
    // firmware regs SC_MAIN_ENABLE_REGISTER_W so that it only starts on a 0->1 transition
    write_register_wait(SC_MAIN_ENABLE_REGISTER_W, 0x0, 100);
    
    // check again if SC Main is done
    count = 0;
    while(count < 1000){
        if ( read_register_ro(SC_MAIN_STATUS_REGISTER_R) == 0x00000001 ) break;
        count++;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    if(count==1000){
        printf("MudaqDevice::FEB_write Timeout for done reg");
        return -1;
    }
    
    return 0;
}*/

// ----------------------------------------------------------------------------
// DmaMudaqDevice

  DmaMudaqDevice::DmaMudaqDevice(const string& path) :
    MudaqDevice(path), _dmabuf_ctrl(nullptr), _last_end_of_buffer(0)
  {
    // boring
  }


  bool DmaMudaqDevice::open()
  {
    if (!MudaqDevice::open()) return false;
    _dmabuf_ctrl = mmap_ro(MUDAQ_DMABUF_CTRL_INDEX, MUDAQ_DMABUF_CTRL_WORDS);
    return (_dmabuf_ctrl != nullptr);
  }

  void DmaMudaqDevice::close()
  {
    munmap_wrapper(&_dmabuf_ctrl, MUDAQ_DMABUF_CTRL_WORDS, "could not unmap dma control buffer");
    MudaqDevice::close();
  }

  bool DmaMudaqDevice::operator!() const
  {
    return MudaqDevice::operator!() || (_dmabuf_ctrl == nullptr);
      //|| (_dmabuf_data == nullptr);
  }

  int DmaMudaqDevice::read_block(DataBlock& buffer,  volatile uint32_t * pinned_data)
  {
      uint32_t end_write = _dmabuf_ctrl[3]>>2; // dma address next to be written to (last written to + 1) (in words)
      if(end_write==0) // This is problematic if runs get bigger than the DMA bufer
           return READ_NODATA;
     // cout <<hex<< interrupt << ", "<<end_write<<endl;
      size_t begin = _last_end_of_buffer & MUDAQ_DMABUF_DATA_WORDS_MASK;
      size_t end = (end_write-1) & MUDAQ_DMABUF_DATA_WORDS_MASK;
      size_t len = ((end - begin+1) & MUDAQ_DMABUF_DATA_WORDS_MASK);

      if(len == 0){
          return READ_NODATA;
      }

    _last_end_of_buffer = end + 1;


    buffer = DataBlock( pinned_data, begin, len );
    return READ_SUCCESS;
}

int DmaMudaqDevice::get_current_interrupt_number()
{
  int interrupt_number;
  int ret_val = ioctl( _fd, REQUEST_INTERRUPT_COUNTER, &interrupt_number);
  if ( ret_val == -1 ) {
    printf("Requesting the interrupt number failed with %d \n", errno);
    return ret_val;
  }
  else
    return interrupt_number;
}

int DmaMudaqDevice::map_pinned_dma_mem( struct mesg user_message ) {
  int ret_val = ioctl( _fd, MAP_DMA, &user_message);
  if ( ret_val == -1 ) {
      if ( errno == 12 ) {
      cout << "DMA mapping failed: no memory" << endl;
      cout << "Physical memory is too scattered, FPGA cannot take more than 4096 addresses" << endl;
      cout << "Restart computer to obtain contiguous memory buffers" << endl;
      return ret_val;
    }
    else {
      printf("DMA mapping failed with :%d\n", errno);
      return ret_val;
    }
  }
  cout << "Mapped "<< dec << user_message.size << " bytes of pinned memory as DMA data buffer" << endl;
  return 0;
}

// in words!
uint32_t DmaMudaqDevice::last_written_addr() const
{
    // returns: remoteaddress_var <= remoteaddress_var + (packet_length & "00");
    // shifted by two bits
    return (_dmabuf_ctrl[3]>>2);

}

uint32_t DmaMudaqDevice::last_endofevent_addr() const
{
    // returns: d0 := memwriteaddreoedma_long(31 downto 0);
    return _dmabuf_ctrl[0];
}

// enable interrupts
int DmaMudaqDevice::enable_continous_readout(int interTrue)
{

   _last_end_of_buffer = 0;
  if (interTrue == 1){
    write_register(DMA_REGISTER_W, 0x9);
  }
  else {
    write_register(DMA_REGISTER_W, SET_DMA_BIT_ENABLE(0x0));
  }
  return 0;
}

void DmaMudaqDevice::disable()
{
   write_register(DMA_REGISTER_W, UNSET_DMA_BIT_ENABLE(0x0));
}


// ----------------------------------------------------------------------------
// convenience output functions

ostream& operator<<(ostream& os, const MudaqDevice& dev)
{
    os << "MudaqDevice '" << dev._path << "' "
       << "status: " << (!dev ? "ERROR" : "ok");
    return os;
}





} // namespace mudaq
