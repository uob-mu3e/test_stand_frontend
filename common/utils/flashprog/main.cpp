#include <iostream>
#include <cstring>
#include <mscb.h>
#include <unistd.h>

#define FEBSPI_ADDR_GITHASH     0x0
#define FEBSPI_ADDR_STATUS      0x2
#define FEBSPI_ADDR_CONTROL     0x3
#define FEBSPI_ADDR_RESET       0x4
#define FEBSPI_ADDR_PROGRAMMING_STATUS     0x10
#define FEBSPI_ADDR_PROGRAMMING_COUNT      0x11
#define FEBSPI_ADDR_PROGRAMMING_CTRL       0x12
#define FEBSPI_ADDR_PROGRAMMING_ADDR       0x13
#define FEBSPI_ADDR_PROGRAMMING_WFIFO      0x14

uint32_t endian(uint32_t in){
        return ((in & 0xFF) << 24) | ((in & 0xFF00) << 8) | ((in & 0xFF0000) >> 8) | ((in & 0xFF000000) >> 24);  
}

int main(int argc, char ** argv) {
    
    if(argc < 5){
        std::cout << "Usage: flashprog <mscb_node> <device> <slot> <programmig_file.rbf>" << std::endl;
        return -1;
    }
    
    
   char node[256];
   strcpy(node, argv[1]);
   int fd = mscb_init(node, sizeof(node), nullptr, 0);
   if (fd < 0) {
      std::cout << "Cannot connect to " << node << std::endl;
      return 0;
   }
   
   int device = atoi(argv[2]);
   int slot = atoi(argv[3]);
   char filename[1024];
   strcpy(filename, argv[4]);

   // Try to read SPI
   int ret;
   unsigned int rbuffer;
   ret = mscb_read_mem(fd, device, slot, FEBSPI_ADDR_GITHASH, &rbuffer, 4);
   std::cout << "MAX10 git hash:" << std::hex << rbuffer << " ret: " << ret << std::endl; 
      
   
   FILE * f = fopen(filename, "rb");
   if(!f){
       std::cout << "Failed to open " << filename << std::endl;
       return -1;
   }
   
    // Get the file size
    fseek (f , 0 , SEEK_END);
    long fsize = ftell(f);
    rewind (f);
   
   std::cout << "Programming size " << fsize << std::endl;
   
   uint32_t status = 0;
   mscb_read_mem(fd, device, slot, FEBSPI_ADDR_PROGRAMMING_STATUS, &status, sizeof(status));
   status = endian(status);
   std::cout << "Status0: " << std::hex << status << std::endl;
   // Reset FIFO
   uint32_t cmdswapped = endian(0x2);
   mscb_write_mem(fd, device, slot, FEBSPI_ADDR_PROGRAMMING_CTRL, &cmdswapped, sizeof(cmdswapped));
   mscb_read_mem(fd, device, slot, FEBSPI_ADDR_PROGRAMMING_STATUS, &status, sizeof(status));
   status = endian(status);
   std::cout << "Status1: " << std::hex << status << std::endl;
   cmdswapped = endian(0x0);
   mscb_write_mem(fd, device, slot, FEBSPI_ADDR_PROGRAMMING_CTRL, &cmdswapped, sizeof(cmdswapped));
   mscb_read_mem(fd, device, slot, FEBSPI_ADDR_PROGRAMMING_STATUS, &status, sizeof(status));
   status = endian(status);
   std::cout << "Status2: " << std::hex << status << std::endl;
   
   uint32_t addr=0;
   while(addr < fsize){
        
                // Write address
        uint32_t addrswapped = endian(addr);
        mscb_write_mem(fd, device, slot, FEBSPI_ADDR_PROGRAMMING_ADDR, &addrswapped, 4);
        
        char buffer[256];
        fread(buffer,sizeof(char),256, f);
        // Write 256 byte to fifo
        mscb_write_mem(fd, device, slot, FEBSPI_ADDR_PROGRAMMING_WFIFO, buffer, 256);

        //mscb_read_mem(fd, 1, 0, 0x10, &status, sizeof(status));
        //status = endian(status);
        //std::cout << "Status3: " << std::hex << status << std::endl;
        
        // write start command
        cmdswapped = endian(0x1);
        mscb_write_mem(fd, device, slot, FEBSPI_ADDR_PROGRAMMING_CTRL, &cmdswapped, sizeof(cmdswapped));
       
        
        
        do{
            mscb_read_mem(fd, device, slot, FEBSPI_ADDR_PROGRAMMING_STATUS, &status, sizeof(status));
            status = endian(status);
          //  std::cout << std::hex << status << " ";
        }while(status & 0x1);
        //std::cout << std::endl;
        
         // and set 0 again
        cmdswapped = endian(0x0);
        mscb_write_mem(fd, device, slot, FEBSPI_ADDR_PROGRAMMING_CTRL, &cmdswapped, sizeof(cmdswapped));
            
        addr += 256;
        if(addr%(4096)==0)
            std::cout << (double)addr/fsize << " loaded" << std::endl;
    }
   
    std:: cout << "Done!" << std::endl;
      fclose(f);
   return 0;
}
