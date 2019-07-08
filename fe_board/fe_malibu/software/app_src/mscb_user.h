/*
 * main.c
 *
 *  Created on: Apr 1, 2016
 *      Author: fwauters
 *
 *      Description: Get the MSCB interpreter for the Wavedream going on the NIOS processor
 */


#define EOT 0x4

//#include "system.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "sys/alt_stdio.h"
#include "mscb.h"
#include <time.h>
#include<unistd.h>
#include <sys/alt_timestamp.h>

//altera forum hwoto soft reset
//#define HAL_PLATFORM_RESET()
//  NIOS2_WRITE_STATUS(0);
//  NIOS2_WRITE_IENABLE(0);
//  ((void (*) (void)) NIOS2_RESET_ADDR) ();


//#include "wd2_dbg.h"
//#include "times.h"
//#include "git-revision.h"
//#include "gpio_slow_control.h"


//a bit weird that

// move this later !!!
#define PARALLEL_MSCB_IN_BASE 0x700f0300
#define PARALLEL_MSCB_OUT_BASE 0x700f0320
#define COUNTER_BASE 0x700f0340


/*------------------------------------------------------------------*/

/* GET_INFO attributes */
#define GET_INFO_GENERAL  0
#define GET_INFO_VARIABLE 1

/* Variable attributes */
#define SIZE_8BIT         1
#define SIZE_16BIT        2
#define SIZE_24BIT        3
#define SIZE_32BIT        4

/* Address modes */
#define ADDR_NONE         0
#define ADDR_NODE         1
#define ADDR_GROUP        2
#define ADDR_ALL          3

/* local variables */
int addressed = 0;
int addr_mode = 0;
int quit = 0;

unsigned char g_n_sub_addr;
int g_cur_sub_addr = 0;

SYS_INFO sys_info;
extern MSCB_INFO_VAR *variables;
unsigned char g_n_variables, g_var_size;

int DBG_INFO = 1;

/*------------------------------------------------------------------*/


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include<unistd.h>

#include "mscb.h"
#include "sys/alt_stdio.h"

/* number of status bytes */
#define N_STATUS_BYTES 2
/* number of setting bytes */
#define N_SETTING_BYTES 1


// remove this later !!!
#define PARALLEL_FPGA_STATUS_BASE 0x81020
#define PARALLEL_FPGA_STATUS2_BASE 0x81090
#define PARALLEL_FPGA_SETTING_BASE 0x81000

volatile alt_u32* sc_data = (alt_u32*)AVM_SC_BASE;

char node_name[] = "FPGA_TEST";

/*---- Define variable parameters returned to CMD_GET_INFO command ----*/

typedef struct {
   unsigned char fpga_status[N_STATUS_BYTES];
   unsigned char fpga_status2[N_STATUS_BYTES];
   unsigned char fpga_setting[N_SETTING_BYTES];
} USER_DATA;

USER_DATA user_data;

MSCB_INFO_VAR vars[] = {
   { 0 }
};

MSCB_INFO_VAR *variables = vars;

void read_fpga_status();
void set_fpga();

void user_vars(unsigned char *n_sub_addr, unsigned char *var_size)
{
  // declare number of sub-addresses and data size to framework
  *n_sub_addr = 1;
  *var_size = (unsigned char)sizeof(USER_DATA);
}



/*---- User init function ------------------------------------------*/

extern SYS_INFO sys_info;

void user_init(unsigned char init)
{
   int i;

   /* initial nonzero EEPROM values */
   if (init) {
      memset(&user_data, 0, sizeof(user_data));
      //user_data.adc_gain = 1;
      //user_data.adc_offset = 0;
   }

   /* set default group address */
   if (sys_info.group_addr == 0xFFFF)
      sys_info.group_addr = 1600;

   sys_info.group_addr = 0xF000;
   sys_info.node_addr  = 0xACA0;
   strcpy(sys_info.node_name, "StratixIVFrontend00");

   user_data.fpga_setting[0] = 0x00;


}


/*---- User read function ------------------------------------------*/

unsigned char user_read(unsigned char index)
{
   if (index);
   return 0;
}

/*---- User get and set functions -----------------------------------------*/

void set_fpga()
{

	int value = user_data.fpga_setting[0];
	//printf("set fpga with %d\n",value);
	IOWR_ALTERA_AVALON_PIO_DATA(PARALLEL_FPGA_SETTING_BASE,value);
	usleep(1000);
	return;
}

//void read_fpga_status()
//{
//	user_data.fpga_status[0] = IORD_ALTERA_AVALON_PIO_DATA(PARALLEL_FPGA_STATUS_BASE);
//	user_data.fpga_status2[0] = IORD_ALTERA_AVALON_PIO_DATA(PARALLEL_FPGA_STATUS2_BASE);
//	return;
//}

/*---- User write function -----------------------------------------*/

void user_write(unsigned char index)
{
	set_fpga();
	return;
}


/*---- User loop function ------------------------------------------*/

//void user_loop(void)
//{
//	read_fpga_status();
	//set_fpga();
//}



void led_blink(unsigned int times)
{
  float period_blink = 0.1;
  //alt_u32 num_ticks = 0;

  unsigned int i = 0;
  while(i < times)
  {
	  IOWR_ALTERA_AVALON_PIO_CLEAR_BITS(PARALLEL_MSCB_OUT_BASE,0x800); // mask pattern for the par port. 0x80 = 1000 0000
	  usleep(100000);
	  IOWR_ALTERA_AVALON_PIO_SET_BITS(PARALLEL_MSCB_OUT_BASE,0x800);
	  usleep(100000);
	  i++;
  }
  IOWR_ALTERA_AVALON_PIO_CLEAR_BITS(PARALLEL_MSCB_OUT_BASE,0x800);

}

void print_byte(unsigned char byte)
{
  int i;
  //printf("byte to print %0x\n",byte);
  for(i = 7; i >=0; i--)
  {
	  //printf("%d",(byte >> i) & 0x1);
  }
  //printf("\n");
}


int get_times(void)
{
	int counter = IORD_ALTERA_AVALON_PIO_DATA(COUNTER_BASE);
	return counter;
}


int time_diff(int t1, int t2)
{
   //int diff_ = (t2 - t1);
   //alt_u32 diff = (alt_u32)(t2-t1);/// (alt_u32)TIME_FREQUENCY;
   return t2-t1;
}

/*------------------------------------------------------------------*/

unsigned char crc8_data[] = {
   0x00, 0x5e, 0xbc, 0xe2, 0x61, 0x3f, 0xdd, 0x83,
   0xc2, 0x9c, 0x7e, 0x20, 0xa3, 0xfd, 0x1f, 0x41,
   0x9d, 0xc3, 0x21, 0x7f, 0xfc, 0xa2, 0x40, 0x1e,
   0x5f, 0x01, 0xe3, 0xbd, 0x3e, 0x60, 0x82, 0xdc,
   0x23, 0x7d, 0x9f, 0xc1, 0x42, 0x1c, 0xfe, 0xa0,
   0xe1, 0xbf, 0x5d, 0x03, 0x80, 0xde, 0x3c, 0x62,
   0xbe, 0xe0, 0x02, 0x5c, 0xdf, 0x81, 0x63, 0x3d,
   0x7c, 0x22, 0xc0, 0x9e, 0x1d, 0x43, 0xa1, 0xff,
   0x46, 0x18, 0xfa, 0xa4, 0x27, 0x79, 0x9b, 0xc5,
   0x84, 0xda, 0x38, 0x66, 0xe5, 0xbb, 0x59, 0x07,
   0xdb, 0x85, 0x67, 0x39, 0xba, 0xe4, 0x06, 0x58,
   0x19, 0x47, 0xa5, 0xfb, 0x78, 0x26, 0xc4, 0x9a,
   0x65, 0x3b, 0xd9, 0x87, 0x04, 0x5a, 0xb8, 0xe6,
   0xa7, 0xf9, 0x1b, 0x45, 0xc6, 0x98, 0x7a, 0x24,
   0xf8, 0xa6, 0x44, 0x1a, 0x99, 0xc7, 0x25, 0x7b,
   0x3a, 0x64, 0x86, 0xd8, 0x5b, 0x05, 0xe7, 0xb9,
   0x8c, 0xd2, 0x30, 0x6e, 0xed, 0xb3, 0x51, 0x0f,
   0x4e, 0x10, 0xf2, 0xac, 0x2f, 0x71, 0x93, 0xcd,
   0x11, 0x4f, 0xad, 0xf3, 0x70, 0x2e, 0xcc, 0x92,
   0xd3, 0x8d, 0x6f, 0x31, 0xb2, 0xec, 0x0e, 0x50,
   0xaf, 0xf1, 0x13, 0x4d, 0xce, 0x90, 0x72, 0x2c,
   0x6d, 0x33, 0xd1, 0x8f, 0x0c, 0x52, 0xb0, 0xee,
   0x32, 0x6c, 0x8e, 0xd0, 0x53, 0x0d, 0xef, 0xb1,
   0xf0, 0xae, 0x4c, 0x12, 0x91, 0xcf, 0x2d, 0x73,
   0xca, 0x94, 0x76, 0x28, 0xab, 0xf5, 0x17, 0x49,
   0x08, 0x56, 0xb4, 0xea, 0x69, 0x37, 0xd5, 0x8b,
   0x57, 0x09, 0xeb, 0xb5, 0x36, 0x68, 0x8a, 0xd4,
   0x95, 0xcb, 0x29, 0x77, 0xf4, 0xaa, 0x48, 0x16,
   0xe9, 0xb7, 0x55, 0x0b, 0x88, 0xd6, 0x34, 0x6a,
   0x2b, 0x75, 0x97, 0xc9, 0x4a, 0x14, 0xf6, 0xa8,
   0x74, 0x2a, 0xc8, 0x96, 0x15, 0x4b, 0xa9, 0xf7,
   0xb6, 0xe8, 0x0a, 0x54, 0xd7, 0x89, 0x6b, 0x35,
};

unsigned char crc8(unsigned char *buffer, int len)
/********************************************************************\

  Routine: crc8

  Purpose: Calculate 8-bit cyclic redundancy checksum for a full
           buffer

  Input:
    unsigned char *data     data buffer
    int len                 data length in bytes


  Function value:
    unsighend char          CRC-8 code

\********************************************************************/
{
   int i;
   unsigned char crc8_code, index;

   crc8_code = 0;
   for (i = 0; i < len; i++) {
      index = buffer[i] ^ crc8_code;
      crc8_code = crc8_data[index];
   }

   return crc8_code;
}



void GetInputString(char* entry, int size, FILE* stream)
{
	int i = 0;
	int ch =0;

	for(i=0;(ch!='\n')&&(i<size);)
	{
		//if( (ch = getc(stream)) != '\r' )
		//{
		//	entry[i]= ch;
		//	i++;
		//}
	}

}

static char TopMenu(void)
{
  static char ch;

  while(1)
  {
	  printf("\n\n");
	  printf("running main, type 'y' to go to user loop, type q to quit\n");
	  printf("\n\n");


	  static char entry[4];
	  GetInputString(entry,sizeof(entry),stdin);
	  //if(sscanf(entry,"%c\n",&ch))
	  {
		  if(ch>='A' && ch <= 'Z')
			  ch+='a'-'A';
		  if(ch==27)
			  ch='q';
	  }

      if(ch=='q') break;
      if(ch=='y') break;

  }
  return(ch);

}

void mscb_init()
{
	printf("Starting MSCB test program\n");
	//start timer
	//if( alt_timestamp_start() < 0) printf("no timer available\n");
	//set default out values
	IOWR_ALTERA_AVALON_PIO_DATA(PARALLEL_MSCB_OUT_BASE,0x0FF);
	IOWR_ALTERA_AVALON_PIO_CLEAR_BITS(PARALLEL_MSCB_OUT_BASE,0x200);
	IOWR_ALTERA_AVALON_PIO_CLEAR_BITS(PARALLEL_MSCB_OUT_BASE,0x400); //set read request bit to 0
	printf("PARALLEL_MSCB_IN_BASE %x\n",PARALLEL_MSCB_IN_BASE);
	led_blink(8);



  const char *p;
  int i, adr, flag;

  // obtain number of sub addresses
  user_vars(&g_n_sub_addr, &g_var_size);


  // count variables
  for (g_n_variables = 0; variables[g_n_variables].width > 0 ; g_n_variables++);
/*
  // obtain settings and variables from flash
  flag = mscb_flash_read();

  if (!flag) {
     // default addresses if flash info is invalid
     sys_info.group_addr = 0;
     sys_info.node_addr  = 0;
     strcpy(sys_info.node_name, "WD001");

     // default variables
     for (i = 0; i < g_n_variables ; i++)
        if (!(variables[i].flags & MSCBF_DATALESS)) {
           // do it for each sub-address
           for (adr = 0 ; adr < g_n_sub_addr ; adr++) {
              memset((char*)variables[i].ud + g_var_size*adr, 0, variables[i].width);
           }
        }
  }

  // get GIT revision from src/git-revision.h and extract hash
  p = GIT_REVISION + strlen(GIT_REVISION)-4;
  sys_info.revision = strtol(p, NULL, 16);
*/
  user_init(!flag);

  if (DBG_INFO)
    printf("MSCB address   : 0x%04X\r\n", sys_info.node_addr);


}

/*------------------------------------------------------------------*/

int mscb_loop(void)
{
   unsigned int reg;
   quit=0;

   // read UART only if we are in the crate -> slot_id != 0xFF
   //reg_bank_read(SYSTEM->wd2_reg_bank_ptr, REG_BANK_CTRL_REG, 4, &reg, 1);
   //if ((reg & 0xFF) != 0xFF)  mscb_uart_handler();

   mscb_uart_handler();
   static char ch = 'y';
   return ch;
}

int input_data_ready(void)
{
   //2560 = 1010 0000 0000
  //printf("input parrallel port reading polling 0x%x\n",IORD_ALTERA_AVALON_PIO_DATA(PARALLEL_MSCB_IN_BASE));
  int input = IORD_ALTERA_AVALON_PIO_DATA(PARALLEL_MSCB_IN_BASE);
  int empty_bit = 0x200;
  int empty = input&empty_bit;
  //usleep(10);
  //printf("empty bit = %0d\n",empty);
  if( empty != 0)
  {return 0;} //check empty flag of the fifo
  else
  {
	  //printf("found data ready\n");
	  return 1;
  }
}

unsigned char read_mscb_command(void)
{
	//printf("input parrallel port reading before request 0x%x\n",IORD_ALTERA_AVALON_PIO_DATA(PARALLEL_MSCB_IN_BASE));
	//generate read request for the fifo
	IOWR_ALTERA_AVALON_PIO_CLEAR_BITS(PARALLEL_MSCB_OUT_BASE,0x400);\
	usleep(1);
	IOWR_ALTERA_AVALON_PIO_SET_BITS(PARALLEL_MSCB_OUT_BASE,0x400);
	usleep(1);
	//assume the on signal lasts at least a clock cycle
	IOWR_ALTERA_AVALON_PIO_CLEAR_BITS(PARALLEL_MSCB_OUT_BASE,0x400);

	//printf("input parrallel port reading 0x%x\n",IORD_ALTERA_AVALON_PIO_DATA(PARALLEL_MSCB_IN_BASE));

	int byte = IORD_ALTERA_AVALON_PIO_DATA(PARALLEL_MSCB_IN_BASE) & 0x0FF;
	unsigned char mscb_byte = (char)byte;
	return mscb_byte;
}

void send_data(unsigned char *buf, unsigned int n)
{
	//printf("Data (%d bytes) ready to be send \n",n);
	unsigned int i;
	for(i=0; i<n;i++)
	{
		print_byte(buf[i]);

		int data = buf[i];
		IOWR_ALTERA_AVALON_PIO_DATA(PARALLEL_MSCB_OUT_BASE,data);

		//write request to FIFO
		IOWR_ALTERA_AVALON_PIO_CLEAR_BITS(PARALLEL_MSCB_OUT_BASE,0x200);
		usleep(1);
		IOWR_ALTERA_AVALON_PIO_SET_BITS(PARALLEL_MSCB_OUT_BASE,0x200);
		usleep(1);
		IOWR_ALTERA_AVALON_PIO_CLEAR_BITS(PARALLEL_MSCB_OUT_BASE,0x200);

	}



}

/*------------------------------------------------------------------*/

unsigned char in_buf[256], out_buf[256]; //A char is one byte long, or 8 bits FW
unsigned int i_in = 0;
alt_u32  last_received = 0;

void mscb_uart_handler(void)
{
   unsigned char data;
   unsigned int cmd_len, n;

   // drop partial buffer if no char received for 100 ms
   int now = get_times();
   //printf("i_in %d , time passed %f\n",i_in,time_diff(last_received,now));
   if (i_in > 0 && time_diff(last_received,now) > 12500) {
      i_in = 0;//printf("resseting buffer\n");
   }

   while (input_data_ready())
   {
	  //printf("int read byte loop, input_data_ready = %d\n",input_data_ready());
      data = read_mscb_command();
      //printf("mscb byte = 0x%0x, i buffer = %d, tdiff = %d , last = %d , now = %d \n",data,i_in,time_diff(last_received,now),last_received,now);

      //print_byte(data);
      last_received = get_times();

      //a stream of 0's which we don`t interpret? FW
      // check for padding characte
      //is the this the start bit?
      if (data == 0 && i_in == 0)
         return;

      //read a byte FW
      in_buf[i_in++] = data;

      if (i_in == sizeof(in_buf)) {
         i_in = 0;                      // don't interpret command on buffer overflow
         return;
      }

      // initialize command length from first byte. 0x07 = 00000111. In the MSCb protocol, the last 3 bits of the first byte indicate the command length.
      cmd_len = (in_buf[0] & 0x07) + 2; // + cmd + crc

      // if the first bit is xxxx x111, the command is pariable in length
      if (i_in > 1 && cmd_len == 9) {
         // variable length command
         if (in_buf[1] & 0x80)
            cmd_len = (((in_buf[1] & 0x7F) << 8) | in_buf[2]) + 4;
         else
            cmd_len = in_buf[1] + 3;    // + cmd + N + crc
      }

      // ignore ping acknowledge from other node
      if (i_in == 1 && in_buf[0] == 0x78) {
         i_in = 0;
         //printf("Ingnoring ping ackowledge signal\n");
         return;
      }

      //hmmm, so what is hapemning here, return out of the function, so the data in reinterpreted?
      // I believe this is because the data gets transmitted byte by byte
      if (i_in < cmd_len)
      {
         //printf("Buffer not full yet, %d\n",cmd_len);
         return;                        // return if command not yet complete
      }

      usleep(2);                       // wait inter-char delay

      //led_blink(2);
      n = mscb_interprete(0, in_buf, out_buf);
	
      for(int i_printf=0; i_printf < i_in;i_printf++){
	printf("0x%0x\t",in_buf[i_printf]);	
      }
	printf("\n");

      if (n > 0){
	 printf("sending reply\n");
	 send_data(out_buf, n);
      }
      i_in = 0;
   }

   // drop partial buffer if no char received for 100 ms
   //alt_u32 now = get_times();
   //printf ("time passed %f\n",time_diff(last_received,now));
   //if (i_in > 0 && time_diff(last_received,now) > 0.1) {
   //   i_in = 0;printf("resseting buffer\n");
   //}

   
}

/*------------------------------------------------------------------*/

void addr_node16(unsigned char mode, unsigned int adr, unsigned int node_addr)
{
   //printf("entering addressing, current node address = %x, MSCB command address = %x, addressed status = %d, mode = %d\n",node_addr, adr, addressed, mode);
   if (node_addr == 0xFFFF) {
      if (adr == node_addr) {
         addressed = 1;
         g_cur_sub_addr = 0;
         addr_mode = mode;
      } else {
         addressed = 0;
         addr_mode = ADDR_NONE;
      }
   } else {
      if (mode == ADDR_NODE) {
         if (adr >= node_addr &&
             adr <  node_addr + g_n_sub_addr) {

            addressed = 1; //led_blink(5); //printf("addressed \n");
            g_cur_sub_addr = adr - node_addr;
            addr_mode = ADDR_NODE;
         } else {
            addressed = 0;
            addr_mode = ADDR_NONE;
         }
      } else if (mode == ADDR_GROUP) {
         if (adr == node_addr) {
            addressed = 1;
            g_cur_sub_addr = 0;
            addr_mode = ADDR_GROUP;
         } else {
            addressed = 0;
            addr_mode = ADDR_NONE;
         }
      }
   }
   //printf("addressed set to %d\n",addressed);
}

int mscb_main()
{
    printf("starting mscb node ");
    int ch;
    mscb_init();
    //ch = TopMenu();
    ch='y';
    led_blink(2);
        while(1)
        {
      //if(ch=='y')
      //{
          ch = mscb_loop();
          //user_loop();
      //}
          if(ch=='q' or quit == 1)
          {
            printf("\n exiting \n");
                printf("%c",EOT);
                led_blink(4);
                break;
          }
        }
    return 0;
}



unsigned int mscb_interprete(int submaster, unsigned char *buf, unsigned char *rb)
{
  //printf("interpret mscb command ... \n");
  unsigned char ch, a1, a2;
  unsigned int size, u, adr, i, j, buflen, n = 0;

  // determine length of MSCB command
  buflen = (buf[0] & 0x07);
  if (buflen == 7)
  {
    if (buf[1] & 0x80)  buflen = (((buf[1] & 0x7F) << 8) | buf[2]) + 4;
	else   buflen = (buf[1] & 0x7F) + 3;  // add command, length and CRC
  }
  else  buflen += 2; // add command and CRC

  if(DBG_INFO) {
  //printf("MSCB interprete: %d bytes\r\n", buflen);
//    for (j=0 ; j<buflen ; j++) {
      //printf("%02X ", buf[j]);
  //    if ((j+1) % 16 == 0)
        //printf("\r\n");
    //}
    //printf("\r\n");
  }

  // check CRC
  if (crc8(buf, buflen-1) != buf[buflen-1]) {
    //printf("MSCB interprete: Invalid CRC %02X vs %02X\r\n", crc8(buf, buflen-1), buf[buflen-1]);
    return 0;
  }
  //else printf("CRC ok\n");

  if (!addressed &&
		  buf[0] != MCMD_ADDR_NODE16 &&
          buf[0] != MCMD_ADDR_GRP16 &&
          buf[0] != MCMD_ADDR_BC &&
          buf[0] != MCMD_PING16)
       return 0;

  //printf("Ready to switch, %02X\n",buf[0]);


  switch (buf[0]) {


    case MCMD_ADDR_NODE16:
      //printf("MCMD_ADDR_NODE16\n");
      addr_node16(ADDR_NODE, (buf[1] << 8) | buf[2], sys_info.node_addr);
      break;


    case MCMD_ADDR_GRP16:
      //printf("MCMD_ADDR_GRP16\n");
      addr_node16(ADDR_GROUP, (buf[1] << 8) | buf[2], sys_info.group_addr);
      break;

    case MCMD_PING16:
      addr_node16(ADDR_NODE, (buf[1] << 8) | buf[2], sys_info.node_addr);
      //printf("processing ping command\n");
      if (addressed) {
         //led_blink(1);
         rb[0] = MCMD_ACK;
         n = 1;
      //   printf("ping recieved\n");
      } else {
         if (submaster) {
            rb[0] = 0xFF;
            n = 1;
         }
      }
      break;

    case MCMD_INIT:
      //printf("\r\nRebooting...\r\n");
      quit=1;
      mscb_main();
      break;

    case MCMD_GET_INFO:
    	//  printf("info request\n");
           /* general info */

          n = 0;
          rb[n++] = MCMD_ACK + 7;               // send acknowledge, variable data length
          rb[n++] = 32;                         // data length
          rb[n++] = PROTOCOL_VERSION;           // send protocol version
          rb[n++] = 3;                          // n_variables   send number of variables
          rb[n++] = sys_info.node_addr >> 8;    // send node address
          rb[n++] = sys_info.node_addr & 0xFF;
          rb[n++] = sys_info.group_addr >> 8;   // send group address
          rb[n++] = sys_info.group_addr & 0xFF;
          rb[n++] = sys_info.revision >> 8;     // send revision
          rb[n++] = sys_info.revision & 0xFF;

          for (i = 0; i < 16; i++)  // send node name
            rb[n++] = sys_info.node_name[i];

          for (i = 0; i < 6 ; i++)  // no RTC
            rb[n++] = 0;

          rb[n++] = 1024 >> 8;      // max. buffer size 1024 bytes
          rb[n++] = 1024 & 0xFF;

          rb[n] = crc8(rb, n);
          n++;
          break;

    case MCMD_GET_INFO + 1:
           printf("variable info request\n");
           /* send variable info */
           if (buf[1] < g_n_variables) {
             MSCB_INFO_VAR *pvar;
             pvar = variables + buf[1];

             n = 0;
             rb[n++] = MCMD_ACK + 7;            // send acknowledge, variable data length
             rb[n++] = 13;                      // data length
             rb[n++] = pvar->width; printf("width : %02X\n",pvar->width);
             rb[n++] = pvar->unit;
             rb[n++] = pvar->prefix;
             rb[n++] = pvar->status;
             rb[n++] = pvar->flags;

             for (i = 0; i < 8; i++)            // send variable name
               rb[n++] = pvar->name[i];

             rb[n] = crc8(rb, n);
             n++;
           } else {
             /* just send dummy ack */
             rb[0] = MCMD_ACK;
             rb[1] = 0;
             n = 2;
           }
           break;



    case MCMD_SET_ADDR:
    	   printf("MCMD_SET_ADDR\n");
           if (buf[1] == ADDR_SET_NODE)
              /* complete node address */
              sys_info.node_addr = (buf[2] << 8) | buf[3];
           else if (buf[1] == ADDR_SET_HIGH)
              /* only high byte node address */
              *((unsigned char *)(&sys_info.node_addr)) = (buf[2] << 8) | buf[3];
           else if (buf[1] == ADDR_SET_GROUP)
              /* group address */
              sys_info.group_addr = (buf[2] << 8) | buf[3];

           /* copy address to flash */
           //mscb_flash_write();
           break;

    case MCMD_SET_NAME:
    	   printf("SET NAME\n");
           /* set node name in RAM */
           for (i = 0; i < 16 && i < buf[1]; i++)
              sys_info.node_name[i] = buf[2 + i];
           sys_info.node_name[15] = 0;

           /* copy address to flash */
           //mscb_flash_write(); /////////// Temp comment out, FW
           break;

    case MCMD_FLASH:
        	printf("FLASH WRITE\n");
           //mscb_flash_write(); /////////// Temp comment out, FW
           break;

    case MCMD_ECHO:
    	printf("ECHO\n");
           //led_blink(1);
           rb[0] = MCMD_ACK + 1;
           rb[1] = buf[1];
           rb[2] = crc8(rb, 2);
           n = 3;
           break;

    case MCMD_WRITE_MEM:
       //led_blink(1);
       size = buflen - 9; // minus cmd, len, subadr, adr and CRC
       // subadr = buf[3]; ignored
       adr = buf[4];
       adr = (adr << 8) | buf[5];
       adr = (adr << 8) | buf[6];
       adr = (adr << 8) | buf[7];
	printf("WRITE MEM:\t addr:%08X size %d\n", adr, size);
        sc_data[adr] = buf[8];
	//for(j=0;j<size;j++) // TODO: block writing
        //	sc_data[adr] = (sc_data[adr] << 8) | buf[8+j];
	
	// avoid blocks for the moment:
	if(size>=2){
                sc_data[adr] = (sc_data[adr] << 8) | buf[9];
		if(size==4){
                        sc_data[adr] = (sc_data[adr] << 8) | buf[10];
                        sc_data[adr] = (sc_data[adr] << 8) | buf[11];
		}
	}

       /* only flash supported right now */
       //if ((adr & MSCB_BASE_FLASH) != MSCB_BASE_FLASH)  /////////// Temp comment out, FW
       //   break; /////////// Temp comment out, FW
       //adr = adr & ~MSCB_BASE_FLASH; /////////// Temp comment out, FW

       /* if address is on start of block, erase it */
       //if ((adr & 0xFFFF) == 0)
      //   spi_flash_block64_erase(SYSTEM->spi_flash_ptr, adr); /////////// Temp comment out, FW

       /* write flash */
       //spi_flash_write(SYSTEM->spi_flash_ptr, (unsigned char*)buf+8, adr, size); /////////// Temp comment out, FW

       /*
       printf("\r\nA:%08X size %d\n", adr, size);
       for (i=0 ; i<32 ; i++) {
          for (j=0 ; j<32 ; j++)
             printf("%02X ", buf[7+i*32+j]);
          printf("\r\n");
       }
       */

       /* read back flash and send CRC */
       //spi_flash_read(SYSTEM->spi_flash_ptr, (unsigned char*)buf, adr, size); /////////// Temp comment out, FW

       /*
       printf("\r\nA:%08X size %d\r\n", adr, size);
       for (i=0 ; i<32 ; i++) {
          for (j=0 ; j<32 ; j++)
             printf("%02X ", buf[i*32+j]);
          printf("\r\n");
       }
       printf("CRC = %02X\r\n", crc8(buf, size));
       */
	//    for(int i = 0; i < 32; i++) {
	//	sc_data[i] = i*2;
	//   }
	
	for(int j=0; j<size;j++){ // crc for mem write expects only crc of written data
		buf[j]=buf[j+8];	
	}

      	rb[0] = MCMD_ACK;
       	rb[1] = crc8(buf, size);
	//memset(buf, 0, size);
	n = 2;

       buf[0] = 0; // do not re-interprete command below

       break;

    case MCMD_READ_MEM:
           size = (buf[2] << 8) | buf[3];
           // subadr = buf[3]; ignored
           adr = buf[5];
           adr = (adr << 8) | buf[6];
           adr = (adr << 8) | buf[7];
           adr = (adr << 8) | buf[8];
	   //printf("READ MEM:\t addr:%08X size %d\n", adr, size);

	// mscb only works with bytes 
	// sc_data contains 32 bit per addr --> send 4 bytes per sc addr read:
	   //buf[0]=(sc_data[adr]>>24);
	   //buf[1]=(sc_data[adr]>>16);
	   //buf[2]=(sc_data[adr]>>8);
	   //buf[3]=sc_data[adr];
	
	// the same with a block of memory: 
	   for(int j = 0; j < (size/4+(size%4!=0)) ; j++){
		for(int i=0; i<4; i++){
                        buf[j*4+i]= (sc_data[adr+j]>>(24-8*i));
		}
	   }

           rb[0] = MCMD_ACK + 7;
           rb[1] = 0x80 | ((size >> 8) & 0x7F);
           rb[2] = size & 0xFF;
           memcpy(rb+3, buf, size);
           rb[3+size] = crc8(rb, size+3);
           n = size+4;

           buf[0] = 0; // do not re-interprete command below
           break;

	// old stuff:
           ///* only flash supported right now */
           //if ((adr & MSCB_BASE_FLASH) != MSCB_BASE_FLASH)
           //   break;
           //adr = adr & ~MSCB_BASE_FLASH;

           //memset(buf, 0, size);
	   //for(int j = 0; j< size; j++ ){
	   //	buf[j]=sc_data[adr+j];
		//printf("read addr %08X value: %08X\n",adr+j,sc_data[adr+j]);
	   //}
           //spi_flash_read(SYSTEM->spi_flash_ptr, (unsigned char*)buf, adr, size);  /////////// Temp comment out, FW
	   
  }

  if ((buf[0] & 0xF8) == MCMD_READ) {
	  printf("read request\n");
      if (buf[0] == MCMD_READ + 1) {       // single variable
        if (buf[1] < g_n_variables) {
          n = variables[buf[1]].width;     // number of bytes to return

          if (variables[buf[1]].flags & MSCBF_DATALESS) {
            n = user_read(buf[1]);         // for data less variables, user routine returns bytes
            rb[0] = MCMD_ACK + 7;          // and places data directly in out_buf
            rb[1] = n;
            n += 2;
            rb[n] = crc8(rb, n);           // generate CRC code
            n += 1;
          } else {

            user_read(buf[1]);

            if (n > 6) {
              /* variable length buffer */
              rb[0] = MCMD_ACK + 7;        // send acknowledge, variable data length
              rb[1] = n;                   // send data length

              for (i = 0; i < n; i++)      // copy user data
                rb[2+i] = ((char *) variables[buf[1]].ud)[i+g_var_size*g_cur_sub_addr];
              n += 2;
            } else {

              rb[0] = MCMD_ACK + n;

              for (i = 0; i < n; i++)      // copy user data
                rb[1+i] = ((char *) variables[buf[1]].ud)[i+g_var_size*g_cur_sub_addr];
              n += 1;
            }

            rb[n] = crc8(rb, n);           // generate CRC code
            n++;
          }
        } else {
          /* just send dummy ack to indicate error */
          rb[0] = MCMD_ACK;
          n = 1;
        }

      } else if (buf[0] == MCMD_READ + 2) {   // variable range

        if (buf[1] < g_n_variables && buf[2] < g_n_variables && buf[1] <= buf[2]) {
          /* calculate number of bytes to return */
          for (i = buf[1], size = 0; i <= buf[2]; i++) {
            user_read(i);
            size += variables[i].width;
          }

          n = 0;
          rb[n++] = MCMD_ACK + 7;             // send acknowledge, variable data length
          if (size < 0x80)
            rb[n++] = size;                   // send data length one byte
          else {
            rb[n++] = 0x80 | size / 0x100;    // send data length two bytes
            rb[n++] = size & 0xFF;
          }

          /* loop over all variables */
          for (i = buf[1]; i <= buf[2]; i++) {
            for (j = 0; j < variables[i].width; j++)    // send user data
              rb[n++] = ((char *) variables[i].ud)[j+g_var_size * g_cur_sub_addr];
          }

          rb[n] = crc8(rb, n);
          n++;
        } else {
          /* just send dummy ack to indicate error */
          rb[0] = MCMD_ACK;
          n = 1;
        }
      }
    }

  if ((buf[0] & 0xF8) == MCMD_WRITE_NA || (buf[0] & 0xF8) == MCMD_WRITE_ACK) {

    //led_blink(1);

    n = buf[0] & 0x07;

    if (n == 0x07) {  // variable length
      j = 1;
      n = buf[1];
      ch = buf[2];
    } else {
      j = 0;
      ch = buf[1];
    }

    n--; // data size (minus channel)

    if (ch < g_n_variables) {

      /* don't exceed variable width */
      if (n > variables[ch].width)
        n = variables[ch].width;

      if (addr_mode == ADDR_NODE)
        a1 = a2 = g_cur_sub_addr;
      else {
        a1 = 0;
        a2 = g_n_sub_addr-1;
      }

      for (g_cur_sub_addr = a1 ; g_cur_sub_addr <= a2 ; g_cur_sub_addr++) {
        for (i = 0; i < n; i++)
          if (!(variables[ch].flags & MSCBF_DATALESS)) {
            if (variables[ch].unit == UNIT_STRING) {
              if (n > 4)
                /* copy bytes in normal order */
                ((char *) variables[ch].ud)[i + g_var_size*g_cur_sub_addr] =
                    buf[2 + j + i];
              else
                /* copy bytes in reverse order (got swapped on host) */
                ((char *) variables[ch].ud)[i + g_var_size*g_cur_sub_addr] =
                    buf[buflen - 2 - i];
            } else
              /* copy LSB bytes, needed for BYTE if DWORD is sent */
              ((char *) variables[ch].ud)[i + g_var_size*g_cur_sub_addr] =
                  buf[buflen - 1 - variables[ch].width + i + j];
          }

        user_write(ch);
      }
      g_cur_sub_addr = a1; // restore previous value

      if ((buf[0] & 0xF8) == MCMD_WRITE_ACK) {
        rb[0] = MCMD_ACK;
        rb[1] = buf[buflen - 1];
        n = 2;
      }
    }
  }

  //if (DBG_INFO && n>0) {
    //printf("MSCB return: %d bytes\r\n", n);
    //for (j=0 ; j<n ; j++) {
      //printf("%02X ", rb[j]);
      //if ((j+1) % 16 == 0)
       // printf("\r\n");
   // }
   // printf("\r\n");
  //}

  return n;

}
