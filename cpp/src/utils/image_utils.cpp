#include "image_utils.hpp"

namespace imgutils{

  void resize_image(
        const uint8_t* in_imgbuff,
        uint8_t* out_imgbuff,
        const unsigned int width,
        const unsigned int height,
        const unsigned int block_size){

    unsigned int ipos = 0;
    for (unsigned int hstep = 0; hstep < height/block_size; hstep++){
      for (unsigned int wstep = 0; wstep < width/block_size; wstep++){
        uint32_t currentpoint = 0;
        for(unsigned int jj = 0; jj < block_size; jj++){
          for(unsigned int kk = 0; kk < block_size; kk++){
            currentpoint += (in_imgbuff[(wstep * block_size)
                            + (hstep * width * block_size)
                            + (kk + (jj * width))]);
          }
        }
        out_imgbuff[ipos] = static_cast<uint8_t>(currentpoint / (block_size * block_size));
        ipos++;
      }
    }
  }

  void stats(int8_t* in_imgbuff, unsigned int length){
    int8_t min = 127;
    int8_t max = -128;
    for (unsigned int ii = 0; ii < length; ii++){
      if (in_imgbuff[ii] > max)  max = in_imgbuff[ii];
      if (in_imgbuff[ii] < min)  min = in_imgbuff[ii];
    }
    printf("min: %d, max: %d", min, max);
  }
}