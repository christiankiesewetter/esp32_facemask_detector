#include <cmath>
#include <cstdio>

namespace imgutils{
  void resize_image(
        const uint8_t* in_imgbuff,
        uint8_t* out_imgbuff,
        const unsigned int width,
        const unsigned int height,
        const unsigned int block_size);
  
  void stats(
        const uint8_t* in_imgbuff, 
        unsigned int length);
}
