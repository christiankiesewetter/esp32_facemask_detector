#pragma once
#include "esp_camera.h"

// #define F_SIZE FRAMESIZE_QQVGA // 160x120
#define F_SIZE FRAMESIZE_96X96 

constexpr int PWDN_GPIO_NUM =     32;
constexpr int RESET_GPIO_NUM =    -1;
constexpr int XCLK_GPIO_NUM =      0;
constexpr int SIOD_GPIO_NUM =     26;
constexpr int SIOC_GPIO_NUM =     27;
constexpr int Y9_GPIO_NUM =       35;
constexpr int Y8_GPIO_NUM =       34;
constexpr int Y7_GPIO_NUM =       39;
constexpr int Y6_GPIO_NUM =       36;
constexpr int Y5_GPIO_NUM =       21;
constexpr int Y4_GPIO_NUM =       19;
constexpr int Y3_GPIO_NUM =       18;
constexpr int Y2_GPIO_NUM =        5;
constexpr int VSYNC_GPIO_NUM =    25;
constexpr int HREF_GPIO_NUM =     23;
constexpr int PCLK_GPIO_NUM =     22;

namespace dimensions{
  constexpr unsigned int WIDTH = 96;
  constexpr unsigned int HEIGHT = 96;
  constexpr unsigned int CHANNELS = 1;
  constexpr unsigned int LEN_IMG_BUF = WIDTH * HEIGHT;

  constexpr unsigned int BLOCK_SIZE = 1;
  constexpr unsigned int TARGET_IMG_BUFF_SIZE = LEN_IMG_BUF / (BLOCK_SIZE*BLOCK_SIZE);

  constexpr int NEW_WIDTH = WIDTH / BLOCK_SIZE;
  constexpr int NEW_HEIGHT = HEIGHT / BLOCK_SIZE;
}

extern const camera_config_t camera_config;
esp_err_t init_camera();