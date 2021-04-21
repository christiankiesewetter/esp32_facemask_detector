#pragma once

#include <cinttypes>
#include <stdio.h>
#include <math.h>

#include <esp_event.h>
#include <esp_log.h>
#include <esp_system.h>
#include <nvs_flash.h>

//#include <img_converters.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_camera.h"
#include "cam_config.hpp"

#include "esp_spi_flash.h"

#include "esp32/spiram.h"
#include "esp32/himem.h"

#include "neuralnet.hpp"

#include "mask_model.hpp"
#include "image_utils.hpp"


