#include "cam_config.hpp"

const camera_config_t camera_config = {
  .pin_pwdn = PWDN_GPIO_NUM,
  .pin_reset = RESET_GPIO_NUM,
  .pin_xclk = XCLK_GPIO_NUM,
  .pin_sscb_sda = SIOD_GPIO_NUM,
  .pin_sscb_scl = SIOC_GPIO_NUM,

  .pin_d7 = Y9_GPIO_NUM,
  .pin_d6 = Y8_GPIO_NUM,
  .pin_d5 = Y7_GPIO_NUM,
  .pin_d4 = Y6_GPIO_NUM,
  .pin_d3 = Y5_GPIO_NUM,
  .pin_d2 = Y4_GPIO_NUM,
  .pin_d1 = Y3_GPIO_NUM,
  .pin_d0 = Y2_GPIO_NUM,

  .pin_vsync = VSYNC_GPIO_NUM,
  .pin_href = HREF_GPIO_NUM,
  .pin_pclk = PCLK_GPIO_NUM,
  
  .xclk_freq_hz = 20000000,
  
  .ledc_timer = LEDC_TIMER_0,
  .ledc_channel = LEDC_CHANNEL_0,

  .pixel_format = PIXFORMAT_GRAYSCALE, //YUV422,GRAYSCALE,RGB565,JPEG
  .frame_size = F_SIZE,    //QQVGA-UXGA Do not use sizes above QVGA when not JPEG
  
  .jpeg_quality = 12, //0-63 lower number means higher quality
  .fb_count = 1      //if more than one, i2s runs in continuous mode. Use only with JPEG
};


esp_err_t init_camera()
{
    esp_err_t err = esp_camera_init(&camera_config);
    sensor_t *sensor = esp_camera_sensor_get();
    sensor->set_framesize(sensor, F_SIZE);
    sensor->set_contrast(sensor, 1);

    if (err != ESP_OK){
        printf("Camera Init Failed");
        return err;
    }
    return ESP_OK;
}