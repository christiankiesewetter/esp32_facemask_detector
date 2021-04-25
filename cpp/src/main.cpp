#include "main.hpp"

static const char *TAG = "MAIN";

camera_fb_t *pic = nullptr;
NN *nn = nullptr;
const double SAFETY_MARGIN = 0.1;

constexpr gpio_num_t BLINK_GPIO = gpio_num_t(4);
constexpr uint8_t BLOCKSIZE = 10;

void setup_interpreter(){
  const unsigned int tensor_arena_size = 183 * 1024;
  nn = new NN(&mask_model_tflite, tensor_arena_size);
}

void setup_led(){
  gpio_pad_select_gpio(BLINK_GPIO);
  gpio_set_direction(BLINK_GPIO, GPIO_MODE_OUTPUT);
}

static bool process_input(uint8_t* buff){
  int8_t *output = nn->run(buff, dimensions::LEN_IMG_BUF);
  int8_t result_mask = output[0];
  int8_t result_nomask = output[1];
  printf("Uncovered Face: %f  Covered Face: %f \r\n", ((result_nomask + 128) / 255.), ((result_mask + 128) / 255.));
  return (result_nomask + SAFETY_MARGIN) > result_mask;
}

static bool take_picture_and_evaluate(){
  ESP_LOGI(TAG, "Taking Picture");
  pic = esp_camera_fb_get();

  if (pic != NULL){
    bool result = process_input(pic->buf);
    esp_camera_fb_return(pic);
    return result;
  }
  
  return false;
}

void turn_led_on_if_mask(void *n){
  while(true){
      gpio_set_level(BLINK_GPIO, take_picture_and_evaluate() ? 1 : 0);
  }
}

extern "C" void app_main()
{
  while (init_camera() != ESP_OK){
    ESP_LOGE(TAG, "Camera Init Failed");
  }

  setup_led();
  setup_interpreter();
  xTaskCreate(&turn_led_on_if_mask, (const char*) "Toggle LED", 18*1024, NULL, 1, NULL);
  
  //delete nn;
  //nn = nullptr;
}
