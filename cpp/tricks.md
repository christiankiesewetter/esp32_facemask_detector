# Issues

# Issue 1 
<code>.pio/libdeps/esp32cam/esp32-camera/driver/twi.c:61:24: error: 'rtc_gpio_desc' undeclared (first use in this function); did you mean 'rtc_io_desc'?
     uint32_t rtc_reg = rtc_gpio_desc[pin].reg;
                        ^~~~~~~~~~~~~
                        rtc_io_desc
.pio/libdeps/esp32cam/esp32-camera/driver/twi.c:61:24: note: each undeclared identifier is reported only once for each function it appears in
*** [.pio/build/esp32cam/lib84c/esp32-camera/driver/twi.o] Error 1
cc1: warning: command line option '-fno-rtti' is valid for C++/ObjC++ but not for
</code>

## Solution:
Change file
sdkconfig
Line 258:
'# CONFIG_RTCIO_SUPPORT_RTC_GPIO_DESC is not set
To
CONFIG_RTCIO_SUPPORT_RTC_GPIO_DESC=y


# Issue 2 Baud Rate

add to platformio.ini
<code>monitor_speed = 115200</code>

# Issue 4 Load Prohibited - Detected Camera not supported
Add KConfig file to src folder


# Issue 5 Watchdog deativate