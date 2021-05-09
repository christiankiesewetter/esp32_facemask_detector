# ESP32 Facemask Detector

## Introduction
I wanted to build a project with a microcontroller, which supposedly was able to
deal with a small neural network to predict whether or not a person in front of
the camera is wearing a mask.
For the sake of simplicity I found the Esp32 Ai Thinker to be a good choice.
The ESP Camera board is cheap, can be used together with the Arduino framework,
but also has a comparably huge amount of flash memory, to store even more sophisticated
models. Additionaly I really wanted to re-learn C++ and therefore chose to start
the whole project on the ESP-IDF platform using PlatformIO.

## What it does
The ESP32 inbuilt LED will flash unless you put a facemask on.

## Folder Structure
<code>cpp</code> contains the source files for the esp32cam inference.
<code>python</code> contains the source files I used for the python training.

## Future Projects
transfer the project to the ESP native dl3_matrix Framework,
since inference on tf is quite slow ~1-2 seconds for the current implementation.
I assume this is due to the overhead of the tfmicro framework
(the final model file is quite much in the same range as the face recognition models
implemented in the esp32-webcam-server sketch). But I wanted to keep it as simple
as possible for me to learn. I'm also trying to document some of the pitfalls I
stepped into.
