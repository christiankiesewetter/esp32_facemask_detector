#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

class NN{
    private:
        unsigned int m_tensor_arena_size;
        uint8_t* m_tensor_arena;

        tflite::ErrorReporter* m_micro_error_reporter;
        tflite::MicroInterpreter* m_interpreter;
        
    public:
        NN(const void* mask_model_tflite, const unsigned int tensor_arena_size);
        int8_t* run(uint8_t* buff, const unsigned int buff_len);
        ~NN();
        
};