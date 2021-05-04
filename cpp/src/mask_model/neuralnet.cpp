#include "neuralnet.hpp"

NN::NN(const void* model, const unsigned int tensor_arena_size){
    m_tensor_arena_size = tensor_arena_size;
    //tensor_arena = (uint8_t*) heap_caps_calloc(TENSOR_ARENA_SIZE, 1, MALLOC_CAP_8BIT);
    m_tensor_arena = (uint8_t*) malloc(tensor_arena_size);
    m_micro_error_reporter = new tflite::MicroErrorReporter();

    const tflite::Model* m_model = tflite::GetModel(model);
    if (m_model->version() != TFLITE_SCHEMA_VERSION) {
        TF_LITE_REPORT_ERROR(
            m_micro_error_reporter,
            "Model provided is schema version %d not equal "
            "to supported version %d.",
            m_model->version(),
            TFLITE_SCHEMA_VERSION);
        return;
    }

    static tflite::MicroMutableOpResolver<7> mop_res;
    mop_res.AddQuantize();
    mop_res.AddReshape();
    mop_res.AddConv2D();
    mop_res.AddMaxPool2D();
    mop_res.AddSoftmax();
    mop_res.AddRelu();
    mop_res.AddFullyConnected();

    // Build an interpreter to run the model with.
    m_interpreter = new tflite::MicroInterpreter(
                        m_model,
                        mop_res,
                        m_tensor_arena,
                        tensor_arena_size,
                        m_micro_error_reporter);

    if (kTfLiteOk != m_interpreter->AllocateTensors() ) {
        TF_LITE_REPORT_ERROR(m_micro_error_reporter, "AllocateTensors() failed");
        return;
    }
};

int8_t* NN::run(uint8_t* buff, const unsigned int buff_len){
    TfLiteTensor* input = m_interpreter->input(0);
    for (unsigned int ii = 0; ii < buff_len; ii++){
        input->data.int8[ii] = static_cast<int8>(static_cast<int16>(buff[ii]) - 128);
    }
    if (kTfLiteOk != m_interpreter->Invoke()) {
        TF_LITE_REPORT_ERROR(m_micro_error_reporter, "Invoke failed.");
    }
    TfLiteTensor* output = m_interpreter->output(0);
    return output->data.int8;
};

NN::~NN(){
    delete m_micro_error_reporter;
    delete m_interpreter;
};