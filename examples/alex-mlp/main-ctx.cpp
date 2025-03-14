#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <inttypes.h>
#include "ggml.h"
#include "gguf.h"


/**
 * Essentially, the function we need to implement is: 
 * 1. parse_args: This is to handle the command line arguments
 * 2. load_model: Load the weights to the model. Note that we need to 
 *    save the model weights into GGUF first. And this function gets us the 
 *    weights from the GGUF file. 
 * 3. build_graph: This is to build the graph. This is where we define the forward propagation. 
 * 4. compute_graph: This is to compute the graph.
 */

// Define the structure for a two layer MLP
struct mlp_model {
    // Weights and biases for each layer
    struct ggml_tensor * w1;
    struct ggml_tensor * b1;
    struct ggml_tensor * w2;
    struct ggml_tensor * b2;
    struct ggml_context * ctx;
};


struct ProgramArgs {
    std::string model_path;
    std::string dot_path;
};

ProgramArgs parse_args(int argc, char** argv, 
                      const char* default_model_path,
                      const char* default_dot_path) {
    ProgramArgs args;
    args.model_path = default_model_path;
    args.dot_path = default_dot_path;

    if (argc >= 2) {
        args.model_path = argv[1];
    }
    if (argc >= 3) {
        args.dot_path = argv[2];
    }
    if (argc > 3) {
        fprintf(stderr, "Usage: %s [path/to/model.gguf] [path/to/output.dot]\n", argv[0]);
        exit(1);
    }

    fprintf(stderr, "Using model path: %s\n", args.model_path.c_str());
    fprintf(stderr, "Using dot output path: %s\n", args.dot_path.c_str());
    return args;
}


bool load_model(const std::string & fname, mlp_model & model) {
    struct gguf_init_params params = {
        /*.no_alloc   =*/ false,
        /*.ctx        =*/ &model.ctx,
    };

    struct gguf_context * ctx = gguf_init_from_file(fname.c_str(), params);
    if (!ctx) {
        fprintf(stderr, "%s: gguf_init_from_file() failed\n", __func__);
        return false;
    }

    model.w1 = ggml_get_tensor(model.ctx, "fc1.weight");
    model.b1 = ggml_get_tensor(model.ctx, "fc1.bias");
    model.w2 = ggml_get_tensor(model.ctx, "fc2.weight");
    model.b2 = ggml_get_tensor(model.ctx, "fc2.bias");

    if (!model.w1 || !model.b1 || !model.w2 || !model.b2) {
        fprintf(stderr, "%s: failed to load model tensors\n", __func__);
        gguf_free(ctx);
        return false;
    }

    gguf_free(ctx);
    return true;
}

void print_tensor_stats(const char* /*name*/, struct ggml_tensor* t) {
    float* data = (float*)t->data;
    size_t size = ggml_nelements(t);
    float sum = 0, min = INFINITY, max = -INFINITY;
    for (size_t i = 0; i < size; i++) {
        sum += data[i];
        if (data[i] < min) min = data[i];
        if (data[i] > max) max = data[i];
    }
}


struct ggml_cgraph * build_graph(
        struct ggml_context * ctx0,
        const mlp_model & model,
        const std::vector<float> & input_data) {

    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    struct ggml_tensor * input = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, input_data.size());
    memcpy(input->data, input_data.data(), ggml_nbytes(input));
    ggml_set_name(input, "input");

    ggml_tensor * cur = input;

    cur = ggml_mul_mat(ctx0, model.w1, cur);
    ggml_set_name(cur, "mul_mat_0");
    cur = ggml_add(ctx0, cur, model.b1);
    ggml_set_name(cur, "add_0");
    cur = ggml_relu(ctx0, cur);
    ggml_set_name(cur, "relu_0");

    cur = ggml_mul_mat(ctx0, model.w2, cur);
    ggml_set_name(cur, "mul_mat_1");
    cur = ggml_add(ctx0, cur, model.b2);
    ggml_set_name(cur, "add_1");

    // set the output tensor
    ggml_set_output(cur);
    ggml_set_name(cur, "final_result");

    ggml_build_forward_expand(gf, cur);

    return gf;
}

void compute_graph(
        struct ggml_cgraph * gf,
        struct ggml_context * ctx0,
        const int n_threads,
        const char * fname_cgraph) {
    ggml_graph_compute_with_ctx(ctx0, gf, n_threads);
}

int main(int argc, char ** argv) {
    srand(time(NULL));
    ggml_time_init();

    
    //==============================================
    // STEP 1: Read the argv and argc
    //==============================================
    const char* default_model_path = "/home/ubuntu/nexa-ggml/examples/mlp/model/mlp.gguf";
    const char* default_dot_path = "/home/azureuser/llama.cpp/examples/alex-mlp/model.dot";
    ProgramArgs args = parse_args(argc, argv, default_model_path, default_dot_path);
    
    
    //==============================================
    // STEP 2: Initialize model and load weights
    //==============================================
    mlp_model model;
    const int64_t t_start_us = ggml_time_us();
    if (!load_model(args.model_path, model)) {
        fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, args.model_path.c_str());
        return 1;
    }
    const int64_t t_load_us = ggml_time_us() - t_start_us;
    fprintf(stdout, "%s: loaded model in %8.2f ms\n", __func__, t_load_us / 1000.0f);


    //==============================================
    // STEP 3: Prepare tensors and build graph
    //==============================================
    std::vector<float> input_data = {0.5, 0.4, 0.3, 0.2, 0.1};
    size_t buf_size = 16*1024*1024;
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = build_graph(ctx0, model, input_data);
    
    // Optional: This is to create the graph
    ggml_graph_dump_dot(gf, NULL, args.dot_path.c_str());
    

    //==============================================
    // STEP 4: Compute graph and process results
    //==============================================
    const int64_t t_start_compute_us = ggml_time_us();
    compute_graph(gf, ctx0, 1, nullptr);
    const int64_t t_compute_us = ggml_time_us() - t_start_compute_us;
    fprintf(stdout, "%s: computed graph in %8.2f ms\n", __func__, t_compute_us / 1000.0f);

    struct ggml_tensor * result = ggml_graph_get_tensor(gf, "final_result");
    const float * output_data = ggml_get_data_f32(result);
    std::vector<float> output_vector(output_data, output_data + ggml_nelements(result));

    fprintf(stdout, "%s: output vector: [", __func__);
    for (size_t i = 0; i < output_vector.size(); ++i) {
        fprintf(stdout, "%f", output_vector[i]);
        if (i < output_vector.size() - 1) fprintf(stdout, ", ");
    }
    fprintf(stdout, "]\n");


    //==============================================
    // STEP 5: Cleanup and free memory
    //==============================================
    ggml_free(ctx0);
    ggml_free(model.ctx);
    return 0;
}