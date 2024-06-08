#include "engine.h"
#include "nn.h"
#include "debugger.h"

void _backward(Value *v) {
    if (v->operation == NULL || v->requiresGrad == false) {
        // no grad to compute
        return;
    }
    // if (DEBUG == true) {
    //     printf("Backwards on:\n");
    //     displayValue(v);
    // }
    if (strcmp(v->operation, "+") == 0) {
        v->previous[0]->grad += 1.0 * v->grad;
        v->previous[1]->grad += 1.0 * v->grad;
        _backward(v->previous[0]);
        _backward(v->previous[1]);
    }
    else if (strcmp(v->operation, "*") == 0) {
        if (v->previous != NULL && v->previous[0] != NULL && v->previous[1] != NULL) {
            v->previous[0]->grad += v->previous[1]->data * v->grad;
            v->previous[1]->grad += v->previous[0]->data * v->grad;
            _backward(v->previous[0]);
            _backward(v->previous[1]);
        }
    }
    else if (strcmp(v->operation, "tanh") == 0) {
        v->previous[0]->grad += (1 - pow(v->data, 2)) * v->grad;
        _backward(v->previous[0]);
    }
    else if (strcmp(v->operation, "sigmoid") == 0) {
        v->previous[0]->grad += v->data * (1 - v->data) * v->grad;
        _backward(v->previous[0]);
    }
    else if (strcmp(v->operation, "leaky") == 0) {
        v->previous[0]->grad += (v->previous[0]->data > 0) ? 1 * v->grad : 0.01 * v->grad;
        _backward(v->previous[0]);
    }
    else if (strcmp(v->operation, "relu") == 0) {
        v->previous[0]->grad += (v->previous[0]->data > 0) ? 1 * v->grad : 0 * v->grad;
        _backward(v->previous[0]);
    }
    else if (strcmp(v->operation, "exp") == 0) {
        v->previous[0]->grad += v->data * v->grad;
        _backward(v->previous[0]);
    }
    else if (strcmp(v->operation, "pow2") == 0) {
        v->previous[0]->grad += (2 * v->previous[0]->data) * v->grad;
        _backward(v->previous[0]);
    }
}

Neuron *createNeuron(size_t nin) {
    Value **weights = malloc(nin * sizeof(Value*));
    Value *bias = malloc(sizeof(Value));
    if (bias == NULL || weights == NULL) {
        fprintf(stderr, "Memory allocation for Neuron bias or weights failed\n");
        exit(EXIT_FAILURE);
    }
    for (size_t i = 0; i < nin; i++) {
        char *label = malloc(10 * sizeof(char)); // we're assuming up to four digit number of weights
        sprintf(label, "weight[%zu]", i);
        weights[i] = newValue(randomUniform(-1.0, 1.0), NULL, NULL, label, 0.0, true);
        if (DEBUG == true) {
            printf("Neuron Weight[%zu] = %f\n", i, weights[i]->data);
        }
    }
    bias = newValue(randomUniform(-1.0, 1.0), NULL, NULL, "bias", 0.0, true);
    if (DEBUG == true) {
        printf("Neuron Bias = %f\n", bias->data);
    }
    Neuron *neuron = malloc(sizeof(Neuron));
    if (neuron == NULL) {
        fprintf(stderr, "Memory allocation for Neuron failed\n");
        exit(EXIT_FAILURE);
    }
    neuron->bias = bias;
    neuron->weights = weights;
    neuron->nin = nin;
    return neuron;
}

Value *forwardNeuron(Neuron *n, Value **x, size_t inputSize) {
    Value *sumProds = newValue(0.0, NULL, NULL, "sumProds", 0.0, true);
    for (size_t i = 0; i < inputSize; i++) {
        char *label = malloc(10 * sizeof(char));
        sprintf(label, "x*w[%zu]", i);
        Value *prod = mul(x[i], n->weights[i], label);
        sumProds = sum(sumProds, prod, "sumProds");
    }
    Value *out = htan(sum(sumProds, n->bias, "sumBias"), "tanh");
    return out;
}

Layer *createLayer(size_t nin, size_t nout) {
    Layer *layer = malloc(sizeof(Layer));
    Neuron **neurons = malloc(nout * sizeof(Neuron*));
    if (neurons == NULL || layer == NULL) {
        fprintf(stderr, "Memory allocation for Layer or Neurons failed\n");
        exit(EXIT_FAILURE);
    }
    for (size_t i = 0; i < nout; i++) {
        neurons[i] = createNeuron(nin);
    }
    layer->neurons = neurons;
    layer->nout = nout;
    return layer;
}

Value **forwardLayer(Layer *layer, Value **x, size_t inputSize) {
    Value **outputs = malloc(layer->nout * sizeof(Value*));
    for (size_t i = 0; i < layer->nout; i++) {
        outputs[i] = forwardNeuron(layer->neurons[i], x, inputSize);
    }
    return outputs; 
}

MLP *createMLP(size_t nin, size_t *nouts, size_t nLayers) {
    Layer **layers = malloc(nLayers * sizeof(Layer*));
    if (layers == NULL) {
        fprintf(stderr, "Memory allocation for Layers failed\n");
        exit(EXIT_FAILURE);
    }
    for (size_t i = 0; i < nLayers; i++) {
        size_t layerNin = (i == 0) ? nin : nouts[i-1];
        layers[i] = createLayer(layerNin, nouts[i]);
    }
    MLP *mlp = malloc(sizeof(MLP));
    if (mlp == NULL) {
        fprintf(stderr, "Memory allocation for MLP failed\n");
        exit(EXIT_FAILURE);
    }
    mlp->layers = layers;
    mlp->nLayers = nLayers;
    return mlp;
}

Value **forwardMLP(MLP *mlp, Value **x, size_t inputSize) {
    Value **layerOutputs = x;
    size_t layerInputSize = inputSize;
    for (size_t i = 0; i < mlp->nLayers; i++) {
        layerOutputs = forwardLayer(mlp->layers[i], layerOutputs, layerInputSize);
        layerInputSize = mlp->layers[i]->nout;
    }
    return layerOutputs;
}

void freeMLP(MLP *mlp) {
    for (size_t i = 0; i < mlp->nLayers; i++) {
        Layer *layer = mlp->layers[i];
        for (size_t j = 0; j < layer->nout; j++) {
            Neuron *neuron = layer->neurons[j];
            for (size_t k = 0; k < neuron->nin; k++) {
                free(neuron->weights[k]->label);
                free(neuron->weights[k]);
            }
            free(neuron->weights);
            free(neuron->bias);
            free(neuron);
        }
        free(layer->neurons);
        free(layer);
    }
    free(mlp->layers);
    free(mlp);
}

void stepMLP(MLP *mlp) {
    float learningRate = 0.01;
    for (size_t i = 0; i < mlp->nLayers; i++) {
        Layer *layer = mlp->layers[i];
        for (size_t j = 0; j < layer->nout; j++) {
            Neuron *neuron = layer->neurons[j];
            for (size_t k = 0; k < neuron->nin; k++) {
                neuron->weights[k]->data += learningRate * -neuron->weights[k]->grad;
                if (DEBUG == true) {
                    printf("Gradient for layer %zu, neuron %zu, weight %zu: %.8f\n", i, j, k, neuron->weights[k]->grad);
                }
            }
            neuron->bias->data += learningRate * -neuron->bias->grad;
            if (DEBUG == true) {
                printf("Gradient for layer %zu, neuron %zu, bias: %.8f\n", i, j, neuron->bias->grad);
            }
        }
    }
}

void zeroGrad(MLP *mlp) {
    for (size_t i = 0; i < mlp->nLayers; i++) {
        Layer *layer = mlp->layers[i];
        for (size_t j = 0; j < layer->nout; j++) {
            Neuron *neuron = layer->neurons[j];
            for (size_t k = 0; k < neuron->nin; k++) {
                neuron->weights[k]->grad = 0.0;
            }
            neuron->bias->grad = 0.0;
        }
    }
}

Value *squaredErrorLoss(Value *prediction, Value *target) {
    Value *error = sum(prediction, mul(target, newValue(-1.0, NULL, NULL, "neg_target", 0.0, false), "neg_target"), "error");
    Value *squaredError = pow2(error, "squaredError");
    return squaredError;
}

Value *averageLosses(Value **losses, int count) {
    Value *sumLosses = losses[0]; // Start with the first loss
    for (int i = 1; i < count; i++) {
        sumLosses = sum(sumLosses, losses[i], "sumLosses");
    }
    Value *avgLoss = mul(sumLosses, newValue(1.0 / count, NULL, NULL, "reciprocalCount", 0.0, false), "averageLoss"); // Multiply sum by reciprocal of count
    return avgLoss;
}