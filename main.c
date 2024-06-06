#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))

typedef struct Value {
    float data;
    char *operation;
    struct Value **previous;
    char *label;
    float grad;
    bool requiresGrad;
} Value;

typedef struct Neuron {
    struct Value **weights;
    struct Value *bias;
    size_t nin;
} Neuron;

typedef struct Layer {
    struct Neuron **neurons;
    size_t nout;
} Layer;

typedef struct MLP {
    struct Layer **layers;
    size_t nLayers;
} MLP;

bool DEBUG = false;

// Constructor for Value
Value *newValue(float data, char *operation, struct Value **previous, char *label, float grad, bool requiresGrad) {
    Value *v = malloc(sizeof(Value));
    if (v == NULL) {
        fprintf(stderr, "Memory allocation for new value failed\n");
        exit(EXIT_FAILURE);
    }
    v->data = data;
    v->operation = operation;
    v->previous = previous;
    v->label = label;
    v->grad = grad;
    v->requiresGrad = requiresGrad;
    return v;
}

Value **createValuePointerArray(Value *v1, Value *v2) {
    int size = (v2 == NULL) ? 1 : 2; 
    Value **array = malloc(size * sizeof(Value*)); 
    if (array == NULL) {
        fprintf(stderr, "Memory allocation for pointer array failed\n");
        exit(EXIT_FAILURE);
    }
    array[0] = v1;
    if (v2 != NULL) {
        array[1] = v2;
    }
    return array;
}

float randomUniform(float min, float max) {
    float normalized = rand() / (float)RAND_MAX;
    float range = max - min;
    float random = (normalized * range) + min; 
    return random;
}

Value *sum(Value *v1, Value *v2, char *label) {
    Value **previous = createValuePointerArray(v1, v2);
    return newValue(v1->data + v2->data, "+", previous, label, 0.0, true);
}

Value *mul(Value *v1, Value *v2, char *label) {
    Value **previous = createValuePointerArray(v1, v2);
    return newValue(v1->data * v2->data, "*", previous, label, 0.0, true);
}

Value *htan(Value *v, char *label) {
    Value **previous = createValuePointerArray(v, NULL);
    float t = (exp(2 * v->data) - 1) / (exp(2 * v->data) + 1);
    return newValue(t, "tanh", previous, label, 0.0, true);
}

Value *sigmoid(Value *v, char *label) {
    Value **previous = createValuePointerArray(v, NULL);
    float sigma = 1.0 / (1 + exp(-v->data));
    return newValue(sigma, "sigmoid", previous, label, 0.0, true);
}

Value *ReLU(Value *v, char *label) {
    Value **previous = createValuePointerArray(v, NULL);
    float relu = MAX(0, v->data);
    return newValue(relu, "relu", previous, label, 0.0, true);
}

Value *LeakyReLU(Value *v, char *label) {
    Value **previous = createValuePointerArray(v, NULL);
    float leaky = (v->data > 0) ? v->data : v->data * 0.01;
    return newValue(leaky, "leaky", previous, label, 0.0, true);
}

Value *euler(Value *v, char *label) {
    Value **previous = createValuePointerArray(v, NULL);
    float e = exp(v->data);
    return newValue(e, "exp", previous, label, 0.0, true);
}

Value *pow2(Value *v, char *label) {
    Value **previous = createValuePointerArray(v, NULL);
    float p = pow(v->data, 2);
    return newValue(p, "pow2", previous, label, 0.0, true);
}

void displayValue(Value *v) {
    float previous1 = (v->previous != NULL && v->previous[0] != NULL) ? v->previous[0]->data : 0.0;
    float previous2 = (v->previous != NULL && v->previous[1] != NULL) ? v->previous[1]->data : 0.0;
    char *label1 = (v->previous != NULL && v->previous[0] != NULL) ? v->previous[0]->label : "n/a";
    char *label2 = (v->previous != NULL && v->previous[1] != NULL) ? v->previous[1]->label : "n/a";
    printf("%s Value(data=%.2f, previous=(%.2f, %s | %.2f, %s), operation=%s, grad=%.2f)\n", v->label, v->data, previous1, label1, previous2, label2, v->operation, v->grad);
}

void _backward(Value *v) {
    if (v->operation == NULL) {
        // no grad to compute
        return;
    }
    if (v->requiresGrad == false) {
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
    for (int i = 0; i < nin; i++) {
        char *label = malloc(10 * sizeof(char)); // we're assuming up to four digit number of weights
        sprintf(label, "weight[%d]", i);
        weights[i] = newValue(randomUniform(-1.0, 1.0), NULL, NULL, label, 0.0, true);
        if (DEBUG == true) {
            printf("Neuron Weight[%d] = %f\n", i, weights[i]->data);
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
    for (int i = 0; i < inputSize; i++) {
        char *label = malloc(10 * sizeof(char));
        sprintf(label, "x*w[%d]", i);
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
    for (int i = 0; i < nout; i++) {
        neurons[i] = createNeuron(nin);
    }
    layer->neurons = neurons;
    layer->nout = nout;
    return layer;
}

Value **forwardLayer(Layer *layer, Value **x, size_t inputSize) {
    Value **outputs = malloc(layer->nout * sizeof(Value*));
    for (int i = 0; i < layer->nout; i++) {
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
    for (int i = 0; i < nLayers; i++) {
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
    for (int i = 0; i < mlp->nLayers; i++) {
        layerOutputs = forwardLayer(mlp->layers[i], layerOutputs, layerInputSize);
        layerInputSize = mlp->layers[i]->nout;
    }
    return layerOutputs;
}

void freeMLP(MLP *mlp) {
    for (int i = 0; i < mlp->nLayers; i++) {
        Layer *layer = mlp->layers[i];
        for (int j = 0; j < layer->nout; j++) {
            Neuron *neuron = layer->neurons[j];
            for (int k = 0; k < neuron->nin; k++) {
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
    for (int i = 0; i < mlp->nLayers; i++) {
        Layer *layer = mlp->layers[i];
        for (int j = 0; j < layer->nout; j++) {
            Neuron *neuron = layer->neurons[j];
            for (int k = 0; k < neuron->nin; k++) {
                neuron->weights[k]->data += learningRate * -neuron->weights[k]->grad;
                if (DEBUG == true) {
                    printf("Gradient for layer %d, neuron %d, weight %d: %.8f\n", i, j, k, neuron->weights[k]->grad);
                }
            }
            neuron->bias->data += learningRate * -neuron->bias->grad;
            if (DEBUG == true) {
                printf("Gradient for layer %d, neuron %d, bias: %.8f\n", i, j, neuron->bias->grad);
            }
        }
    }
}

void zeroGrad(MLP *mlp) {
    for (int i = 0; i < mlp->nLayers; i++) {
        Layer *layer = mlp->layers[i];
        for (int j = 0; j < layer->nout; j++) {
            Neuron *neuron = layer->neurons[j];
            for (int k = 0; k < neuron->nin; k++) {
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

void generateDot(MLP* mlp, Value** inputData, size_t inputSize, Value* prediction, Value* loss, int graphN) {
    char filename[20];
    snprintf(filename, sizeof(filename), "graph%d.dot", graphN);
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }

    fprintf(file, "digraph G {\n");
    fprintf(file, " rankdir=LR;\n");
    fprintf(file, " nodesep=1.0;\n");
    fprintf(file, " ranksep=1.5;\n");
    fprintf(file, " bgcolor=transparent;\n");
    fprintf(file, " fontcolor=white;\n");
    fprintf(file, " node [shape=circle, fontsize=11, fixedsize=true, width=0.8, color=white, fontcolor=white];\n");
    fprintf(file, " edge [arrowsize=0.7, color=white, penwidth=1.0, fontcolor=white, labeldistance=2, labelangle=45];\n");

    // Create input nodes
    fprintf(file, " subgraph cluster_input {\n");
    fprintf(file, " style=invis;\n");
    fprintf(file, " rank=same;\n");
    for (int i = 0; i < inputSize; i++) {
        fprintf(file, " input%d [label=\"%.4f\", xlabel=\"%s\", fontcolor=white];\n", i, inputData[i]->data, inputData[i]->label);
    }
    fprintf(file, " }\n");

    // Loop over hidden layers in the MLP
    for (int i = 0; i < mlp->nLayers - 1; i++) {
        Layer* layer = mlp->layers[i];
        fprintf(file, " subgraph cluster_layer%d {\n", i);
        fprintf(file, " style=invis;\n");
        fprintf(file, " rank=same;\n");
        // Loop over neurons in the layer
        for (int j = 0; j < layer->nout; j++) {
            Neuron* neuron = layer->neurons[j];
            fprintf(file, " hidden%d_%d [label=\"%.4f\", xlabel=\"%s\"];\n", i, j, neuron->bias->data, neuron->bias->label);
        }
        fprintf(file, " }\n");
    }

    // Create output node
    Layer* output_layer = mlp->layers[mlp->nLayers - 1];
    Neuron* output_neuron = output_layer->neurons[0];
    fprintf(file, " output [label=\"%.4f\", xlabel=\"%s\"];\n", output_neuron->bias->data, output_neuron->bias->label);

    // Create prediction and loss nodes
    fprintf(file, " subgraph cluster_output {\n");
    fprintf(file, " style=invis;\n");
    fprintf(file, " rank=same;\n");
    fprintf(file, " prediction [label=\"%.4f\", xlabel=\"%s\", shape=circle];\n", prediction->data, prediction->label);
    fprintf(file, " loss [label=\"%.4f\", xlabel=\"%s\", shape=circle];\n", loss->data, loss->label);
    fprintf(file, " }\n");

    // Connect input nodes to first hidden layer
    for (int i = 0; i < inputSize; i++) {
        for (int j = 0; j < mlp->layers[0]->nout; j++) {
            fprintf(file, " input%d -> hidden0_%d [label=\"%s\\n%.4f\", fontsize=9];\n", i, j, mlp->layers[0]->neurons[j]->weights[i]->label, mlp->layers[0]->neurons[j]->weights[i]->data);
        }
    }

    // Connect hidden layers
    for (int i = 0; i < mlp->nLayers - 2; i++) {
        Layer* currLayer = mlp->layers[i];
        Layer* nextLayer = mlp->layers[i + 1];
        for (int j = 0; j < currLayer->nout; j++) {
            for (int k = 0; k < nextLayer->nout; k++) {
                fprintf(file, " hidden%d_%d -> hidden%d_%d [label=\"%s\\n%.4f\", fontsize=9];\n", i, j, i + 1, k, nextLayer->neurons[k]->weights[j]->label, nextLayer->neurons[k]->weights[j]->data);
            }
        }
    }

    // Connect last hidden layer to output node
    Layer* lastHiddenLayer = mlp->layers[mlp->nLayers - 2];
    for (int i = 0; i < lastHiddenLayer->nout; i++) {
        fprintf(file, " hidden%zu_%d -> output;\n", mlp->nLayers - 2, i);
    }

    // Connect output node to prediction and loss nodes
    fprintf(file, " output -> prediction;\n");
    fprintf(file, " prediction -> loss;\n");

    fprintf(file, "}\n");
    fclose(file);
}

int main() {
    srand((unsigned int)time(NULL));
    // DATA
    // 4 data points, each containing an array of 3 Values
    int numFeatures = 3;
    int numDataPoints = 4;
    Value ***data = malloc(numDataPoints * sizeof(Value**));
    float predefinedValues[4][3] = {
        {2, 3, -1},
        {3, -1, 0.5},
        {0.5, 1, 1},
        {1, 1, -1}
    };

    for (int i = 0; i < numDataPoints; i++) {
        data[i] = malloc(numFeatures * sizeof(Value*));
        for (int j = 0; j < numFeatures; j++) {
            char *label = malloc(20 * sizeof(char));
            sprintf(label, "data[%d][%d]", i, j);
            data[i][j] = newValue(predefinedValues[i][j], NULL, NULL, label, 0.0, false);
            if (DEBUG == true) {
                printf("creating data point %f with label %s\n", data[i][j]->data, label);
            }
        }
    }
    // TARGETS
    float *targets = malloc(4 * sizeof(float));
    targets[0] = -1.0;
    targets[1] = 1.0;
    targets[2] = -1.0;
    targets[3] = 1.0;
    // NUMBER OF OUTPUTS PER LAYER
    size_t *nouts = malloc(3 * sizeof(size_t));
    nouts[0] = 4;
    nouts[1] = 3;
    nouts[2] = 1;

    MLP *mlp = createMLP(3, nouts, 3);
    // TRAINING LOOP
    int datasetSize = 4;
    int numEpochs = 50;
    size_t inputSize = 3;
    for (int epoch = 0; epoch < numEpochs; epoch++) {
        Value **epochLosses = malloc(datasetSize * sizeof(Value*));
        for (int i = 0; i < datasetSize; i++) {
            // Forward pass
            Value **output = forwardMLP(mlp, data[i], inputSize);

            // Compute loss
            Value *target = newValue(targets[i], NULL, NULL, "target", 0.0, false);
            if (DEBUG == true) {
                displayValue(*output);
                displayValue(target);
            }
            Value *loss = squaredErrorLoss(*output, target);
            epochLosses[i] = loss;

            // Backward pass
            loss->grad = 1.0;
            _backward(loss);
            
            // Update weights
            stepMLP(mlp);
            if (epoch == numEpochs-1) {
                displayValue(*output);
                displayValue(target);
                generateDot(mlp, data[i], numFeatures, *output, loss, i);
            }

            // Zero gradients
            zeroGrad(mlp);
        }
        // Compute average loss for the epoch
        Value *avgLoss = averageLosses(epochLosses, datasetSize);
        printf("Epoch %d, Loss: %.4f\n", epoch, avgLoss->data);

        // Free memory
        for (int i = 0; i < datasetSize; i++) {
            free(epochLosses[i]);
        }
        free(epochLosses);
        free(avgLoss);
    }
    freeMLP(mlp);
    for (int i = 0; i < numDataPoints; i++) {
         for (int j = 0; j < numFeatures; j++) {
             free(data[i][j]->label);
             free(data[i][j]);
         }
         free(data[i]);
     }
    free(data);
    free(targets);
    free(nouts);
    return 0;
}