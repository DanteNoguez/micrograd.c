#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

typedef struct Value {
    float data;
    char *operation;
    struct Value **previous;
    char *label;
    float grad;
    bool requires_grad;
} Value;

typedef struct Neuron {
    struct Value **weights;
    struct Value *bias;
    int nin;
} Neuron;

typedef struct Layer {
    struct Neuron **neurons;
    int nout;
} Layer;

typedef struct MLP {
    struct Layer **layers;
    int nouts_size;
} MLP;

bool DEBUG = true;

// Constructor for Value
Value *newValue(float data, char *operation, struct Value **previous, char *label, float grad, bool requires_grad) {
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
    v->requires_grad = requires_grad;
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

float random_uniform(float min, float max) {
    float normalized = rand() / (float)RAND_MAX;
    float range = max - min;
    float random = (normalized * range) + min; 
    return random;
}

Neuron *createNeuron(int nin) {
    Value **weights = malloc(nin * sizeof(Value*));
    Value *bias = malloc(sizeof(Value));
    for (int i = 0; i < nin; i++) {
        char *label = malloc(10 * sizeof(char)); // we're assuming up to four digit number of weights
        sprintf(label, "weight[%d]", i);
        weights[i] = newValue(random_uniform(-1, 1), NULL, NULL, label, 0.0, true);
        if (DEBUG == true) {
            printf("Neuron Weight[%d] = %f\n", i, weights[i]->data);
        }
    }
    bias = newValue(random_uniform(-1, 1), NULL, NULL, "bias", 0.0, true);
    if (DEBUG == true) {
        printf("Neuron Bias = %f\n", bias->data);
    }
    Neuron *neuron = malloc(sizeof(Neuron));
    neuron->bias = bias;
    neuron->weights = weights;
    neuron->nin = nin;
    return neuron;
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


void _backward(Value *v) {
    if (v->operation == NULL) {
        // no grad to compute
        return;
    }
    if (v->requires_grad == false) {
        // no grad to compute
        return;
    }
    if (strcmp(v->operation, "+") == 0) {
        v->previous[0]->grad += (v->previous != NULL && v->previous[0] != NULL) ? 1.0 * v->grad : 0.0;
        v->previous[1]->grad += (v->previous != NULL && v->previous[1] != NULL) ? 1.0 * v->grad : 0.0;
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
    else if (strcmp(v->operation, "exp") == 0) {
        v->previous[0]->grad += v->data * v->grad;
        _backward(v->previous[0]);
    }
    else if (strcmp(v->operation, "pow2") == 0) {
        v->previous[0]->grad += (2 * v->data) * v->grad;
        _backward(v->previous[0]);
    }
}

void displayValue(Value *v) {
    float previous1 = (v->previous != NULL && v->previous[0] != NULL) ? v->previous[0]->data : 0.0;
    float previous2 = (v->previous != NULL && v->previous[1] != NULL) ? v->previous[1]->data : 0.0;
    printf("%s Value(data=%.2f, previous=(%.2f, %.2f), operation=%s, grad=%.2f)\n", v->label, v->data, previous1, previous2, v->operation, v->grad);
}

Value *forward_neuron(Neuron *n, Value **x) {
    Value **prods = malloc(n->nin * sizeof(Value*));
    for (int i = 0; i < n->nin; i++) {
        char *label = malloc(10 * sizeof(char));
        sprintf(label, "x*w[%d]", i);
        if (DEBUG == true) {
            printf("Multiplying input %s with weight %s\n", x[i]->label, n->weights[i]->label);
        }
        prods[i] = mul(x[i], n->weights[i], label);
    }

    Value *sum_prods = prods[0]; // start with the first product
    for (int i = 1; i < n->nin; i++) {
        char *label = malloc(10 * sizeof(char));
        sprintf(label, "sum[%d]", i);
        sum_prods = sum(sum_prods, prods[i], label); // sum up the products
    }
    Value *out = htan(sum_prods, "tanh");
    return out;
}

Layer *createLayer(int nin, int nout) {
    Layer *layer = malloc(sizeof(Layer));
    Neuron **neurons = malloc(nout * sizeof(Neuron*));
    for (int i = 0; i < nout; i++) {
        neurons[i] = createNeuron(nin);
    }
    layer->neurons = neurons;
    layer->nout = nout;
    return layer;
}

Value **forward_layer(Layer *layer, Value **x) {
    Value **outputs = malloc(layer->nout * sizeof(Value*));
    for (int i = 0; i < layer->nout; i++) {
        outputs[i] = forward_neuron(layer->neurons[i], x);
    }
    return outputs;
}

MLP *createMLP(int nin, int *nouts, int nouts_size) {
    Layer **layers = malloc(nouts_size * sizeof(Layer*));
    for (int i = 0; i < nouts_size; i++) {
        int layer_nin = (i == 0) ? nin : nouts[i-1];
        layers[i] = createLayer(layer_nin, nouts[i]);
    }
    MLP *mlp = malloc(sizeof(MLP));
    mlp->layers = layers;
    mlp->nouts_size = nouts_size;
    return mlp;
}

Value **forward_MLP(MLP *mlp, Value **x, int input_size) {
    Value **outputs = malloc(input_size * sizeof(Value*));
    for (int i = 0; i < input_size; i++) {
        Value **input = &x[i];
        Value **layer_outputs = input;
        for (int j = 0; j < mlp->nouts_size; j++) {
            if (DEBUG == true) {
            printf("Forwarding layer %d for input %d\n", j, i);
            }
            layer_outputs = forward_layer(mlp->layers[j], layer_outputs);
        }
        outputs[i] = layer_outputs[0];
    }
    return outputs;
}

void freeMLP(MLP *mlp) {
    for (int i = 0; i < mlp->nouts_size; i++) {
        Layer *layer = mlp->layers[i];
        for (int j = 0; j < layer->nout; j++) {
            Neuron *neuron = layer->neurons[j];
            for (int k = 0; k < neuron->nin; k++) {
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

int main() {
    srand((unsigned int)time(NULL));
    // DATASET
    Value **data = malloc(6 * sizeof(Value*));
    data[0] = newValue(1.0, NULL, NULL, "one", 0.0, false);
    data[1] = newValue(2.0, NULL, NULL, "two", 0.0, false);
    data[2] = newValue(3.0, NULL, NULL, "three", 0.0, false);
    data[3] = newValue(4.0, NULL, NULL, "four", 0.0, false);
    data[4] = newValue(5.0, NULL, NULL, "five", 0.0, false);
    data[5] = newValue(6.0, NULL, NULL, "six", 0.0, false);
    // TARGETS
    float **targets = malloc(6 * sizeof(float*));
    for(int i = 0; i < 6; i++) {
    targets[i] = malloc(sizeof(float));
    }
    *targets[0] = -1.0;
    *targets[1] = 1.0;
    *targets[2] = -1.0;
    *targets[3] = 1.0;
    *targets[4] = -1.0;
    *targets[5] = 1.0;
    // NUMBER OF OUTPUTS PER LAYER
    int *nouts = malloc(3 * sizeof(int));
    nouts[0] = 4;
    nouts[1] = 4;
    nouts[2] = 1;

    // FORWARD MLP ON EACH X
    MLP *mlp = createMLP(1, nouts, 3);
    Value **out = forward_MLP(mlp, data, 6);
    for (int i = 0; i < 6; i++) {
        out[i]->grad = 1.0;
        _backward(out[i]);
        displayValue(out[i]);
        printf("TARGET VALUE IS: %.2f\n", *targets[i]);
        printf("LOSS: %.2f\n", pow(out[i]->data - *targets[i], 2));
    }
    freeMLP(mlp);
    return 0;
}