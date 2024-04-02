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
    struct Value **inputs;
    int nin;
} Neuron;

typedef struct Layer {
    struct Neuron **neurons;
    int nout;
} Layer;

bool DEBUG = false;

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

Neuron *createNeuron(int nin, float *x) {
    Value **weights = malloc(nin * sizeof(Value*));
    Value *bias = malloc(sizeof(Value));
    Value **inputs = malloc(nin * sizeof(Value*));
    for (int i = 0; i < nin; i++) {
        inputs[i] = newValue(x[i], NULL, NULL, NULL, 0.0, false);
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
    neuron->inputs = inputs;
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
    if (strcmp(v->operation, "tanh") == 0) {
        v->previous[0]->grad += (1 - pow(v->data, 2)) * v->grad;
        _backward(v->previous[0]);
    }
    else if (strcmp(v->operation, "+") == 0) {
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

Value *forward_neuron(Neuron *n) {
    Value **prods = malloc(n->nin * sizeof(Value*));
    for (int i = 0; i < n->nin; i++) {
        char *label = malloc(10 * sizeof(char));
        sprintf(label, "x*w[%d]", i);
        prods[i] = mul(n->inputs[i], n->weights[i], label);
    }

    Value *sum_prods = prods[0]; // start with the first product
    for (int i = 1; i < n->nin; i++) {
        char *label = malloc(10 * sizeof(char));
        sprintf(label, "sum[%d]", i);
        sum_prods = sum(sum_prods, prods[i], label); // sum up the products
    }
    Value *out = htan(sum_prods, "tanh");

    for (int i = 0; i < n->nin; i++) {
        free(prods[i]->label);
        free(prods[i]);
    }
    free(prods);

    return out;
}

Layer *createLayer(int nin, int nout, float *x) {
    Layer *layer = malloc(sizeof(Layer));
    Neuron **neurons = malloc(nout * sizeof(Neuron*));
    for (int i = 0; i < nout; i++) {
        neurons[i] = createNeuron(nin, x);
    }
    layer->neurons = neurons;
    layer->nout = nout;
    return layer;
}

Value **forward_layer(Layer *layer) {
    Value **outputs = malloc(layer->nout * sizeof(Value*));
    for (int i = 0; i < layer->nout; i++) {
        outputs[i] = forward_neuron(layer->neurons[i]);
    }
    return outputs;
}

int main() {
    srand((unsigned int)time(NULL));
    float *array = malloc(3 * sizeof(float));
    array[0] = 1.0;
    array[1] = -2.0;
    array[2] = 3.0;
    Layer *layer = createLayer(3, 2, array);
    Value **out = forward_layer(layer);
    out[0]->grad = 1.0;
    out[1]->grad = 1.0;
    _backward(out[0]);
    _backward(out[1]);
    displayValue(out[0]);
    displayValue(out[1]);
    free(layer);
    free(out);
    return 0;
}