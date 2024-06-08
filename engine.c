#include "engine.h"

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