#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct Value {
    float data;
    char *operation;
    struct Value **previous;
    char *label;
    float grad;
} Value;

// Constructor for Value
Value *newValue(float data, char *operation, struct Value **previous, char *label, float grad) {
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

Value *sum(Value *v1, Value *v2, char *label) {
    Value **previous = createValuePointerArray(v1, v2);
    return newValue(v1->data + v2->data, "+", previous, label, 0.0);
}

Value *mul(Value *v1, Value *v2, char *label) {
    Value **previous = createValuePointerArray(v1, v2);
    return newValue(v1->data * v2->data, "*", previous, label, 0.0);
}

Value *htan(Value *v, char *label) {
    Value **previous = createValuePointerArray(v, NULL);
    float t = (exp(2 * v->data) - 1) / (exp(2 * v->data) + 1);
    return newValue(t, "tanh", previous, label, 0.0);
}

void _backward(Value *v) {
    if (v->operation == NULL) {
        // no grad to compute
        return;
    }
    if (strcmp(v->operation, "tanh") == 0) {
        printf("Backpropagating tanh\n");
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
}

void displayValue(Value *v) {
    float previous1 = (v->previous != NULL && v->previous[0] != NULL) ? v->previous[0]->data : 0.0;
    float previous2 = (v->previous != NULL && v->previous[1] != NULL) ? v->previous[1]->data : 0.0;
    printf("%s Value(data=%.2f, previous=(%.2f, %.2f), operation=%s, grad=%.2f)\n", v->label, v->data, previous1, previous2, v->operation, v->grad);
}

int main() {
    Value *w = newValue(-3.0, NULL, NULL, "w", 0.0);
    Value *v = newValue(0.5, NULL, NULL, "v", 0.0);
    Value *z = newValue(4.0, NULL, NULL, "z", 0.0);

    Value *s = sum(z, w, "sum");
    Value *p = mul(v, s, "prod");
    Value *Loss = htan(p, "Loss");

    Loss->grad = 1.0;
    _backward(Loss);
    displayValue(Loss);
    displayValue(p);
    displayValue(s);
    displayValue(w);
    displayValue(v);
    displayValue(z);

    free(w);
    free(v);
    free(z);
    free(s);
    free(p);
    free(Loss);

    return 0;
}