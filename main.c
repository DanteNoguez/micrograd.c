#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    float data;
    char* operation;
    struct Value** previous;
    char* label;
    float grad;
} Value;

// Constructor for Value
Value* newValue(float data, char* operation, struct Value** previous, char* label, float grad) {
    Value* v = malloc(sizeof(Value));
    if (v == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    v->data = data;
    v->operation = operation;
    v->previous = previous;
    v->label = label;
    v->grad = grad;
    return v;
}

Value* sum(Value* v1, Value* v2, char* label) {
    int buffer_size = snprintf(NULL, 0, "Value(data=%.2f), Value(data=%.2f)\n", v1->data, v2->data) + 1; // +1 for '\0'
    char* buffer = malloc(buffer_size);
    if (buffer == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    sprintf(buffer, "Value(data=%.2f), Value(data=%.2f)\n", v1->data, v2->data);
    return newValue(v1->data + v2->data, "+", buffer, label);
}

Value* mul(Value* v1, Value* v2, char* label) {
    int buffer_size = snprintf(NULL, 0, "Value(data=%.2f), Value(data=%.2f)\n", v1->data, v2->data) + 1; // +1 for '\0'
    char* buffer = malloc(buffer_size);
    if (buffer == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    sprintf(buffer, "Value(data=%.2f), Value(data=%.2f)\n", v1->data, v2->data);
    return newValue(v1->data * v2->data, "*", buffer, label);
}

Value* htan(Value* v, char* label) {
    int buffer_size = snprintf(NULL, 0, "Value(data=%.2f)\n", v->data) + 1; // +1 for '\0'
    char* buffer = malloc(buffer_size);
    if (buffer == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    sprintf(buffer, "Value(data=%.2f)\n", v->data);
    float t = (exp(2 * v->data) - 1) / (exp(2 * v->data) + 1);
    Value* result = newValue(t, "tanh", buffer, label);
    return result;
}

float _backward(Value* v) {
    if (strcmp(v->operation, "tanh") == 0) {
        return 1 - pow(v->data, 2);
    }
    return 0;
}

void displayValue(Value* v) {
    printf("Value(data=%.2f, previous=%s, operation=%s)\n", v->data, v->previous, v->operation);
}

int main() {
    Value* w = newValue(10, NULL, NULL, "w", 0);
    Value* v = newValue(20, NULL, NULL, "w", 0);

    Value* s = sum(w, v, "s");
    displayValue(s);
    Value* p = mul(w, v, "p");
    displayValue(p);
    Value* t = htan(p, "t");
    displayValue(t);
    float grad = _backward(t);
    printf("Gradient: %.2f\n", grad);
    float grad_p = _backward(p);

    free(w);
    free(v);
    free(s);
    free(p);
    free(t);

    return 0;
}