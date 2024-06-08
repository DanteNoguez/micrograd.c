#ifndef ENGINE_H
#define ENGINE_H
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

typedef struct Value {
    float data;
    char *operation;
    struct Value **previous;
    char *label;
    float grad;
    bool requiresGrad;
} Value;

// Constructors
Value *newValue(float data, char *operation, struct Value **previous, char *label, float grad, bool requiresGrad);
Value **createValuePointerArray(Value *v1, Value *v2);

// Helpers
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
float randomUniform(float min, float max);
void displayValue(Value *v);

// Operations
Value *sum(Value *v1, Value *v2, char *label);
Value *mul(Value *v1, Value *v2, char *label);
Value *htan(Value *v, char *label);
Value *sigmoid(Value *v, char *label);
Value *ReLU(Value *v, char *label);
Value *LeakyReLU(Value *v, char *label);
Value *euler(Value *v, char *label);
Value *pow2(Value *v, char *label);

#endif // ENGINE_H