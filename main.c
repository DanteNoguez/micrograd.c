#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float data;
    char previous[50];
    char operation;
    char label;
} Value;

// Constructor for Value
Value* newValue(float data, char operation, char* previous, char label) {
    Value* v = malloc(sizeof(Value));
    if (v == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE); // Always check for malloc failure
    }
    v->data = data;
    v->operation = operation;
    strcpy(v->previous, previous);
    v->label = label;
    return v;
}

// Method to add Values
Value* addValues(Value* v1, Value* v2, char label) {
    char buffer[50];
    sprintf(buffer, "Value(data=%.2f), Value(data=%.2f)\n", v1->data, v2->data);
    return newValue(v1->data + v2->data, '+', buffer, label);
}

// Method to multiply Values
Value* multiplyValues(Value* v1, Value* v2, char label) {
    char buffer[50];
    sprintf(buffer, "Value(data=%.2f), Value(data=%.2f)\n", v1->data, v2->data);
    return newValue(v1->data * v2->data, '*', buffer, label);
}

// Display the Value
void displayValue(Value* v) {
    printf("Value(data=%.2f, previous=%s, operation=%c)\n", v->data, v->previous, v->operation);
}

int main() {
    Value* w = newValue(10, '\0', "", 'w');
    Value* v = newValue(20, '\0', "", 'v');

    Value* sum = addValues(w, v, 's');
    displayValue(sum);
    Value* product = multiplyValues(w, v, 'p');
    displayValue(product);

    free(w);
    free(v);
    free(sum);
    free(product);

    return 0;
}
