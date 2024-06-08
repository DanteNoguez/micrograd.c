#ifndef NN_H
#define NN_H
#include "engine.h"
#include <stddef.h>

// Structs
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

// Constructors
Neuron *createNeuron(size_t nin);
Layer *createLayer(size_t nin, size_t nout);
MLP *createMLP(size_t nin, size_t *nouts, size_t nLayers);

// Operations
void _backward(Value *v);
Value *forwardNeuron(Neuron *n, Value **x, size_t inputSize);
Value **forwardLayer(Layer *layer, Value **x, size_t inputSize);
Value **forwardMLP(MLP *mlp, Value **x, size_t inputSize);
void zeroGrad(MLP *mlp);
void stepMLP(MLP *mlp);
Value *squaredErrorLoss(Value *prediction, Value *target);
Value *averageLosses(Value **losses, int count);

// Helpers
void freeMLP(MLP *mlp);

#endif // NN_H