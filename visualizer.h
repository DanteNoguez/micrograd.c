#ifndef VISUALIZER_H
#define VISUALIZER_H
#include <stdio.h>
#include <stdlib.h>
#include "engine.h"
#include "nn.h"

void generateDot(MLP* mlp, Value** inputData, size_t inputSize, Value* prediction, Value* loss, int graphN);

#endif // VISUALIZER_H