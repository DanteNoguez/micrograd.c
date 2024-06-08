#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

#include "engine.h"
#include "nn.h"
#include "visualizer.h"
#include "debugger.h"

const bool DEBUG = false;

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