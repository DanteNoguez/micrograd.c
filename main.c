#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))

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
    int n_layers;
} MLP;

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

void _backward(Value *v) {
    if (v->operation == NULL) {
        // no grad to compute
        return;
    }
    if (v->requires_grad == false) {
        // no grad to compute
        return;
    }
    // if (DEBUG == true) {
    //     printf("Backwards on:\n");
    //     displayValue(v);
    // }
    if (strcmp(v->operation, "+") == 0) {
        v->previous[0]->grad += 1.0 * v->grad;
        v->previous[1]->grad += 1.0 * v->grad;
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
    else if (strcmp(v->operation, "sigmoid") == 0) {
        v->previous[0]->grad += v->data * (1 - v->data) * v->grad;
        _backward(v->previous[0]);
    }
    else if (strcmp(v->operation, "leaky") == 0) {
        v->previous[0]->grad += (v->previous[0]->data > 0) ? 1 * v->grad : 0.01 * v->grad;
        _backward(v->previous[0]);
    }
    else if (strcmp(v->operation, "relu") == 0) {
        v->previous[0]->grad += (v->previous[0]->data > 0) ? 1 * v->grad : 0 * v->grad;
        _backward(v->previous[0]);
    }
    else if (strcmp(v->operation, "exp") == 0) {
        v->previous[0]->grad += v->data * v->grad;
        _backward(v->previous[0]);
    }
    else if (strcmp(v->operation, "pow2") == 0) {
        v->previous[0]->grad += (2 * v->previous[0]->data) * v->grad;
        _backward(v->previous[0]);
    }
}

Neuron *createNeuron(int nin) {
    Value **weights = malloc(nin * sizeof(Value*));
    Value *bias = malloc(sizeof(Value));
    for (int i = 0; i < nin; i++) {
        char *label = malloc(10 * sizeof(char)); // we're assuming up to four digit number of weights
        sprintf(label, "weight[%d]", i);
        weights[i] = newValue(randomUniform(-1.0, 1.0), NULL, NULL, label, 0.0, true);
        if (DEBUG == true) {
            printf("Neuron Weight[%d] = %f\n", i, weights[i]->data);
        }
    }
    bias = newValue(randomUniform(-1.0, 1.0), NULL, NULL, "bias", 0.0, true);
    if (DEBUG == true) {
        printf("Neuron Bias = %f\n", bias->data);
    }
    Neuron *neuron = malloc(sizeof(Neuron));
    neuron->bias = bias;
    neuron->weights = weights;
    neuron->nin = nin;
    return neuron;
}

Value *forwardNeuron(Neuron *n, Value **x, int input_size) {
    Value *sum_prods = newValue(0.0, NULL, NULL, "sum_prods", 0.0, true);
    for (int i = 0; i < input_size; i++) {
        char *label = malloc(10 * sizeof(char));
        sprintf(label, "x*w[%d]", i);
        Value *prod = mul(x[i], n->weights[i], label);
        sum_prods = sum(sum_prods, prod, "sum_prods");
    }
    Value *out = htan(sum(sum_prods, n->bias, "sum_bias"), "tanh");
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

Value **forwardLayer(Layer *layer, Value **x, int input_size) {
    Value **outputs = malloc(layer->nout * sizeof(Value*));
    for (int i = 0; i < layer->nout; i++) {
        outputs[i] = forwardNeuron(layer->neurons[i], x, input_size);
    }
    return outputs; 
}

MLP *createMLP(int nin, int *nouts, int n_layers) {
    Layer **layers = malloc(n_layers * sizeof(Layer*));
    for (int i = 0; i < n_layers; i++) {
        int layer_nin = (i == 0) ? nin : nouts[i-1];
        layers[i] = createLayer(layer_nin, nouts[i]);
    }
    MLP *mlp = malloc(sizeof(MLP));
    mlp->layers = layers;
    mlp->n_layers = n_layers;
    return mlp;
}

Value **forwardMLP(MLP *mlp, Value **x, int input_size) {
    Value **layer_outputs = x;
    int layer_input_size = input_size;
    for (int i = 0; i < mlp->n_layers; i++) {
        layer_outputs = forwardLayer(mlp->layers[i], layer_outputs, layer_input_size);
        layer_input_size = mlp->layers[i]->nout;
    }
    return layer_outputs;
}

void freeMLP(MLP *mlp) {
    for (int i = 0; i < mlp->n_layers; i++) {
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

void stepMLP(MLP *mlp) {
    float learning_rate = 0.01;
    for (int i = 0; i < mlp->n_layers; i++) {
        Layer *layer = mlp->layers[i];
        for (int j = 0; j < layer->nout; j++) {
            Neuron *neuron = layer->neurons[j];
            for (int k = 0; k < neuron->nin; k++) {
                neuron->weights[k]->data += learning_rate * -neuron->weights[k]->grad;
                if (DEBUG == true) {
                    printf("Gradient for layer %d, neuron %d, weight %d: %.8f\n", i, j, k, neuron->weights[k]->grad);
                }
            }
            neuron->bias->data += learning_rate * -neuron->bias->grad;
            if (DEBUG == true) {
                printf("Gradient for layer %d, neuron %d, bias: %.8f\n", i, j, neuron->bias->grad);
            }
        }
    }
}

void zeroGrad(MLP *mlp) {
    for (int i = 0; i < mlp->n_layers; i++) {
        Layer *layer = mlp->layers[i];
        for (int j = 0; j < layer->nout; j++) {
            Neuron *neuron = layer->neurons[j];
            for (int k = 0; k < neuron->nin; k++) {
                neuron->weights[k]->grad = 0.0;
            }
            neuron->bias->grad = 0.0;
        }
    }
}

Value *squared_error_loss(Value *prediction, Value *target) {
    Value *error = sum(prediction, mul(target, newValue(-1.0, NULL, NULL, "neg_target", 0.0, false), "neg_target"), "error");
    Value *squared_error = pow2(error, "squared_error");
    return squared_error;
}

Value *average_losses(Value **losses, int count) {
    Value *sum_losses = losses[0]; // Start with the first loss
    for (int i = 1; i < count; i++) {
        sum_losses = sum(sum_losses, losses[i], "sum_losses");
    }
    Value *avg_loss = mul(sum_losses, newValue(1.0 / count, NULL, NULL, "reciprocal_count", 0.0, false), "average_loss"); // Multiply sum by reciprocal of count
    return avg_loss;
}

void generate_dot(MLP* mlp, Value** input_data, int input_size, Value* prediction, Value* loss) {
    FILE* file = fopen("graph.dot", "w");
    if (file == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }

    fprintf(file, "digraph G {\n");
    fprintf(file, "  rankdir=LR;\n");
    fprintf(file, "  nodesep=1.0;\n");
    fprintf(file, "  ranksep=1.5;\n");
    fprintf(file, "  bgcolor=transparent;\n");
    fprintf(file, "  fontcolor=white;\n");
    fprintf(file, "  node [shape=circle, fontsize=11, fixedsize=true, width=0.8, color=white, fontcolor=white];\n");
    fprintf(file, "  edge [arrowsize=0.7, color=white, penwidth=1.0, fontcolor=white];\n");

    // Create input nodes
    fprintf(file, "  subgraph cluster_input {\n");
    fprintf(file, "    style=invis;\n");
    fprintf(file, "    rank=same;\n");
    for (int i = 0; i < input_size; i++) {
        fprintf(file, "    input%d [label=\"%.4f\", fontcolor=white];\n", i, input_data[i]->data);
    }
    fprintf(file, "  }\n");

    // Loop over hidden layers in the MLP
    for (int i = 0; i < mlp->n_layers - 1; i++) {
        Layer* layer = mlp->layers[i];

        fprintf(file, "  subgraph cluster_layer%d {\n", i);
        fprintf(file, "    style=invis;\n");
        fprintf(file, "    rank=same;\n");

        // Loop over neurons in the layer
        for (int j = 0; j < layer->nout; j++) {
            Neuron* neuron = layer->neurons[j];
            fprintf(file, "    hidden%d_%d [label=\"%.4f\"];\n", i, j, neuron->bias->data);
        }

        fprintf(file, "  }\n");
    }

    // Create output node
    Layer* output_layer = mlp->layers[mlp->n_layers - 1];
    Neuron* output_neuron = output_layer->neurons[0];
    fprintf(file, "  output [label=\"%.4f\"];\n", output_neuron->bias->data);

    // Create prediction and loss nodes
    fprintf(file, "  subgraph cluster_output {\n");
    fprintf(file, "    style=invis;\n");
    fprintf(file, "    rank=same;\n");
    fprintf(file, "    prediction [label=\"%.4f\", shape=circle];\n", prediction->data);
    fprintf(file, "    loss [label=\"%.4f\", shape=circle];\n", loss->data);
    fprintf(file, "  }\n");

    // Connect input nodes to first hidden layer
    for (int i = 0; i < input_size; i++) {
        for (int j = 0; j < mlp->layers[0]->nout; j++) {
            fprintf(file, "  input%d -> hidden0_%d [label=\"%.4f\", fontsize=9];\n", i, j, mlp->layers[0]->neurons[j]->weights[i]->data);
        }
    }

    // Connect hidden layers
    for (int i = 0; i < mlp->n_layers - 2; i++) {
        Layer* curr_layer = mlp->layers[i];
        Layer* next_layer = mlp->layers[i + 1];

        for (int j = 0; j < curr_layer->nout; j++) {
            for (int k = 0; k < next_layer->nout; k++) {
                fprintf(file, "  hidden%d_%d -> hidden%d_%d [label=\"%.4f\", fontsize=9];\n", i, j, i + 1, k, next_layer->neurons[k]->weights[j]->data);
            }
        }
    }

    // Connect last hidden layer to output node
    Layer* last_hidden_layer = mlp->layers[mlp->n_layers - 2];
    for (int i = 0; i < last_hidden_layer->nout; i++) {
        fprintf(file, "  hidden%d_%d -> output;\n", mlp->n_layers - 2, i);
    }

    // Connect output node to prediction and loss nodes
    fprintf(file, "  output -> prediction;\n");
    fprintf(file, "  prediction -> loss;\n");

    fprintf(file, "}\n");
    fclose(file);
}

int main() {
    srand((unsigned int)time(NULL));
    // DATA
    // 4 data points, each containing an array of 3 Values
    int num_features = 3;
    int num_data_points = 4;
    Value ***data = malloc(num_data_points * sizeof(Value**));
    float predefined_values[4][3] = {
        {2, 3, -1},
        {3, -1, 0.5},
        {0.5, 1, 1},
        {1, 1, -1}
    };

    for (int i = 0; i < num_data_points; i++) {
        data[i] = malloc(num_features * sizeof(Value*));
        for (int j = 0; j < num_features; j++) {
            char *label = malloc(20 * sizeof(char));
            sprintf(label, "data[%d][%d]", i, j);
            data[i][j] = newValue(predefined_values[i][j], NULL, NULL, label, 0.0, false);
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
    int *nouts = malloc(3 * sizeof(int));
    nouts[0] = 4;
    nouts[1] = 3;
    nouts[2] = 1;

    MLP *mlp = createMLP(3, nouts, 3);
    // TRAINING LOOP
    int dataset_size = 4;
    int num_epochs = 50;
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        Value **epoch_losses = malloc(dataset_size * sizeof(Value*));
        for (int i = 0; i < dataset_size; i++) {
            // Forward pass
            Value **output = forwardMLP(mlp, data[i], 3);

            // Compute loss
            Value *target = newValue(targets[i], NULL, NULL, "target", 0.0, false);
            if (DEBUG == true) {
                displayValue(*output);
                displayValue(target);
            }
            Value *loss = squared_error_loss(*output, target);
            epoch_losses[i] = loss;

            // Backward pass
            loss->grad = 1.0;
            _backward(loss);
            
            // Update weights
            stepMLP(mlp);
            if (epoch == num_epochs-1) {
                displayValue(*output);
                displayValue(target);
                generate_dot(mlp, data[0], num_features, *output, loss);
            }

            // Zero gradients
            zeroGrad(mlp);
        }
        // Compute average loss for the epoch
        Value *avg_loss = average_losses(epoch_losses, dataset_size);
        printf("Epoch %d, Loss: %.4f\n", epoch, avg_loss->data);

        // Free memory
        for (int i = 0; i < dataset_size; i++) {
            free(epoch_losses[i]);
        }
        free(epoch_losses);
        free(avg_loss);
    }
    freeMLP(mlp);
    return 0;
}