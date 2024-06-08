#include "engine.h"
#include "nn.h"

void generateDot(MLP* mlp, Value** inputData, size_t inputSize, Value* prediction, Value* loss, int graphN) {
    char filename[20];
    snprintf(filename, sizeof(filename), "graph%d.dot", graphN);
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }

    fprintf(file, "digraph G {\n");
    fprintf(file, " rankdir=LR;\n");
    fprintf(file, " nodesep=1.0;\n");
    fprintf(file, " ranksep=1.5;\n");
    fprintf(file, " bgcolor=transparent;\n");
    fprintf(file, " fontcolor=white;\n");
    fprintf(file, " node [shape=circle, fontsize=11, fixedsize=true, width=0.8, color=white, fontcolor=white];\n");
    fprintf(file, " edge [arrowsize=0.7, color=white, penwidth=1.0, fontcolor=white, labeldistance=2, labelangle=45];\n");

    // Create input nodes
    fprintf(file, " subgraph cluster_input {\n");
    fprintf(file, " style=invis;\n");
    fprintf(file, " rank=same;\n");
    for (size_t i = 0; i < inputSize; i++) {
        fprintf(file, " input%zu [label=\"%.4f\", xlabel=\"%s\", fontcolor=white];\n", i, inputData[i]->data, inputData[i]->label);
    }
    fprintf(file, " }\n");

    // Loop over hidden layers in the MLP
    for (size_t i = 0; i < mlp->nLayers - 1; i++) {
        Layer* layer = mlp->layers[i];
        fprintf(file, " subgraph cluster_layer%zu {\n", i);
        fprintf(file, " style=invis;\n");
        fprintf(file, " rank=same;\n");
        // Loop over neurons in the layer
        for (size_t j = 0; j < layer->nout; j++) {
            Neuron* neuron = layer->neurons[j];
            fprintf(file, " hidden%zu_%zu [label=\"%.4f\", xlabel=\"%s\"];\n", i, j, neuron->bias->data, neuron->bias->label);
        }
        fprintf(file, " }\n");
    }

    // Create output node
    Layer* output_layer = mlp->layers[mlp->nLayers - 1];
    Neuron* output_neuron = output_layer->neurons[0];
    fprintf(file, " output [label=\"%.4f\", xlabel=\"%s\"];\n", output_neuron->bias->data, output_neuron->bias->label);

    // Create prediction and loss nodes
    fprintf(file, " subgraph cluster_output {\n");
    fprintf(file, " style=invis;\n");
    fprintf(file, " rank=same;\n");
    fprintf(file, " prediction [label=\"%.4f\", xlabel=\"%s\", shape=circle];\n", prediction->data, prediction->label);
    fprintf(file, " loss [label=\"%.4f\", xlabel=\"%s\", shape=circle];\n", loss->data, loss->label);
    fprintf(file, " }\n");

    // Connect input nodes to first hidden layer
    for (size_t i = 0; i < inputSize; i++) {
        for (size_t j = 0; j < mlp->layers[0]->nout; j++) {
            fprintf(file, " input%zu -> hidden0_%zu [label=\"%s\\n%.4f\", fontsize=9];\n", i, j, mlp->layers[0]->neurons[j]->weights[i]->label, mlp->layers[0]->neurons[j]->weights[i]->data);
        }
    }

    // Connect hidden layers
    for (size_t i = 0; i < mlp->nLayers - 2; i++) {
        Layer* currLayer = mlp->layers[i];
        Layer* nextLayer = mlp->layers[i + 1];
        for (size_t j = 0; j < currLayer->nout; j++) {
            for (size_t k = 0; k < nextLayer->nout; k++) {
                fprintf(file, " hidden%zu_%zu -> hidden%zu_%zu [label=\"%s\\n%.4f\", fontsize=9];\n", i, j, i + 1, k, nextLayer->neurons[k]->weights[j]->label, nextLayer->neurons[k]->weights[j]->data);
            }
        }
    }

    // Connect last hidden layer to output node
    Layer* lastHiddenLayer = mlp->layers[mlp->nLayers - 2];
    for (size_t i = 0; i < lastHiddenLayer->nout; i++) {
        fprintf(file, " hidden%zu_%zu -> output;\n", mlp->nLayers - 2, i);
    }

    // Connect output node to prediction and loss nodes
    fprintf(file, " output -> prediction;\n");
    fprintf(file, " prediction -> loss;\n");

    fprintf(file, "}\n");
    fclose(file);
}