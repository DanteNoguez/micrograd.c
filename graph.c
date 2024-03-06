#include <stdio.h>
#include <stdlib.h>

// Define a struct for a Node in the graph
typedef struct Node {
    char* label;  // Label of the node
    float value;  // Value of the node (for simplicity, all nodes have a float value)
    struct Node** prev; // An array of pointers to predecessor nodes
    int prev_count; // Number of predecessor nodes
    char* op;  // Operation performed at this node, if applicable
} Node;

// Function to create a new node
Node* create_node(char* label, float value, char* op, Node** prev, int prev_count) {
    Node* newNode = (Node*)malloc(sizeof(Node));
    newNode->label = label;
    newNode->value = value;
    newNode->op = op;
    newNode->prev = prev;
    newNode->prev_count = prev_count;
    return newNode;
}

void generate_dot(Node** nodes, int node_count) {
    FILE* file = fopen("graph.dot", "w");
    if (file == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }
  
    fprintf(file, "digraph G {\n");
    fprintf(file, "node [shape=record];\n");
  
    // Loop over nodes to create their representations
    for (int i = 0; i < node_count; i++) {
        Node* n = nodes[i];
        fprintf(file, "\"%p\" [label=\"{%s|value %.4f}\"];\n", (void*)n, n->label, n->value);
        if (n->op) {
            fprintf(file, "\"%p%s\" [label=\"%s\"];\n", (void*)n, n->op, n->op);
            fprintf(file, "\"%p%s\" -> \"%p\";\n", (void*)n, n->op, (void*)n);
        }
        for (int j = 0; j < n->prev_count; j++) {
            fprintf(file, "\"%p\" -> \"%p%s\";\n", (void*)n->prev[j], (void*)n, n->op);
        }
    }
  
    fprintf(file, "}\n");
    fclose(file);
}

int main() {
    // Manually construct a simple graph
    Node* n1 = create_node("n1", 1.0, NULL, NULL, 0);
    Node* n2 = create_node("n2", 2.0, NULL, NULL, 0);
    
    Node* prev[] = {n1, n2};
    Node* n3 = create_node("n3", 3.0, "add", prev, 2);
    
    Node* nodes[] = {n1, n2, n3};
    int node_count = 3;
    
    generate_dot(nodes, node_count);
    
    // Remember to free allocated memory here in a real application
    return 0;
}

// dot -Tsvg graph.dot -o graph.svg