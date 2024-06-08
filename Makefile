CC = gcc
CFLAGS = -Wall -Wextra -std=c99

SRCS = test.c engine.c nn.c visualizer.c

OBJS = $(SRCS:.c=.o)

PROGRAM = micrograd

all: $(PROGRAM)

$(PROGRAM): $(OBJS)
	$(CC) $(CFLAGS) -o $(PROGRAM) $(OBJS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(PROGRAM) $(OBJS)