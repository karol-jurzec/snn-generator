
CC = gcc  
CFLAGS = -Wall -Wextra -g -I$(INCLUDE_DIR) -I$(JSON_C_INCLUDE)

LDFLAGS = -L$(JSON_C_LIB)
LDLIBS = -ljson-c -lpthread -lm -lmatio -lpsapi -lopenblas

SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build

JSON_C_INCLUDE = /mingw64/include
JSON_C_LIB = /mingw64/lib

SRC_FILES = $(wildcard $(SRC_DIR)/**/*.c $(SRC_DIR)/*.c)
OBJ_FILES = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(SRC_FILES))

TARGET = snn_generator

all: $(TARGET)
release: CFLAGS += -O2 -DNDEBUG
release: all

$(TARGET): $(OBJ_FILES)
	@echo "Linking $@..."
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	@echo "Cleaning up..."
	rm -rf $(BUILD_DIR) $(TARGET)

run: all
	@echo "Running $(TARGET)..."
	./$(TARGET)

.PHONY: all clean run