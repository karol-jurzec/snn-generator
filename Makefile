#OPENBLAS_INCLUDE = /ucrt64/include/openblas
#OPENBLAS_LIB = /ucrt64/lib

OPENBLAS_INCLUDE = /mingw64/include/openblas
OPENBLAS_LIB = /mingw64/lib

#HDF5_INCLUDE = /c/Program\ Files/HDF_Group/HDF5/1.14.6/include
#HDF5_LIB = /c/Program\ Files/HDF_Group/HDF5/1.14.6/lib

# Compiler and Flags
CC = gcc  
CFLAGS = -Wall -Wextra -g -I$(INCLUDE_DIR) -I$(JSON_C_INCLUDE) #-I$(HDF5_INCLUDE) 

LDFLAGS = -L$(JSON_C_LIB) # -L$(HDF5_LIB)
LDLIBS = -ljson-c -lpthread -lm -lopenblas -lmatio -lpsapi#-lhdf5

# Directories
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build
#JSON_C_PREFIX = $(shell brew --prefix json-c)
JSON_C_INCLUDE = /mingw64/include
JSON_C_LIB = /mingw64/lib

#JSON_C_INCLUDE = $(JSON_C_PREFIX)/include
#JSON_C_LIB = $(JSON_C_PREFIX)/lib

# Source Files and Object Files
SRC_FILES = $(wildcard $(SRC_DIR)/**/*.c $(SRC_DIR)/*.c)
OBJ_FILES = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(SRC_FILES))

# Target Executable
TARGET = snn_generator

# Default Target
all: $(TARGET)
release: CFLAGS += -O2 -DNDEBUG
release: all

# Build Target
$(TARGET): $(OBJ_FILES)
	@echo "Linking $@..."
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

# Compile Object Files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) -c $< -o $@

# Clean Build Files
clean:
	@echo "Cleaning up..."
	rm -rf $(BUILD_DIR) $(TARGET)

# Run the Program
run: all
	@echo "Running $(TARGET)..."
	./$(TARGET)

# Phony Targets
.PHONY: all clean run