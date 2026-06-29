# Makefile

# Compiler and compiler flags
CXX = g++
CXXFLAGS = -Wall -fPIC -std=c++17 -O3
INC = -I$(SRCDIR)
LIBDIR = ${FOAM_USER_LIBBIN}
SRCDIR = ./src
BUILDDIR = ./build

# Name of the shared library
LIBNAME = libglobalModelParameter.so

# Source files for the shared library
SOURCES = \
	$(SRCDIR)/mlp.cpp \
	$(SRCDIR)/support_func.cpp \
	$(SRCDIR)/wsggMLPParameter.cpp \
	$(SRCDIR)/wsggmlp.cpp
OBJECTS = $(SOURCES:$(SRCDIR)/%.cpp=$(BUILDDIR)/%.o)
HEADERS = $(wildcard $(SRCDIR)/*.h)

# Rule to make the object files
$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp $(HEADERS)
	mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

# Rule to make the shared library
$(LIBNAME): $(OBJECTS)
	mkdir -p $(LIBDIR)
	$(CXX) -shared -o $(LIBDIR)/$@ $(OBJECTS)

# Phony targets
.PHONY: all clean

# Default rule
all: $(LIBNAME)

# Rule to clean files
clean:
	rm -f $(BUILDDIR)/*.o $(LIBDIR)/$(LIBNAME)
