#ifndef QUAD_H
#define QUAD_H

#define NqDB 32

#include <string>

std::string GetLibraryPath();
void quadgen2(int NoP, double alpha, double* g[], double* w[]);
void gaucheb2(int n, double* x[], double* w[]);
void afun(double g[], double k[], double kll[], double w[], double an[]);
void linearInterpMono(int nxy, int ni, double xx[], double yy[], double xi[], double yi[]);
void simple_interp(int nxy, int ni, double xx[], double yy[], double xi[], double yi[]);

#endif
