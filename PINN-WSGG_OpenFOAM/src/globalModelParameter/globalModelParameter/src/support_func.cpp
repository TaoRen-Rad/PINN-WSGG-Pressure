#include <cmath>
#include <string>
#include "support_func.h"
#include <dlfcn.h>

std::string GetLibraryPath() {
    Dl_info dl_info;
    dladdr((void*)&GetLibraryPath, &dl_info);
    std::string fullPath(dl_info.dli_fname);

    size_t pos = fullPath.find_last_of('/');
    if (pos != std::string::npos) {
        return fullPath.substr(0, pos);
    } else {
        return fullPath;
    }
}

void quadgen2(int NoP, double alpha, double* g[], double* w[])
{
	void gaucheb2(int n, double * x[], double * w[]);
	*g = (double*)malloc(NoP * sizeof(double));
	*w = (double*)malloc(NoP * sizeof(double));
	double* gg, * ww;
	gaucheb2(2 * NoP, &gg, &ww);
	double dummy = 0;
	if ((*g != NULL) && (*w != NULL))
	{
		for (int i = 0; i < NoP; i++)
		{
			*((*g) + i) = -gg[i + NoP];
			*((*w) + i) = ww[i + NoP];
			*((*w) + i) = alpha * (*((*w) + i)) * pow((1 - (*((*g) + i))), alpha - 1);
			*((*g) + i) = 1 - pow((1 - (*((*g) + i))), alpha);
			dummy = dummy + (*((*w) + i));
		}
		for (int i = 0; i < NoP; i++)
			*((*w) + i) = *((*w) + i) / dummy;
	}
}

void gaucheb2(int n, double* x[], double* w[])
{
	const double pi = acos(-1.0);
	*x = (double*)malloc(n * sizeof(double));
	*w = (double*)malloc(n * sizeof(double));
	if ((*x != NULL) && (*w != NULL))
	{
		for (int k = 0; k < (int)((n + 1) / 2); k++)
		{
			double theta = ((double)k + 1.0) * pi / ((double)n + 1);
			*((*x) + k) = cos(theta);
			double sum = 0;
			for (int m = 0; m < (int)((n + 1) / 2); m++)
				sum = sum + sin((2 * ((double)m + 1.0) - 1) * theta) / (2 * ((double)m + 1.0) - 1);
			*((*w) + k) = 4 * sin(theta) * sum / ((double)n + 1.0);
			*((*x) + n - 1 - k) = -*((*x) + k);
			*((*w) + n - 1 - k) = *((*w) + k);
		}
	}
}

void afun(double g[], double k[], double kll[], double w[], double an[])
{
	double gb[NqDB + 1], ge[NqDB + 1], gd[NqDB];
	gb[0] = 0.0; ge[0] = 0.0;
	for (int i = 0; i < NqDB; i++)
		gb[i + 1] = gb[i] + w[i];
	for (int i = 0; i < NqDB; i++)
		gd[i] = gb[i + 1];
	void linearInterpMono(int nxy, int ni, double xx[], double yy[], double xi[], double yi[]);
	double kq[NqDB], gq[NqDB];
	linearInterpMono(NqDB, NqDB, g, k, gd, kq);
	linearInterpMono(NqDB, NqDB, kll, g, kq, gq);
	for (int i = 0; i < NqDB; i++)
		ge[i + 1] = gq[i];
	for (int i = 0; i < NqDB; i++)
		an[i] = (ge[i + 1] - ge[i]) / w[i];
}

void linearInterpMono(int nxy, int ni, double xx[], double yy[], double xi[], double yi[])
{
	int n = 0;
	for (int i = 0; i < ni; i++)
	{
		int m = 0;
		while (xi[i] >= xx[n])
		{
			n++;
			if (n > (nxy - 1))
			{
				for (int f = i; f < ni; f++)
					yi[f] = yy[nxy - 1];
				m = 1;
				break;
			}
		}
		if (m == 1)
			break;
		if (n == 0)
			yi[i] = yy[0];
		else
			yi[i] = yy[n] + (yy[n - 1] - yy[n]) * (xi[i] - xx[n]) / (xx[n - 1] - xx[n]);
	}
}

void simple_interp(int nxy, int ni, double xx[], double yy[], double xi[], double yi[])
{
	double dx;
	int ibgn = 0;
	for (int iq = 0; iq < ni; iq++)
	{
		if (xi[iq] < xx[0])
			yi[iq] = yy[0];
		else if (xi[iq] >= xx[nxy - 1])
		{
			for (int f = iq; f < ni; f++)
				yi[f] = yy[nxy - 1];
			break;
		}
		for (int i = ibgn; i < nxy; i++)
		{
			if (xx[i] > xi[iq])
			{
				dx = xx[i] - xx[i - 1];
				if (fabs(dx) < 2e-203)
					yi[iq] = yy[i - 1];
				else
				{
					yi[iq] = yy[i - 1] + (yy[i] - yy[i - 1]) * (xi[iq] - xx[i - 1]) / dx;
					ibgn = i;
					break;
				}
			}
		}
	}
}
