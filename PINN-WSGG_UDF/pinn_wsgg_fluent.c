#include "udf.h"
#include "math.h"

#define WSGG_INCLUDE_SG_PDF 0

#if WSGG_INCLUDE_SG_PDF
#include "sg_pdf.h"
#endif

#include "pdf_props.h"

#include "pinn_wsgg_data.h"

#define Nq 9
#define ifCO 1
#define WSGG_USE_PDF_YI 1

int ico2 = -1, ih2o = -1, ico = -1;
real mwco2 = 44.00950, mwh2o = 18.01528, mwco = 28.01010, p_op = 101325.0;
static const real wNq[Nq] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};


static real positive(real x, real xmin)
{
    return (x > xmin) ? x : xmin;
}


static real relu(real x)
{
    return (x > 0.0) ? x : 0.0;
}


static int normalize_gray_band_index(int nb)
{
    if (nb >= 0 && nb < Nq)
    {
        return nb;
    }

    if (nb >= 1 && nb <= Nq)
    {
        return nb - 1;
    }

    return -1;
}


static int gas_parameters_ready(void)
{
    if (ico2 < 0 || ih2o < 0)
    {
        return 0;
    }

    if (ifCO == 1 && ico < 0)
    {
        return 0;
    }

    return 1;
}


static real get_species_mass_fraction(cell_t c, Thread *t, int i)
{
    if (i < 0)
    {
        return 0.0;
    }

#if WSGG_USE_PDF_YI
    return positive(Pdf_Yi(c, t, i), 0.0);
#else
    return positive(C_YI(c, t, i), 0.0);
#endif
}


static void init_gas_parameters(Domain *d)
{
    Thread *t;
    int i;

    thread_loop_c(t, d)
    {
        if (FLUID_THREAD_P(t))
        {
            Material *mix = THREAD_MATERIAL(t), *sp;

            ico2 = mixture_specie_index(mix, "co2");
            ih2o = mixture_specie_index(mix, "h2o");
            if (ifCO == 1)
            {
                ico = mixture_specie_index(mix, "co");
            }

            mixture_species_loop(mix, sp, i)
            {
                if (i == ico2)
                {
                    mwco2 = MATERIAL_PROP(sp, PROP_mwi);
                }
                else if (i == ih2o)
                {
                    mwh2o = MATERIAL_PROP(sp, PROP_mwi);
                }
                else if (ifCO == 1 && i == ico)
                {
                    mwco = MATERIAL_PROP(sp, PROP_mwi);
                }
            }

            break;
        }
    }

    p_op = RP_Get_Real("operating-pressure");

    if (!gas_parameters_ready())
    {
        Message0
        (
            "\nPINN-WSGG error: failed to locate required species."
            " ico2=%d, ih2o=%d, ico=%d\n",
            ico2,
            ih2o,
            ico
        );
    }
}


static real get_mixmf(cell_t c, Thread *t)
{
    int i;
    real mf_mix = 0.0;
    Material *mix = THREAD_MATERIAL(t), *sp;

    mixture_species_loop(mix, sp, i)
    {
        mf_mix += get_species_mass_fraction(c, t, i) / MATERIAL_PROP(sp, PROP_mwi);
    }

    return positive(mf_mix, 1.0e-12);
}


static void forward_layer_0(const real in[WSGG_INPUT_SIZE], real out[WSGG_H1_SIZE])
{
    int i, j;

    for (i = 0; i < WSGG_H1_SIZE; i++)
    {
        real z = (real)wsgg_b0[i];
        for (j = 0; j < WSGG_INPUT_SIZE; j++)
        {
            z += in[j] * (real)wsgg_w0[i][j];
        }
        out[i] = relu(z);
    }
}


static void forward_layer_2(const real in[WSGG_H1_SIZE], real out[WSGG_H2_SIZE])
{
    int i, j;

    for (i = 0; i < WSGG_H2_SIZE; i++)
    {
        real z = (real)wsgg_b2[i];
        for (j = 0; j < WSGG_H1_SIZE; j++)
        {
            z += in[j] * (real)wsgg_w2[i][j];
        }
        out[i] = relu(z);
    }
}


static void forward_layer_4(const real in[WSGG_H2_SIZE], real out[WSGG_OUTPUT_SIZE])
{
    int i, j;

    for (i = 0; i < WSGG_OUTPUT_SIZE; i++)
    {
        real z = (real)wsgg_b4[i];
        for (j = 0; j < WSGG_H2_SIZE; j++)
        {
            z += in[j] * (real)wsgg_w4[i][j];
        }
        out[i] = z;
    }
}


static void get_prediction
(
    real temperature,
    real x_h2o,
    real x_co2,
    real x_co,
    real p_bar,
    real *weights,
    real *abscs
)
{
    int i;
    real in[WSGG_INPUT_SIZE];
    real h1[WSGG_H1_SIZE];
    real h2[WSGG_H2_SIZE];
    real raw[WSGG_OUTPUT_SIZE];
    real max_w, sum_w;

    in[0] = (temperature - (real)wsgg_x_mean[0]) / (real)wsgg_x_std[0];
    in[1] = (x_h2o - (real)wsgg_x_mean[1]) / (real)wsgg_x_std[1];
    in[2] = (x_co2 - (real)wsgg_x_mean[2]) / (real)wsgg_x_std[2];
    in[3] = (x_co - (real)wsgg_x_mean[3]) / (real)wsgg_x_std[3];
    in[4] = ((real)log((double)positive(p_bar, 1.0e-8)) - (real)wsgg_x_mean[4]) / (real)wsgg_x_std[4];

    forward_layer_0(in, h1);
    forward_layer_2(h1, h2);
    forward_layer_4(h2, raw);

    max_w = raw[0];
    for (i = 1; i < Nq; i++)
    {
        if (raw[i] > max_w)
        {
            max_w = raw[i];
        }
    }

    sum_w = 0.0;
    for (i = 0; i < Nq; i++)
    {
        weights[i] = (real)exp((double)(raw[i] - max_w));
        sum_w += weights[i];
    }

    sum_w = positive(sum_w, 1.0e-30);
    for (i = 0; i < Nq; i++)
    {
        weights[i] /= sum_w;
    }

    abscs[0] = 0.0;
    for (i = 1; i < Nq; i++)
    {
        abscs[i] = (real)exp((double)raw[8 + i]);
    }
}


DEFINE_EXECUTE_ON_LOADING(preFin, libname)
{
    Domain *d;

    d = Get_Domain(1);
    init_gas_parameters(d);
    Message0("\nPINN-WSGG preFin loaded.\n");
}


DEFINE_ADJUST(refTem, d)
{
    cell_t c;
    Thread *t;

    if (!gas_parameters_ready())
    {
        init_gas_parameters(d);
    }

    if (!gas_parameters_ready())
    {
        Message0("\nPINN-WSGG error: required species are unavailable; skip refTem.\n");
        return;
    }

    p_op = RP_Get_Real("operating-pressure");

    thread_loop_c(t, d)
    {
        if (FLUID_THREAD_P(t))
        {
            begin_c_loop_int(c, t)
            {
                int i;
                real mf_mix, xco2, xh2o, xco, xsum, p_bar;
                real weights[Nq], kappa_raw[Nq];
                real gray_abs = 0.0;

                mf_mix = get_mixmf(c, t);
                xco2 = get_species_mass_fraction(c, t, ico2) / (mwco2 * mf_mix);
                xh2o = get_species_mass_fraction(c, t, ih2o) / (mwh2o * mf_mix);
                if (ifCO == 1)
                {
                    xco = get_species_mass_fraction(c, t, ico) / (mwco * mf_mix);
                }
                else
                {
                    xco = 0.0;
                }

                xsum = xh2o + xco2 + xco;
                p_bar = positive((C_P(c, t) + p_op) / 1.0e5, 1.0e-8);

                get_prediction(C_T(c, t), xh2o, xco2, xco, p_bar, weights, kappa_raw);

                for (i = 0; i < Nq; i++)
                {
                    real band_k = kappa_raw[i] * xsum * p_bar * 100.0;

                    C_UDMI(c, t, 2 * i) = band_k;
                    C_UDMI(c, t, 2 * i + 1) = weights[i];
                    gray_abs += band_k * weights[i];
                }

                C_UDMI(c, t, 2 * Nq) = gray_abs;
            }
            end_c_loop_int(c, t)
        }
    }
}


DEFINE_GRAY_BAND_ABS_COEFF(user_gray_band_abs, c, t, nb)
{
    int band = normalize_gray_band_index(nb);
    real abs_coeff = 0.0;

    if (band < 0)
    {
        return 0.0;
    }

    abs_coeff = C_UDMI(c, t, 2 * band);
    return positive(abs_coeff, 0.0);
}


DEFINE_EMISSIVITY_WEIGHTING_FACTOR(user_gray_band_str, c, t, T, nb, emissivity_weighting_factor)
{
    int band = normalize_gray_band_index(nb);

    *emissivity_weighting_factor = 0.0;

    if (band < 0)
    {
        return;
    }

    *emissivity_weighting_factor =
        positive(C_UDMI(c, t, 2 * band + 1) * wNq[band], 0.0);
}


DEFINE_EXECUTE_AT_END(calculation)
{
    Domain *d;
    Thread *t;
    cell_t c;

    d = Get_Domain(1);
    thread_loop_c(t, d)
    {
        if (FLUID_THREAD_P(t))
        {
            begin_c_loop_int(c, t)
            {
                C_UDMI(c, t, 2 * Nq) = positive(C_UDMI(c, t, 2 * Nq), 0.0);
            }
            end_c_loop_int(c, t)
        }
    }
}


DEFINE_ON_DEMAND(print_pinn_wsgg_config)
{
    Message0("\nPINN-WSGG configuration\n");
    Message0("Nq = %d\n", Nq);
    Message0("WSGG_USE_PDF_YI = %d\n", WSGG_USE_PDF_YI);
    Message0("WSGG_INCLUDE_SG_PDF = %d\n", WSGG_INCLUDE_SG_PDF);
    Message0("ico2 = %d, ih2o = %d, ico = %d\n", ico2, ih2o, ico);
    Message0("mwco2 = %g, mwh2o = %g, mwco = %g\n", mwco2, mwh2o, mwco);
    Message0("UDMI layout: k_i -> 2*i, a_i -> 2*i+1, gray -> %d\n\n", 2 * Nq);
}
