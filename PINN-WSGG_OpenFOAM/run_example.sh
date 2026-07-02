#!/bin/sh

set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"

if [ -z "${WM_OPTIONS:-}" ] || [ -z "${FOAM_APPBIN:-}" ]; then
    if [ -f /opt/OpenFOAM/OpenFOAM-v2412/etc/bashrc ]; then
        # shellcheck disable=SC1091
        . /opt/OpenFOAM/OpenFOAM-v2412/etc/bashrc
    else
        echo "OpenFOAM environment is not loaded and /opt/OpenFOAM/OpenFOAM-v2412/etc/bashrc was not found." >&2
        exit 1
    fi
fi

LOCAL_LIBBIN="${ROOT_DIR}/platforms/${WM_OPTIONS}/lib"

if [ ! -f "${LOCAL_LIBBIN}/libglobalModelParameter.so" ] || [ ! -f "${LOCAL_LIBBIN}/libglobalModels.so" ]; then
    echo "Local PINN-WSGG libraries were not found in ${LOCAL_LIBBIN}." >&2
    echo "Build them first with: ./src/Allwmake" >&2
    exit 1
fi

export PINN_WSGG_ROOT="${ROOT_DIR}"
export FOAM_USER_LIBBIN="${LOCAL_LIBBIN}"
export LD_LIBRARY_PATH="${LOCAL_LIBBIN}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

exec thermoFoam -case "${ROOT_DIR}/ModestSandiaD" "$@"
