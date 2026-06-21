//=====================================================================
// Scalar fallback for the BC1 endpoint search on non-x86 targets.
//
// The optimized variants (core_simd_sse/avx/avx512.cpp) use x86 SSE/AVX/AVX-512
// intrinsics and x86-only -march flags, so they cannot be built on e.g. Apple
// Silicon (arm64). CMP_Compressonator (codec_dxtc_rgba.cpp) is compiled without
// ASPM_GPU and therefore references sse_/avx_/avx512_bc1ComputeBestEndpoints
// through bc1ToggleSIMD(). On non-x86 the runtime CPU detection never selects a
// SIMD path, but the symbols must still exist at link time — this file provides
// a single scalar implementation behind all three names.
//=====================================================================

#include <math.h>

#include "core_simd.h"
#include "common_def.h"

static CGU_FLOAT scalar_bc1ComputeBestEndpoints(CGU_FLOAT endpointsOut[2],
                                                CGU_FLOAT endpointsIn[2],
                                                CGU_FLOAT prj[16],
                                                CGU_FLOAT prjError[16],
                                                CGU_FLOAT preMRep[16],
                                                int       numColours,
                                                int       numPoints)
{
    static const CGU_FLOAT searchStep = 0.025f;

    const CGU_FLOAT lowStart  = (endpointsIn[0] - 2.0f * searchStep > 0.0f) ? endpointsIn[0] - 2.0f * searchStep : 0.0f;
    const CGU_FLOAT highStart = (endpointsIn[1] + 2.0f * searchStep < 1.0f) ? endpointsIn[1] + 2.0f * searchStep : 1.0f;

    CGU_FLOAT minError = 128000.0f;

    CGU_FLOAT lowStep  = lowStart;
    CGU_FLOAT highStep = highStart;

    for (int low = 0; low < 8; ++low)
    {
        for (int high = 0; high < 8; ++high)
        {
            const CGU_FLOAT stepScalar  = (highStep - lowStep) / (numPoints - 1);
            const CGU_FLOAT stepH        = 0.5f * stepScalar;
            const CGU_FLOAT inverseStep  = 1.0f / stepScalar;

            CGU_FLOAT finalError = 0.0f;
            for (int i = 0; i < numColours; ++i)
            {
                const CGU_FLOAT del       = prj[i] - lowStep;
                const CGU_FLOAT possibleV = floorf(inverseStep * (del + stepH)) * stepScalar + lowStep;

                // Mirrors the SSE blendv selection:
                //   v = (highStep > prj[i]) ? possibleV : highStep
                //   v = (del > 0)           ? v        : lowStep
                CGU_FLOAT v = (highStep > prj[i]) ? possibleV : highStep;
                v           = (del > 0.0f) ? v : lowStep;

                const CGU_FLOAT d = prj[i] - v;

                finalError += preMRep[i] * d * d + prjError[i];
                if (finalError >= minError)
                {
                    finalError = minError;
                    break;
                }
            }

            if (finalError < minError)
            {
                minError        = finalError;
                endpointsOut[0] = lowStep;
                endpointsOut[1] = highStep;
            }

            highStep -= searchStep;
        }

        lowStep += searchStep;
    }

    return minError;
}

CGU_FLOAT sse_bc1ComputeBestEndpoints(CGU_FLOAT endpointsOut[2],
                                      CGU_FLOAT endpointsIn[2],
                                      CGU_FLOAT prj[16],
                                      CGU_FLOAT prjError[16],
                                      CGU_FLOAT preMRep[16],
                                      int       numColours,
                                      int       numPoints)
{
    return scalar_bc1ComputeBestEndpoints(endpointsOut, endpointsIn, prj, prjError, preMRep, numColours, numPoints);
}

CGU_FLOAT avx_bc1ComputeBestEndpoints(CGU_FLOAT endpointsOut[2],
                                      CGU_FLOAT endpointsIn[2],
                                      CGU_FLOAT prj[16],
                                      CGU_FLOAT prjError[16],
                                      CGU_FLOAT preMRep[16],
                                      int       numColours,
                                      int       numPoints)
{
    return scalar_bc1ComputeBestEndpoints(endpointsOut, endpointsIn, prj, prjError, preMRep, numColours, numPoints);
}

CGU_FLOAT avx512_bc1ComputeBestEndpoints(CGU_FLOAT endpointsOut[2],
                                         CGU_FLOAT endpointsIn[2],
                                         CGU_FLOAT prj[16],
                                         CGU_FLOAT prjError[16],
                                         CGU_FLOAT preMRep[16],
                                         int       numColours,
                                         int       numPoints)
{
    return scalar_bc1ComputeBestEndpoints(endpointsOut, endpointsIn, prj, prjError, preMRep, numColours, numPoints);
}
