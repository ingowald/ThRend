#include "owl/owl.h"
#include "OWLRenderer.h"

using namespace owl::common;

__constant__ DeviceGlobals optixLaunchParams;

/*! the per-ray data we use to communicate between closest hit program
    and raygen program */
struct IntersectionResult {
  int   primID;
};

OPTIX_CLOSEST_HIT_PROGRAM(TrianglesCH)()
{
  auto &isec = owl::getPRD<IntersectionResult>();
  auto &self = owl::getProgramData<TrianglesGeom>();
  
  isec.primID = optixGetPrimitiveIndex();
}

OPTIX_CLOSEST_HIT_PROGRAM(QuadsCH)()
{
  auto &isec = owl::getPRD<IntersectionResult>();
  auto &self = owl::getProgramData<QuadsGeom>();
  
  isec.primID = optixGetPrimitiveIndex();
}

OPTIX_MISS_PROGRAM(miss)()
{
}



OPTIX_RAYGEN_PROGRAM(deviceMain)()
{
  vec2i launchIndex = owl::getLaunchIndex();
  if (launchIndex == vec2i(0)) printf("launch!\n");
}


