#include "owl/owl.h"
#include "OWLRenderer.h"

struct Globals {
};

__constant__ Globals optixLaunchParams;

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


