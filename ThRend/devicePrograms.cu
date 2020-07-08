#include "owl/owl.h"
#include "OWLRenderer.h"

using namespace owl;
using namespace owl::common;

__constant__ DeviceGlobals optixLaunchParams;

inline __device__ Ray Camera::generateRay(const vec2f &screen)
{
  const vec3f dir
    = screen_00
    + screen.x * screen_du
    + screen.y * screen_dv;
  return Ray(origin,normalize(dir),1e-3f,1e10f);
}

/*! the per-ray data we use to communicate between closest hit program
    and raygen program */
struct IntersectionResult
{
  int   primID;
  /*! geometry normal */
  vec3f Ng;
};

OPTIX_CLOSEST_HIT_PROGRAM(TrianglesCH)()
{
  auto &isec = owl::getPRD<IntersectionResult>();
  auto &self = owl::getProgramData<TrianglesGeom>();

  isec.primID = optixGetPrimitiveIndex();
  vec3i index = self.triangles[isec.primID];
  const vec3f v0 = self.vertices[index.x];
  const vec3f v1 = self.vertices[index.y];
  const vec3f v2 = self.vertices[index.z];
  isec.Ng = normalize(cross(v1-v0,v2-v0));
}

OPTIX_CLOSEST_HIT_PROGRAM(QuadsCH)()
{
  auto &isec = owl::getPRD<IntersectionResult>();
  auto &self = owl::getProgramData<QuadsGeom>();

  int quadID = optixGetPrimitiveIndex() / 2;
  isec.primID = quadID;
  vec4i index = self.quads[isec.primID];
  const vec3f v0 = self.vertices[index.x];
  const vec3f v1 = self.vertices[index.y];
  const vec3f v2 = self.vertices[index.z];
  isec.Ng = normalize(cross(v1-v0,v2-v0));
}

OPTIX_MISS_PROGRAM(miss)()
{
}

inline __device__
vec3f missColor()
{
  const vec2i pixelID = owl::getLaunchIndex();
  const float t = pixelID.y / (float)optixGetLaunchDimensions().y;
  const vec3f c = (1.0f - t)*vec3f(1.0f, 1.0f, 1.0f) + t * vec3f(0.5f, 0.7f, 1.0f);
  return c;
}

OPTIX_RAYGEN_PROGRAM(deviceMain)()
{
  vec2i pixel = owl::getLaunchIndex();

  vec2f screenCoords = (vec2f(pixel)+vec2f(.5f)) / vec2f(owl::getLaunchDims());
  
  auto &lp = optixLaunchParams;
  Ray ray = lp.camera.generateRay(screenCoords);
  
  IntersectionResult isec;
  isec.primID = -1;
  owl::traceRay(lp.world,ray,isec);

  vec3f color
    = (isec.primID < 0)
    ? missColor()//vec3f(0.f)
    : (.3f+.7f*abs(dot(isec.Ng,ray.direction)))*owl::randomColor(isec.primID);
  const int fbIndex = pixel.x + lp.fb.size.x*pixel.y;
  lp.fb.pointer[fbIndex] = owl::make_rgba(vec4f(color,1.f));
}


