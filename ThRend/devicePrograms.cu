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

inline __device__
void writeAccumulate(const vec3f &pixelColor)
{
  auto &lp = optixLaunchParams;
  const vec2i pixel  = owl::getLaunchIndex();
  const int fbIndex  = pixel.x + lp.fb.size.x*pixel.y;
  float4 *accumBufferValue = ((float4 *)lp.accumBuffer) + fbIndex;
  const int accumID = lp.accumID;
  vec4f pixel4f = vec4f(pixelColor,1.f);
  if (accumID > 0)
    pixel4f += vec4f(*accumBufferValue);
  
  *accumBufferValue = pixel4f;
  lp.fb.pointer[fbIndex] = make_rgba(pixel4f / pixel4f.w);
}

OPTIX_RAYGEN_PROGRAM(deviceMain)()
{
  auto &lp = optixLaunchParams;
  
  vec2i pixel = owl::getLaunchIndex();
  if ((pixel.y % lp.multiGPU.deviceCount) != lp.multiGPU.deviceIndex) return;
  
  vec2f screenCoords = (vec2f(pixel)+vec2f(.5f)) / vec2f(owl::getLaunchDims());
  
  Ray ray = lp.camera.generateRay(screenCoords);
  
  IntersectionResult isec;
  isec.primID = -1;
  owl::traceRay(lp.world,ray,isec);

  vec3f color
    = (isec.primID < 0)
    ? missColor()//vec3f(0.f)
    : (.3f+.7f*abs(dot(isec.Ng,ray.direction)))*owl::randomColor(isec.primID);

  writeAccumulate(color);
}


