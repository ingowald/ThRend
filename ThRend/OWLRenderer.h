#pragma once

#include "Model.h"
#include <owl/owl.h>
#include <owl/common/math/vec.h>

/*! device-side struct for the mesh of trianlges - owl builds this on the device */
struct TrianglesGeom {
  owl::vec3f *vertices;
  owl::vec3i *triangles;
};

/*! device-side struct for the mesh of trianlges - owl builds this on the device */
struct QuadsGeom {
  owl::vec3f *vertices;
  owl::vec3i *quads;
};

struct DeviceGlobals {
  struct {
    owl::vec2i size;
    uint32_t *rgba8;
  } fb;
  OptixTraversableHandle world;
  // int accumID;
};

struct OWLRenderer {
  OWLRenderer(const Model &model);

  void render();
  void resize(const owl::vec2i &fbSize,
              uint32_t *fbPointer);
private:
  
  OWLGeom createTrianglesGeom(const Model &model);
  OWLGeom createQuadsGeom(const Model &model);
  void createWorld(const Model &model);
  void createRayGen();
  void createMissProg();
  void createDeviceGlobals();

  int accum { 0 };
  owl::vec2i fbSize;
  
  OWLContext context { 0 };
  /*! the ptx module that contains all out device programs */
  OWLModule  module  { 0 };
  OWLParams  globals { 0 };
  OWLRayGen  rayGen  { 0 };
  OWLGroup   world   { 0 };
  OWLMissProg missProg { 0 };
};


