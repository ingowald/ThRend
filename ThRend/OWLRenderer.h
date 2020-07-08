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

struct OWLRenderer {
  OWLRenderer(const Model &model);

  OWLGeom createTrianglesGeom(const Model &model);
  OWLGeom createQuadsGeom(const Model &model);
  
  OWLContext context { 0 };
  /*! the ptx module that contains all out device programs */
  OWLModule  module { 0 };
  OWLGroup world { 0 };
};


