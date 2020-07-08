#pragma once

#define _USE_MATH_DEFINES
#define GLM_ENABLE_EXPERIMENTAL

#include <glm.hpp>
#include <vector>
#include <string>
#include "owl/common/math/vec.h"
               
struct Model
{
  std::vector<glm::vec3>  vertices;
  std::vector<owl::vec3i> triangles;
  std::vector<owl::vec4i> quads;
  std::vector<int>        matIDs;
  std::vector<float>      temps;
};

void loadUCD(Model &mode, const std::string &fileName);

// OWLGroup mapToOWL(const Model &model);

