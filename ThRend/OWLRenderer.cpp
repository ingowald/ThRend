#include "OWLRenderer.h"
#include <owl/owl.h>

extern "C" char deviceCode[];

OWLRenderer::OWLRenderer(const Model &model)
{
  PING;
  context = owlContextCreate(nullptr,1);
  PING;
  module = owlModuleCreate(context,deviceCode);
  
  OWLGeom trianglesGeom = createTrianglesGeom(model);
  OWLGeom quadsGeom = createQuadsGeom(model);

  OWLGeom geoms[2] = { trianglesGeom, quadsGeom };
  OWLGroup meshGroup = owlTrianglesGeomGroupCreate(context,2,geoms);
  owlGroupBuildAccel(meshGroup);
  world = owlInstanceGroupCreate(context,1,
                                 &meshGroup);
}



OWLGeom OWLRenderer::createTrianglesGeom(const Model &model)
{
  PING;
  OWLVarDecl vars[]
    = {
       { "triangles", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeom,triangles) },
       { "vertices", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeom,vertices) },
       { /* sentinel */ nullptr }
  };
  OWLGeomType geomType
    = owlGeomTypeCreate(context,OWL_TRIANGLES,sizeof(TrianglesGeom),
                        vars,-1);
  OWLGeom geom
    = owlGeomCreate(context,geomType);

  OWLBuffer vertices
    = owlDeviceBufferCreate(context,OWL_FLOAT3,model.vertices.size(),model.vertices.data());
  OWLBuffer triangles
    = owlDeviceBufferCreate(context,OWL_INT3,model.triangles.size(),model.triangles.data());
  
  owlTrianglesSetVertices(geom,vertices,model.vertices.size(),sizeof(owl::vec3f),0);
  owlTrianglesSetIndices(geom,triangles,model.triangles.size(),sizeof(owl::vec3i),0);
  
  owlGeomSetBuffer(geom,"vertices",vertices);
  owlGeomSetBuffer(geom,"triangles",triangles);
  
  return geom;
}

OWLGeom OWLRenderer::createQuadsGeom(const Model &model)
{
  PING;
  OWLVarDecl vars[]
    = {
       { "quads", OWL_BUFPTR, OWL_OFFSETOF(QuadsGeom,quads) },
       { "vertices", OWL_BUFPTR, OWL_OFFSETOF(QuadsGeom,vertices) },
       { /* sentinel */ nullptr }
  };
  OWLGeomType geomType
    = owlGeomTypeCreate(context,OWL_TRIANGLES,sizeof(QuadsGeom),
                        vars,-1);
  OWLGeom geom
    = owlGeomCreate(context,geomType);

  std::vector<owl::vec3i> splitQuads;
  for (int i=0;i<model.quads.size();i++) {
    owl::vec4i quad = model.quads[i];
    splitQuads.push_back({quad.x,quad.y,quad.z});
    splitQuads.push_back({quad.x,quad.z,quad.w});
  }
  
  OWLBuffer vertices
    = owlDeviceBufferCreate(context,OWL_FLOAT3,model.vertices.size(),model.vertices.data());
  OWLBuffer quads
    = owlDeviceBufferCreate(context,OWL_INT4,model.quads.size(),model.quads.data());
  OWLBuffer triangles
    = owlDeviceBufferCreate(context,OWL_INT3,splitQuads.size(),splitQuads.data());
  
  owlTrianglesSetVertices(geom,vertices,model.quads.size(),sizeof(owl::vec3f),0);
  owlTrianglesSetIndices(geom,triangles,splitQuads.size(),sizeof(owl::vec3i),0);
  
  owlGeomSetBuffer(geom,"vertices",vertices);
  owlGeomSetBuffer(geom,"quads",quads);
  
  return geom;
}
