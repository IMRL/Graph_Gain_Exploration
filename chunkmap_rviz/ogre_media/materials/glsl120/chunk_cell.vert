#version 120

attribute vec4 uv0;

varying vec4 Position;
varying vec2 out_texture_coordinate;

uniform sampler2D u_elevation;
uniform float u_elevation_alpha;
uniform float u_elevation_beta;

void main()
{
  out_texture_coordinate = vec2(uv0);
  Position = gl_Vertex;
  Position.z += texture2D(u_elevation, uv0.st).x * u_elevation_alpha + u_elevation_beta;
  gl_Position = gl_ModelViewProjectionMatrix * Position;
}
