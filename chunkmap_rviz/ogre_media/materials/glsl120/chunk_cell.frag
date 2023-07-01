#version 120

varying vec2 out_texture_coordinate;

uniform sampler2D u_occupancy;

void main()
{
  gl_FragColor = texture2D(u_occupancy, out_texture_coordinate);
  if (gl_FragColor.a == 0)
  {
    discard;
  }
  if (gl_FragColor.r <= 0.498)
  {
    gl_FragColor.rgb = vec3(0, 0, 0);
  }
  if (gl_FragColor.r >= 0.51)
  {
    gl_FragColor.rgb = vec3(1, 1, 1);
  }
}
