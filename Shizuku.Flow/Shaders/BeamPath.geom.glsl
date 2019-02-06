#version 430 core
layout (triangles_adjacency) in;
layout (triangle_strip, max_vertices = 24) out;

in vec2[] modelPos;

out vec4 fColor;

uniform bool Filter = true;

uniform vec2 Target = vec2(0,0);
uniform float ProbeRadius = 0.07;

uniform vec4 SideColor = vec4(0, 0.8, 0, 1);
uniform vec4 TopColor = vec4(0, 1, 0, 1);

void main()
{
	if (!Filter) {
		gl_Position = gl_in[0].gl_Position;
		EmitVertex();

		gl_Position = gl_in[1].gl_Position;
		EmitVertex();

		gl_Position = gl_in[2].gl_Position;
		EmitVertex();
	}
	else
	{
		vec2 center = (modelPos[3] + modelPos[4] + modelPos[5]) / 3.f;
		int i = 0;
		if (length(center - Target) < ProbeRadius)
		{
			//sides
			fColor = SideColor;
			gl_Position = gl_in[0].gl_Position;
			EmitVertex();
			gl_Position = gl_in[2].gl_Position;
			EmitVertex();
			gl_Position = gl_in[3].gl_Position;
			EmitVertex();
			gl_Position = gl_in[5].gl_Position;
			EmitVertex();
			gl_Position = gl_in[1].gl_Position;
			EmitVertex();
			gl_Position = gl_in[4].gl_Position;
			EmitVertex();
			gl_Position = gl_in[0].gl_Position;
			EmitVertex();
			gl_Position = gl_in[3].gl_Position;
			EmitVertex();
			EndPrimitive();
			//top
			fColor = TopColor;
			gl_Position = gl_in[0].gl_Position;
			EmitVertex();
			gl_Position = gl_in[1].gl_Position;
			EmitVertex();
			gl_Position = gl_in[2].gl_Position;
			EmitVertex();
			EndPrimitive();

			//bottom
			fColor = TopColor;
			gl_Position = gl_in[3].gl_Position;
			EmitVertex();
			gl_Position = gl_in[4].gl_Position;
			EmitVertex();
			gl_Position = gl_in[5].gl_Position;
			EmitVertex();
			EndPrimitive();
		}
	}
}