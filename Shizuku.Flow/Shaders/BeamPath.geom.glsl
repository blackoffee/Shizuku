#version 430 core
layout (triangles_adjacency) in;
layout (triangle_strip, max_vertices = 24) out;

in vec2[] modelPos;

out vec4 fColor;

uniform bool Filter = true;

uniform vec2 Target = vec2(0,0);

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
		if (length(center - Target) < 0.05)
		{
			//sides
			fColor = vec4(0, 0, 1, 1);
			fColor = vec4(0, 0, 1, 1);
			fColor = vec4(0, 0, 1, 1);
			fColor = vec4(0, 0, 1, 1);
			fColor = vec4(0, 0, 1, 1);
			fColor = vec4(0, 0, 1, 1);
			fColor = vec4(0, 0, 1, 1);
			fColor = vec4(0, 0, 1, 1);
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
			fColor = vec4(0, 1, 0, 1);
			fColor = vec4(0, 1, 0, 1);
			fColor = vec4(0, 1, 0, 1);
			gl_Position = gl_in[0].gl_Position;
			EmitVertex();
			gl_Position = gl_in[1].gl_Position;
			EmitVertex();
			gl_Position = gl_in[2].gl_Position;
			EmitVertex();
			EndPrimitive();

			//bottom
			fColor = vec4(0, 1, 0, 1);
			fColor = vec4(0, 1, 0, 1);
			fColor = vec4(0, 1, 0, 1);
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