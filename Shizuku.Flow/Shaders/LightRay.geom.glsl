#version 430 core
layout (lines) in;
layout (line_strip, max_vertices = 2) out;

in vec2[] modelPos;

uniform bool Filter = true;

uniform vec2 Target = vec2(0,0);

void main()
{
    if (!Filter) {
        gl_Position = gl_in[0].gl_Position;
        EmitVertex();

        gl_Position = gl_in[1].gl_Position;
        EmitVertex();
    }
    else
    {
        if (length(modelPos[1] - Target) < 0.05)
        {
            gl_Position = gl_in[0].gl_Position;
            EmitVertex();

            gl_Position = gl_in[1].gl_Position;
            EmitVertex();
        }
    }
}