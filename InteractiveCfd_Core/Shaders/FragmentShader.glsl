#version 430 core
in vec4 fColor;
in vec3 texCoords;

out vec4 color;

uniform sampler2D renderedTexture;


void main()
{

    color = vec4(fColor);

//    if (texCoords.z > 0.2f)
//    {
//        color.r = texture( renderedTexture, vec2( texCoords.x, texCoords.y) ).r;
//        color.g = texture( renderedTexture, vec2( texCoords.x, texCoords.y) ).g;
//        color.b = texture( renderedTexture, vec2( texCoords.x, texCoords.y) ).b;
//        //color.r = texture( renderedTexture, vec2( 0.5f, 0.1f )).r;
//        //color.g = texture( renderedTexture, vec2( 0.5f, 0.1f )).g;
//        //color.b = texture( renderedTexture, vec2( 0.5f, 0.1f )).b;
//        color.a = 0.5f;
//    }


}