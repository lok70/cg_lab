#define NOMINMAX

#include <GL/glew.h>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <SFML/OpenGL.hpp>
#include <SFML/System.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
#include <iostream>
#include <tuple>
#include <optional>
#include <filesystem>
#include <fstream>

// ШЕЙДЕРЫ

static const char* VS_Color = R"GLSL(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;

uniform vec3 offset;
uniform vec3 scale;      
uniform mat4 rotation;

out vec3 vColor;

void main() {
    vec3 p = aPos * scale;
    vec4 pr = rotation * vec4(p, 1.0);
    pr.xyz += offset;
    gl_Position = pr;
    vColor = aColor;
}
)GLSL";

static const char* FS_Color = R"GLSL(
#version 330 core
in vec3 vColor;
out vec4 FragColor;

void main() {
    FragColor = vec4(vColor, 1.0);
}
)GLSL";

static const char* VS_TexColor = R"GLSL(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
layout(location = 2) in vec2 aTex;

uniform vec3 offset;
uniform mat4 rotation;

out vec3 vColor;
out vec2 vTex;

void main() {
    vec4 pr = rotation * vec4(aPos, 1.0);
    pr.xyz += offset;
    gl_Position = pr;
    vColor = aColor;
    vTex = aTex;
}
)GLSL";

static const char* FS_TexColor = R"GLSL(
#version 330 core
in vec3 vColor;
in vec2 vTex;

uniform sampler2D texture0;
uniform float colorMix;

out vec4 FragColor;

void main() {
    vec3 t = texture(texture0, vTex).rgb;
    vec3 outRGB = mix(t, vColor, colorMix);
    FragColor = vec4(outRGB, 1.0);
}
)GLSL";

static const char* FS_TwoTex = R"GLSL(
#version 330 core
in vec3 vColor; 
in vec2 vTex;

uniform sampler2D texture1;
uniform sampler2D texture2;
uniform float textureMix;

out vec4 FragColor;

void main() {
    vec3 a = texture(texture1, vTex).rgb;
    vec3 b = texture(texture2, vTex).rgb;
    vec3 outRGB = mix(a, b, textureMix);
    FragColor = vec4(outRGB, 1.0);
}
)GLSL";

// ФУНКЦИИ

static GLuint compileShader(GLenum type, const char* src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);

    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char log[512]; glGetShaderInfoLog(s, 512, nullptr, log);
        std::cerr << "Shader Compilation Error:\n" << log << "\n";
    }
    return s;
}

static GLuint linkProgram(const char* vsSrc, const char* fsSrc)
{
    GLuint vs = compileShader(GL_VERTEX_SHADER, vsSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsSrc);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

static GLuint loadTextureSFML(const std::string& fileName)
{
    sf::Image image;
    std::filesystem::path filePath = std::filesystem::current_path() / fileName;

    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << fileName << "\n";
        return 0;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);

    if (!file.read(buffer.data(), size)) return 0;

    if (!image.loadFromMemory(buffer.data(), buffer.size()))
    {
        std::cerr << "Error: SFML failed to load image from memory.\n";
        return 0;
    }

    image.flipVertically();

    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.getSize().x, image.getSize().y,
        0, GL_RGBA, GL_UNSIGNED_BYTE, image.getPixelsPtr());
    glGenerateMipmap(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    return tex;
}

static GLuint createFallbackTexture()
{
    const std::uint8_t data[] = { 255,0,255,255, 0,0,0,255, 0,0,0,255, 255,0,255,255 };
    GLuint tex; glGenTextures(1, &tex); glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 2, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    return tex;
}

static void hsvToRgb(float h, float s, float v, float& r, float& g, float& b)
{
    float hh = h / 60.0f;
    long i = (long)hh;
    float f = hh - (float)i;

    float p = v * (1.0f - s);
    float q = v * (1.0f - f * s);
    float t = v * (1.0f - (1.0f - f) * s);

    switch (i % 6)
    {
    case 0: r = v; g = t; b = p; break;
    case 1: r = q; g = v; b = p; break;
    case 2: r = p; g = v; b = t; break;
    case 3: r = p; g = q; b = v; break;
    case 4: r = t; g = p; b = v; break;
    case 5: r = v; g = p; b = q; break;
    default: r = v; g = p; b = q; break;
    }
}

static void createCircle(std::vector<float>& vertices, std::vector<unsigned int>& indices)
{
    vertices.insert(vertices.end(), { 0.f, 0.f, 0.f,  1.f, 1.f, 1.f });
    int segments = 64;
    for (int i = 0; i <= segments; ++i)
    {
        float angle = (2.f * 3.14159f * i) / segments;
        float x = 0.5f * std::cos(angle);
        float y = 0.5f * std::sin(angle);
        float hue = (360.f * i) / segments;
        float r, g, b;
        hsvToRgb(hue, 1.f, 1.f, r, g, b);
        vertices.insert(vertices.end(), { x, y, 0.f, r, g, b });
    }
    for (int i = 1; i <= segments; ++i)
    {
        indices.push_back(0);
        indices.push_back(i);
        indices.push_back(i + 1);
    }
}

struct Mesh
{
    GLuint vao = 0, vbo = 0, ebo = 0;
    GLsizei indexCount = 0;
};

static Mesh makeMesh(const std::vector<float>& vertices, const std::vector<unsigned int>& indices,
    int strideBytes, const std::vector<std::tuple<GLuint, GLint, int>>& attribs)
{
    Mesh m;
    m.indexCount = (GLsizei)indices.size();
    glGenVertexArrays(1, &m.vao); glGenBuffers(1, &m.vbo); glGenBuffers(1, &m.ebo);

    glBindVertexArray(m.vao);
    glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    for (const auto& attr : attribs)
    {
        GLuint loc; GLint size; int offset;
        std::tie(loc, size, offset) = attr;
        glVertexAttribPointer(loc, size, GL_FLOAT, GL_FALSE, strideBytes, (void*)(intptr_t)offset);
        glEnableVertexAttribArray(loc);
    }
    glBindVertexArray(0);
    return m;
}

// MAIN

enum class Figure { Tetra = 0, CubeTexColor = 1, CubeTwoTex = 2, Circle = 3 };

int main()
{
    sf::ContextSettings settings;
    settings.depthBits = 24;
    settings.majorVersion = 3;
    settings.minorVersion = 3;

    sf::Window window(sf::VideoMode({ 800, 600 }), "OpenGL Lab 10", sf::Style::Default, sf::State::Windowed, settings);
    window.setFramerateLimit(60);
    (void)window.setActive(true);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) return -1;
    glGetError();

    glEnable(GL_DEPTH_TEST);

    GLuint pColor = linkProgram(VS_Color, FS_Color);
    GLuint pTexCol = linkProgram(VS_TexColor, FS_TexColor);
    GLuint pTwoTex = linkProgram(VS_TexColor, FS_TwoTex);

    // --- Данные фигур ---

    // 1. Тетраэдр
    std::vector<float> tetraData = {
         0.0f, 0.5f, 0.0f, 1,0,0,
        -0.5f,-0.5f, 0.5f, 0,1,0,
         0.5f,-0.5f, 0.5f, 0,0,1,
         0.0f,-0.5f,-0.5f, 1,1,0
    };
    std::vector<unsigned int> tetraInd = { 0,1,2, 0,2,3, 0,3,1, 1,3,2 };
    Mesh tetra = makeMesh(tetraData, tetraInd, 6 * sizeof(float), { {0,3,0}, {1,3,3 * sizeof(float)} });

    // 2. Круг
    std::vector<float> circleVerts;
    std::vector<unsigned int> circleIdx;
    createCircle(circleVerts, circleIdx);
    Mesh circle = makeMesh(circleVerts, circleIdx, 6 * sizeof(float), { {0,3,0}, {1,3,3 * sizeof(float)} });

    // 3. Куб
    std::vector<float> cubeVerts = {
        -0.5f,-0.5f, 0.5f, 1,0,0, 0,0,   0.5f,-0.5f, 0.5f, 0,1,0, 1,0,
         0.5f, 0.5f, 0.5f, 0,0,1, 1,1,  -0.5f, 0.5f, 0.5f, 1,1,0, 0,1,
        -0.5f,-0.5f,-0.5f, 1,0,1, 0,0,   0.5f,-0.5f,-0.5f, 0,1,1, 1,0,
         0.5f, 0.5f,-0.5f, 1,1,1, 1,1,  -0.5f, 0.5f,-0.5f, 0,0,0, 0,1,
        -0.5f,-0.5f,-0.5f, 1,0,0, 0,0,  -0.5f,-0.5f, 0.5f, 0,1,0, 1,0,
        -0.5f, 0.5f, 0.5f, 0,0,1, 1,1,  -0.5f, 0.5f,-0.5f, 1,1,0, 0,1,
         0.5f,-0.5f,-0.5f, 1,0,0, 0,0,   0.5f,-0.5f, 0.5f, 0,1,0, 1,0,
         0.5f, 0.5f, 0.5f, 0,0,1, 1,1,   0.5f, 0.5f,-0.5f, 1,1,0, 0,1,
        -0.5f, 0.5f,-0.5f, 1,0,0, 0,0,   0.5f, 0.5f,-0.5f, 0,1,0, 1,0,
         0.5f, 0.5f, 0.5f, 0,0,1, 1,1,  -0.5f, 0.5f, 0.5f, 1,1,0, 0,1,
        -0.5f,-0.5f,-0.5f, 1,0,0, 0,0,   0.5f,-0.5f,-0.5f, 0,1,0, 1,0,
         0.5f,-0.5f, 0.5f, 0,0,1, 1,1,  -0.5f,-0.5f, 0.5f, 1,1,0, 0,1
    };
    std::vector<unsigned int> cubeIdx = {
        0,1,2, 0,2,3, 4,5,6, 4,6,7, 8,9,10, 8,10,11,
        12,13,14, 12,14,15, 16,17,18, 16,18,19, 20,21,22, 20,22,23
    };
    Mesh cube = makeMesh(cubeVerts, cubeIdx, 8 * sizeof(float),
        { {0,3,0}, {1,3,3 * sizeof(float)}, {2,2,6 * sizeof(float)} });

    // --- Текстуры ---
    GLuint texMeme = loadTextureSFML("meme.jpg");
    GLuint texNugget = loadTextureSFML("nugget.png");

    if (!texMeme) texMeme = createFallbackTexture();
    if (!texNugget) texNugget = createFallbackTexture();

    // --- Состояние ---
    Figure active = Figure::Tetra;
    bool showAll = false;

    std::array<glm::vec3, 4> offsets = {};
    std::array<glm::vec3, 4> scales = { glm::vec3(1), glm::vec3(1), glm::vec3(1), glm::vec3(1) };
    std::array<glm::vec3, 4> basePos = {
        glm::vec3(-1.5f, 0.5f, 0.f), glm::vec3(0.0f, 0.5f, 0.f),
        glm::vec3(1.5f, 0.5f, 0.f),  glm::vec3(0.0f,-0.8f, 0.f)
    };

    float mixColorVal = 0.5f;
    float mixTexVal = 0.5f;
    float rotAngle = 0.f;

    while (window.isOpen())
    {
        while (const std::optional event = window.pollEvent())
        {
            if (event->is<sf::Event::Closed>()) window.close();
            else if (const auto* rs = event->getIf<sf::Event::Resized>()) glViewport(0, 0, rs->size.x, rs->size.y);
            else if (const auto* k = event->getIf<sf::Event::KeyPressed>())
            {
                int idx = (int)active;
                float spd = 0.1f;

                if (k->code == sf::Keyboard::Key::Num1) active = Figure::Tetra;
                if (k->code == sf::Keyboard::Key::Num2) active = Figure::CubeTexColor;
                if (k->code == sf::Keyboard::Key::Num3) active = Figure::CubeTwoTex;
                if (k->code == sf::Keyboard::Key::Num4) active = Figure::Circle;
                if (k->code == sf::Keyboard::Key::Tab)  showAll = !showAll;
                if (k->code == sf::Keyboard::Key::R)
                {
                    offsets[idx] = glm::vec3(0);
                    scales[idx] = glm::vec3(1);
                }

                if (k->code == sf::Keyboard::Key::D) offsets[idx].x += spd;
                if (k->code == sf::Keyboard::Key::A) offsets[idx].x -= spd;
                if (k->code == sf::Keyboard::Key::W) offsets[idx].y += spd;
                if (k->code == sf::Keyboard::Key::S) offsets[idx].y -= spd;
                if (k->code == sf::Keyboard::Key::Q) offsets[idx].z += spd;
                if (k->code == sf::Keyboard::Key::E) offsets[idx].z -= spd;

                if (k->code == sf::Keyboard::Key::Right)
                {
                    if (active == Figure::CubeTexColor) mixColorVal = std::min(1.f, mixColorVal + 0.1f);
                    if (active == Figure::CubeTwoTex)   mixTexVal = std::min(1.f, mixTexVal + 0.1f);
                }
                if (k->code == sf::Keyboard::Key::Left)
                {
                    if (active == Figure::CubeTexColor) mixColorVal = std::max(0.f, mixColorVal - 0.1f);
                    if (active == Figure::CubeTwoTex)   mixTexVal = std::max(0.f, mixTexVal - 0.1f);
                }

                if (active == Figure::Circle)
                {
                    if (k->code == sf::Keyboard::Key::I) scales[idx].y += spd;
                    if (k->code == sf::Keyboard::Key::K) scales[idx].y -= spd;
                    if (k->code == sf::Keyboard::Key::L) scales[idx].x += spd;
                    if (k->code == sf::Keyboard::Key::J) scales[idx].x -= spd;
                }
            }
        }

        rotAngle += 0.5f;

        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glm::mat4 rotMat = glm::mat4(1.0f);
        rotMat = glm::rotate(rotMat, glm::radians(30.0f), glm::vec3(1, 0, 0));
        rotMat = glm::rotate(rotMat, glm::radians(rotAngle), glm::vec3(0, 1, 0));

        auto draw = [&](Figure f)
            {
                int i = (int)f;
                glm::vec3 finalPos = offsets[i];
                if (showAll) finalPos += basePos[i];

                if (f == Figure::Tetra)
                {
                    glUseProgram(pColor);
                    glUniform3f(glGetUniformLocation(pColor, "offset"), finalPos.x, finalPos.y, finalPos.z);
                    glUniform3f(glGetUniformLocation(pColor, "scale"), 1.f, 1.f, 1.f);
                    glUniformMatrix4fv(glGetUniformLocation(pColor, "rotation"), 1, GL_FALSE, glm::value_ptr(rotMat));
                    glBindVertexArray(tetra.vao);
                    glDrawElements(GL_TRIANGLES, tetra.indexCount, GL_UNSIGNED_INT, 0);
                }
                else if (f == Figure::Circle)
                {
                    glUseProgram(pColor);
                    glUniform3f(glGetUniformLocation(pColor, "offset"), finalPos.x, finalPos.y, finalPos.z);
                    glUniform3f(glGetUniformLocation(pColor, "scale"), scales[i].x, scales[i].y, 1.f);
                    glUniformMatrix4fv(glGetUniformLocation(pColor, "rotation"), 1, GL_FALSE, glm::value_ptr(rotMat));
                    glBindVertexArray(circle.vao);
                    glDrawElements(GL_TRIANGLES, circle.indexCount, GL_UNSIGNED_INT, 0);
                }
                else if (f == Figure::CubeTexColor)
                {
                    glUseProgram(pTexCol);
                    glUniform3f(glGetUniformLocation(pTexCol, "offset"), finalPos.x, finalPos.y, finalPos.z);
                    glUniformMatrix4fv(glGetUniformLocation(pTexCol, "rotation"), 1, GL_FALSE, glm::value_ptr(rotMat));
                    glUniform1f(glGetUniformLocation(pTexCol, "colorMix"), mixColorVal);
                    glActiveTexture(GL_TEXTURE0);
                    glBindTexture(GL_TEXTURE_2D, texMeme);
                    glUniform1i(glGetUniformLocation(pTexCol, "texture0"), 0);
                    glBindVertexArray(cube.vao);
                    glDrawElements(GL_TRIANGLES, cube.indexCount, GL_UNSIGNED_INT, 0);
                }
                else if (f == Figure::CubeTwoTex)
                {
                    glUseProgram(pTwoTex);
                    glUniform3f(glGetUniformLocation(pTwoTex, "offset"), finalPos.x, finalPos.y, finalPos.z);
                    glUniformMatrix4fv(glGetUniformLocation(pTwoTex, "rotation"), 1, GL_FALSE, glm::value_ptr(rotMat));
                    glUniform1f(glGetUniformLocation(pTwoTex, "textureMix"), mixTexVal);
                    glActiveTexture(GL_TEXTURE0);
                    glBindTexture(GL_TEXTURE_2D, texMeme);
                    glUniform1i(glGetUniformLocation(pTwoTex, "texture1"), 0);
                    glActiveTexture(GL_TEXTURE1);
                    glBindTexture(GL_TEXTURE_2D, texNugget);
                    glUniform1i(glGetUniformLocation(pTwoTex, "texture2"), 1);
                    glBindVertexArray(cube.vao);
                    glDrawElements(GL_TRIANGLES, cube.indexCount, GL_UNSIGNED_INT, 0);
                }
            };

        if (showAll)
        {
            draw(Figure::Tetra); draw(Figure::CubeTexColor); draw(Figure::CubeTwoTex); draw(Figure::Circle);
        }
        else
        {
            draw(active);
        }

        window.display();
    }

    glDeleteBuffers(1, &tetra.vbo); glDeleteVertexArrays(1, &tetra.vao);
    glDeleteBuffers(1, &circle.vbo); glDeleteVertexArrays(1, &circle.vao);
    glDeleteBuffers(1, &cube.vbo); glDeleteVertexArrays(1, &cube.vao);

    return 0;
}
