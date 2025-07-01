/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <sstream>

#include <cstring>
#include <donut/core/log.h>
#include <donut/core/math/math.h>
#include <cassert>
#include <cstdarg>
#include <iostream>
#include <iterator>
#include <fstream>
#include <array>
#include <map>
#include <tuple>

#include "rtxmg/subdivision/shape.h"
#include "rtxmg/scene/string_utils.h"
#include "rtxmg/utils/debug.h"

using namespace donut::math;
using namespace donut;

namespace fs = std::filesystem;

#define mtl_assert(cond, msg) \
logassert(cond, "Malformed material file %s:%d: %s", m_filepath, lineNumber, msg)

#define NEED_MTL() mtl_assert(mtl != nullptr, "No material defined");

#define obj_assert(cond, msg) \
logassert(cond, "Malformed obj file %s:%d: %s", m_filepath, lineNumber, msg)

#ifndef _MSC_VER
int sscanf_s( const char *buffer, const char *format, ...) 
{
    va_list args;
    va_start(args,format);
    int result = vsscanf(buffer, format, args);
    va_end(args);
    return result;
}
#endif

static std::vector<std::unique_ptr<Shape::material>>
parseMtllib(const char* m_filepath)
{
    std::vector<std::unique_ptr<Shape::material>> mtls;

    std::string mtlstring = ReadASCIIFile(m_filepath);
    char* str = const_cast<char*>(mtlstring.c_str()), line[256];

    Shape::material* mtl = nullptr;

    bool done = false;
    float r, g, b, a;
    int lineNumber = 0;
    while (!done)
    {
        done = sgets(line, sizeof(line), &str) == 0;
        lineNumber++;

        size_t lineLen = strlen(line);
        if (lineLen == 0)
            continue;

        char* end = &line[lineLen - 1];
        if (*end == '\n')
            *end = '\0'; // strip trailing nl
        switch (line[0])
        {
        case 'n':
        {
            char name[256] = { "" };
            if (sscanf_s(line, "newmtl %255s", name, 256) == 1)
            {
                mtl = mtls.emplace_back(std::make_unique<Shape::material>()).get();
                mtl->name = name;
            }
        } break;
        case 'K':
            NEED_MTL();
            mtl_assert(sscanf_s(line + 2, " %f %f %f", &r, &g, &b) == 3, "Missing RGB values");
            switch (line[1])
            {
            case 'a':
                mtl->ka[0] = r;
                mtl->ka[1] = g;
                mtl->ka[2] = b;
                break;
            case 'd':
                mtl->kd[0] = r;
                mtl->kd[1] = g;
                mtl->kd[2] = b;
                break;
            case 's':
                mtl->ks[0] = r;
                mtl->ks[1] = g;
                mtl->ks[2] = b;
                break;
            case 'e':
                mtl->ke[0] = r;
                mtl->ke[1] = g;
                mtl->ke[2] = b;
                break;
            default:
                mtl_assert(false, "Unknown K value");
            }
            break;
        case 'N':
            NEED_MTL();
            mtl_assert(sscanf_s(line + 2, " %f", &a) == 1, "Missing N value");
            switch (line[1])
            {
            case 's':
                mtl->ns = a;
                break;
            case 'i':
                mtl->ni = a;
                break;
            default:
                mtl_assert(false, "Unknown N");
            }
            break;
        case 'd':
            NEED_MTL();
            mtl_assert(sscanf_s(line, "d %f", &a) == 1, "malformed dissolve");
            mtl->d = a;
            break;
        case 'T':
            NEED_MTL();
            mtl_assert(sscanf_s(line, "Tf %f %f %f", &r, &g, &b) == 3, "Malformed transmission filter");
            mtl->tf[0] = r;
            mtl->tf[1] = g;
            mtl->tf[2] = b;
            break;
        case 'i':
            NEED_MTL();
            int illum;
            mtl_assert(sscanf_s(line, "illum %d", &illum) == 1, "Malformed illum");
            mtl->illum = illum;
            break;
        case 's':
            NEED_MTL();
            if (sscanf_s(line, "sharpness %f", &a) == 1)
                mtl->sharpness = a;
            break;
        case 'm':
            NEED_MTL();
            if (strncmp(line, "map_", 4) == 0)
            {
                char buf[1024];

                switch (line[4])
                {
                case 'K':
                    mtl_assert(sscanf_s(line + 6, " %1023s", buf, 1024) == 1, "Malformed map_K line");
                    switch (line[5])
                    {
                    case 'a':
                        mtl->map_ka = buf;
                        break;
                    case 'd':
                        mtl->map_kd = buf;
                        break;
                    case 'e':
                        mtl->map_ke = buf;
                        break;
                    case 's':
                        mtl->map_ks = buf;
                        break;
                    }
                    break;
                case 'B':
                    if (sscanf_s(line + 5, "ump -bm %f -bb %f %1023s", &mtl->bm, &mtl->bb,
                        buf, 1024) == 3)
                        mtl->map_bump = buf;
                    else if (sscanf_s(line + 5, "ump -bm %f %1023s", &mtl->bm, buf, 1024) == 2)
                        mtl->map_bump = buf;
                    else if (sscanf_s(line + 5, "ump %1023s", buf, 1024) == 1)
                        mtl->map_bump = buf;
                    else mtl_assert(false, "Malformed bump map line");
                    break;
                case 'P':
                    switch (line[5])
                    {
                    case 'r':
                        mtl_assert(sscanf_s(line + 5, "r %1023s", buf, 1024) == 1, "Malformed map_Pr line");
                        mtl->map_pr = buf;
                        break;
                    case 'm':
                        mtl_assert(sscanf_s(line + 5, "m %1023s", buf, 1024) == 1, "Malformed map_Pm line");
                        mtl->map_pm = buf;
                        break;
                    default:
                        mtl_assert(false, "Unknown map_P value");
                    }
                    break;
                case 'R':
                    mtl_assert(sscanf_s(line + 5, "ma %1023s", buf, 1024) == 1, "Malformed map_R line");
                    mtl->map_rma = buf;
                    break;
                case 'O':
                    mtl_assert(sscanf_s(line + 5, "rm %1023s", buf, 1024) == 1, "Malformed map_KO line");
                    mtl->map_orm = buf;
                    break;
                }
            }
            break;
        case 'P':
            NEED_MTL();
            switch (line[1])
            {
            case 'r':
                mtl_assert(sscanf_s(line + 2, " %f", &mtl->Pr) == 1, "Malformed Pr line");
                break;
            case 'm':
                mtl_assert(sscanf_s(line + 2, " %f", &mtl->Pm) == 1, "Malformed Pm line");
                break;
            case 's':
                mtl_assert(sscanf_s(line + 2, " %f", &mtl->Ps) == 1, "Malformed Ps line");
                break;
            case 'c':
                switch (line[2])
                {
                case ' ':
                    mtl_assert(sscanf_s(line + 2, " %f", &mtl->Pc) == 1, "Malformed Pc line");
                    break;
                case 'r':
                    mtl_assert(sscanf_s(line + 3, " %f", &mtl->Pcr) == 1, "Malformed Pcr line");
                    break;
                }
            }
            break;
        case 'a':
            NEED_MTL();
            {
                float a = 0.f;
                if (sscanf_s(line, "aniso %f", &a) == 1)
                    mtl->aniso = a;
                else if (sscanf_s(line, "anisor %f", &a) == 1)
                    mtl->anisor = a;
                else mtl_assert(false, "Malformed aniso line");
            }
            break;
        }
    }
    return mtls;
}

static char const* parseDouble2(char const* ptr, std::vector<float2>& values)
{
    double x, y;
    ptr = ParseDouble(ptr, &x);
    ptr = ParseDouble(ptr, &y);
    values.push_back({ (float)x, (float)y });
    return ptr;
}

static char const* parseDouble3(char const* ptr, std::vector<float3>& values)
{
    double x, y, z;
    ptr = SkipWhiteSpace(ptr);
    ptr = ParseDouble(ptr, &x);
    ptr = ParseDouble(ptr, &y);
    ptr = ParseDouble(ptr, &z);
    values.push_back({ (float)x, (float)y, (float)z });
    return ptr;
}

static char const* parseFace(char const* ptr, std::vector<int>& vertcounts,
    std::vector<int>& verts, std::vector<int>& uvs,
    std::vector<int>& facenormals)
{
    static constexpr int invalid_id = -1;

    ptr = SkipWhiteSpace(ptr);

    uint8_t count = 0;
    int4 vert = { invalid_id, invalid_id, invalid_id, invalid_id };
    while (*ptr && !IsNewLine(*ptr))
    {
        ptr = ParseInt(ptr, &vert.x);
        if (*ptr == '/')
        {
            ++ptr;
            if (*ptr != '/')
                ptr = ParseInt(ptr, &vert.y);

            if (*ptr == '/')
                ptr = ParseInt(++ptr, &vert.z);
        }

        if (vert.x != invalid_id)
            verts.push_back(vert.x - 1);
        if (vert.y != invalid_id)
            uvs.push_back(vert.y - 1);
        if (vert.z != invalid_id)
            facenormals.push_back(vert.z - 1);

        ++count;
        ptr = SkipWhiteSpace(ptr);
    }
    vertcounts.push_back(count);
    return ptr;
}

std::unique_ptr<Shape> parseObj(char const* m_filepath, Scheme shapescheme,
    bool isLeftHanded, bool parseMaterials)
{
    uint64_t size = 0;
    std::unique_ptr<uint8_t const []> obj_str = ReadBigFile(m_filepath, &size);
    std::unique_ptr<Shape> s = std::make_unique<Shape>();

    s->scheme = shapescheme;
    s->isLeftHanded = isLeftHanded;
    s->aabb = box3::empty();

    auto tstart = std::chrono::steady_clock::now();

    
    short usemtl = -1;
    int delta = 0;
    uint64_t lineNumber = 1;

    char groupName[512] = { 0 };
    char buf[256];
    char line[1024];
    char* str = (char *)(obj_str.get());

    bool done = false;
    while (!done)
    {
        done = sgets(line, sizeof(line), &str) == 0;
        if (line[0])
        {
            char* end = &line[strlen(line) - 1];
            if (*end == '\n')
                *end = '\0';  // strip trailing nl
        }
        const char* ptr = line;
        switch (*ptr)
        {
        case 'v':
            ++ptr;
            switch (*ptr)
            {
            case ' ':
                ptr = parseDouble3(ptr, s->verts);
                s->aabb |= s->verts.back();
                break;
            case 't':
                ++ptr;
                ptr = parseDouble2(ptr, s->uvs);
                break;
            case 'n':
                ++ptr;
                ptr = parseDouble3(ptr, s->normals);
                break;
            }
            break;
        case 'f':
            ++ptr;
            if (*ptr == ' ' || *ptr == '\t')
            {
                ptr = parseFace(ptr, s->nvertsPerFace, s->faceverts, s->faceuvs,
                    s->facenormals);
                if (!s->mtls.empty())
                {
                    s->mtlbind.push_back(usemtl);
                }
            }
            break;
        case 't':
            ++ptr;
            if (*ptr == ' ')
            {
                Shape::tag t;
                if (Shape::tag::ParseTag(ptr, &t))
                    s->tags.emplace_back(std::move(t));
            }
            break;
        case 'g':
            ++ptr;
            if (*ptr == ' ')
            {
                obj_assert(sscanf_s(ptr, " %255s%n", groupName, 256, &delta) == 1, "Malformed group name");
                ptr += delta;
            }
            break;
        case 'u':
            obj_assert(sscanf_s(ptr, "usemtl %255s%n", buf, 256, &delta) == 1, "Malformed usemtl");
            usemtl = static_cast<short>(s->FindMaterial(buf));
            ptr += delta;
            break;
        case 'm':
            obj_assert(sscanf_s(ptr, "mtllib %255s%n", buf, 256, &delta) == 1, "Malformed mtllib");
            if (parseMaterials)
            {
                fs::path p = buf;

                if (!fs::is_regular_file(p))
                    p = fs::path(m_filepath).parent_path() / buf;
                if (fs::is_regular_file(p))
                {
                    s->mtls =
                        parseMtllib(p.generic_string().c_str());
                    s->mtllib = buf;
                }
            }
            ptr += delta;
            break;
        case 'c':
            obj_assert(sscanf_s(ptr, "capslib %255s%n", buf, 256, &delta) == 1, "Malformed capslib");
            if (parseMaterials)
            {
                fs::path p = buf;

                if (!fs::is_regular_file(p))
                    p = fs::path(m_filepath).parent_path() / buf;

                if (fs::is_regular_file(p))
                    s->capslib = buf;
            }
            ptr += delta;
            break;
        case '#':
            break;
        }
        ++lineNumber;
    }

    return s;
}

bool Shape::tag::ParseTag(char const* cp, tag* t)
{

    char buf[256];

    while (*cp == ' ')
        cp++;
    if (sscanf_s(cp, "%255s", buf, 256) != 1)
        return false;
    while (*cp && *cp != ' ')
        cp++;
    t->name = buf;

    int nints = 0, nfloats = 0, nstrings = 0;
    while (*cp == ' ')
        cp++;
    if (sscanf_s(cp, "%d/%d/%d", &nints, &nfloats, &nstrings) != 3)
        return false;
    while (*cp && *cp != ' ')
        cp++;

    t->intargs.reserve(nints);
    for (int i = 0; i < nints; ++i)
    {
        int val;
        while (*cp == ' ')
            cp++;
        if (sscanf_s(cp, "%d", &val) != 1)
            return false;
        t->intargs.push_back(val);
        while (*cp && *cp != ' ')
            cp++;
    }

    t->floatargs.reserve(nfloats);
    for (int i = 0; i < nfloats; ++i)
    {
        float val;
        while (*cp == ' ')
            cp++;
        if (sscanf_s(cp, "%f", &val) != 1)
            return false;
        t->floatargs.push_back(val);
        while (*cp && *cp != ' ')
            cp++;
    }

    t->stringargs.reserve(nstrings);
    for (int i = 0; i < nstrings; ++i)
    {
        char val[512];
        while (*cp == ' ')
            cp++;
        if (sscanf_s(cp, "%511s", val, 512) != 1)
            return false;
        t->stringargs.push_back(std::string(val));
        while (*cp && *cp != ' ')
            cp++;
    }
    return true;
}

std::string Shape::tag::GenTag() const
{
    std::stringstream t;

    t << "t " << name << " ";

    t << intargs.size() << "/" << floatargs.size() << "/" << stringargs.size()
        << " ";

    std::copy(intargs.begin(), intargs.end(), std::ostream_iterator<int>(t, " "));
    // t<<" ";

    t << std::fixed;
    std::copy(floatargs.begin(), floatargs.end(),
        std::ostream_iterator<float>(t, " "));
    // t<<" ";

    std::copy(stringargs.begin(), stringargs.end(),
        std::ostream_iterator<std::string>(t, " "));
    t << "\n";

    return t.str();
}

int Shape::FindMaterial(char const* name)
{
    for (int i = 0; i < (int)mtls.size(); ++i)
        if (mtls[i]->name == name)
            return i;
    return -1;
}

//
// serialization / deserialization
//

auto writeTrivial = []<typename T>(std::ofstream & os, T const& v) -> std::ofstream&
{
    static_assert(std::is_trivial_v<T> && std::is_standard_layout_v<T>);
    os.write(reinterpret_cast<char const*>(&v), sizeof(T));
    return os;
};

auto readTrivial = []<typename T>(std::ifstream & is, T & v) -> std::ifstream&
{
    static_assert(std::is_trivial_v<T> && std::is_standard_layout_v<T>);
    is.read(reinterpret_cast<char*>(&v), sizeof(T));
    return is;
};

std::ofstream& operator<<(std::ofstream& os, float2 const& v)
{
    writeTrivial(os, v.x);
    writeTrivial(os, v.y);
    return os;
}

std::ifstream& operator>>(std::ifstream& is, float2& v)
{
    readTrivial(is, v.x);
    readTrivial(is, v.y);
    return is;
}


std::ofstream& operator<<(std::ofstream& os, float3 const& v)
{
    writeTrivial(os, v.x);
    writeTrivial(os, v.y);
    writeTrivial(os, v.z);
    return os;
}

std::ifstream& operator>>(std::ifstream& is, float3& v)
{
    readTrivial(is, v.x);
    readTrivial(is, v.y);
    readTrivial(is, v.z);
    return is;
}

//template <unsigned int M, unsigned int N>
//std::ofstream& operator<<(std::ofstream&w os, otk::Matrix<M, N> const& m)
//{
//	constexpr size_t m_size = M * N * sizeof(typename std::remove_pointer<decltype(m.getData())>::type);
//	os.write(reinterpret_cast<char const*>(m.getData()), m_size);
//	return os;
//}
//
//template <unsigned int M, unsigned int N>
//std::ifstream& operator>>(std::ifstream& is, otk::Matrix<M, N>& m)
//{
//	constexpr size_t m_size = M * N * sizeof(typename std::remove_pointer<decltype(m.getData())>::type);
//	is.read(reinterpret_cast<char*>(m.getData()), m_size);
//	return is;
//}

std::ofstream& operator<<(std::ofstream& os, box3 const& aabb)
{
    os << aabb.m_mins;
    os << aabb.m_maxs;
    return os;
}

std::ifstream& operator>>(std::ifstream& is, box3& aabb)
{
    is >> aabb.m_mins;
    is >> aabb.m_maxs;
    return is;
}

std::ofstream& operator<<(std::ofstream& os, std::string const& s)
{
    if (writeTrivial(os, s.size()); !s.empty())
        os.write(s.data(), s.size());
    return os;
}

std::ifstream& operator>>(std::ifstream& is, std::string& s)
{
    size_t size;
    if (readTrivial(is, size); size > 0)
    {
        s.resize(size);
        is.read(s.data(), size);
    }
    return is;
}

template <typename T>
std::ofstream& operator<<(std::ofstream& os, std::vector<T> const& v)
{
    if (writeTrivial(os, v.size()); !v.empty())
    {
        if constexpr (std::is_trivial_v<T> && std::is_standard_layout_v<T>)
            os.write(reinterpret_cast<char const*>(v.data()), v.size() * sizeof(T));
        else
            for (size_t i = 0; i < v.size(); ++i)
                os << v[i];
    }
    return os;
}

template <typename T>
std::ifstream& operator>>(std::ifstream& is, std::vector<T>& v)
{
    v.clear();
    size_t size = 0;
    if (readTrivial(is, size); size > 0)
    {
        v.resize(size);
        if constexpr (std::is_trivial_v<T> && std::is_standard_layout_v<T>)
            is.read(reinterpret_cast<char*>(v.data()), size * sizeof(T));
        else
            for (size_t i = 0; i < v.size(); ++i)
                is >> v[i];
    }
    return is;
}

std::ofstream& operator<<(std::ofstream& os, Shape::tag const& t)
{
    os << t.name;
    os << t.intargs;
    os << t.floatargs;
    os << t.stringargs;
    return os;
}

std::ifstream& operator>>(std::ifstream& is, Shape::tag& t)
{
    is >> t.name;
    is >> t.intargs;
    is >> t.floatargs;
    is >> t.stringargs;
    return is;
}

void Shape::WriteShape(const std::string& objFile) const
{
    using namespace std::chrono;
    fs::path cacheFile = fs::path(objFile).replace_extension(".bin");

    if (std::ofstream os(cacheFile, std::ios::out | std::ofstream::binary); os.is_open())
    {
        system_clock::duration::rep objFileTimeStamp = (fs::last_write_time(objFile).time_since_epoch() + version).count();

        writeTrivial(os, objFileTimeStamp);

        os << verts;
        os << normals;
        os << uvs;
        os << faceverts;
        os << faceuvs;
        os << facenormals;
        os << nvertsPerFace;
        os << tags;

        os << mtllib;
        os << mtlbind;

        os << aabb;

        os << capslib;
    }
}


bool Shape::ReadShape(const std::string& objFile)
{
    using namespace std::chrono;

    system_clock::duration::rep objFileTimeStamp = (fs::last_write_time(objFile).time_since_epoch() + version).count();

    fs::path cacheFile = fs::path(objFile).replace_extension(".bin");

    if (std::ifstream is(cacheFile, std::ios::in | std::ofstream::binary); is.is_open())
    {
        system_clock::duration::rep binFileTimStamp;

        readTrivial(is, binFileTimStamp);

        // if timestamp stored in .bin doesn't match the .obj's timestamp return false
        // i.e. read/load the .obj file instead
        if (binFileTimStamp == objFileTimeStamp)
        {
            is >> verts;
            is >> normals;
            is >> uvs;
            is >> faceverts;
            is >> faceuvs;
            is >> facenormals;
            is >> nvertsPerFace;
            is >> tags;

            is >> mtllib;
            is >> mtlbind;

            is >> aabb;

            is >> capslib;
            return true;
        }
    }
    return false;
}

// Create shape with default cube geometry from catmark_cube.h
std::unique_ptr<Shape> Shape::DefaultShape()
{
    auto shape = std::unique_ptr<Shape>(new Shape);

    shape->verts = std::vector<float3>{
        {-0.5f, -0.5f, 0.5f},  {0.5f, -0.5f, 0.5f},  {-0.5f, 0.5f, 0.5f},
        {0.5f, 0.5f, 0.5f},    {-0.5f, 0.5f, -0.5f}, {0.5f, 0.5f, -0.5f},
        {-0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, -0.5f} };

    shape->uvs = std::vector<float2>{
        {0.375, 0.00}, {0.625, 0.00}, {0.375, 0.25}, {0.625, 0.25}, {0.375, 0.50},
        {0.625, 0.50}, {0.375, 0.75}, {0.625, 0.75}, {0.375, 1.00}, {0.625, 1.00},
        {0.875, 0.00}, {0.875, 0.25}, {0.125, 0.00}, {0.125, 0.25} };

    shape->nvertsPerFace = std::vector<int>{ 4, 4, 4, 4, 4, 4 };

    shape->faceverts = std::vector<int>{ 0, 1, 3, 2, 2, 3, 5, 4, 4, 5, 7, 6,
                                        6, 7, 1, 0, 1, 7, 5, 3, 6, 0, 2, 4 };

    shape->faceuvs = std::vector<int>{ 0, 1, 3, 2, 2, 3,  5,  4, 4,  5, 7, 6,
                                      6, 7, 9, 8, 1, 10, 11, 3, 12, 0, 2, 13 };

    shape->aabb = donut::math::box(float3(-.5f), float3(.5f));

    return shape;
}



//
// udim
//

static std::array<char const*, 2> udim_patterns = { ".<UDIM>", ".<UVTILE>" };

static std::string udimPath(std::string const& filename, char const* udimID = nullptr)
{
    for (char const* pattern : udim_patterns)
        if (size_t p = filename.find(pattern); p != std::string::npos)
            return filename.substr(0, p + 1) + (udimID ? udimID : "%04d")
            + filename.substr(p + strlen(pattern), std::string::npos);
    return {};
};

template <typename... Ts>
constexpr auto materialMaps(Shape::material& m)
{
    return std::forward_as_tuple(m.map_ka, m.map_kd, m.map_ks, m.map_bump, m.map_ke, m.map_pr, m.map_pm, m.map_rma, m.map_orm);
}
template <typename... Ts>
constexpr auto materialMaps(Shape::material const& m)
{
    return std::forward_as_tuple(m.map_ka, m.map_kd, m.map_ks, m.map_bump, m.map_ke, m.map_pr, m.map_pm, m.map_rma, m.map_orm);
}
static bool hasUdims(Shape::material const& mtl)
{
    bool result = false;

    auto hasUdim = [&result](std::string const& texpath)
        {
            if (result || texpath.empty())
                return;
            for (auto pattern : udim_patterns)
                if (size_t p = texpath.find(pattern); p != std::string::npos)
                {
                    result = true;
                    return;
                }
        };

    std::apply([&result, &hasUdim](auto const&... maps) { (hasUdim(maps), ...); }, materialMaps(mtl));
    return result;
}

static bool hasUdims(Shape const& shape)
{
    for (auto& mtl : shape.mtls)
        if (hasUdims(*mtl))
            return true;
    return false;
}

static std::vector<uint32_t> findUdims(fs::path const& basepath, Shape::material const& mtl)
{
    std::vector<uint32_t> udims;

    auto search = [&basepath, &udims](fs::path const& texpath)
        {
            if (udims.empty() && !texpath.empty())
            {
                if (std::string pattern = udimPath(texpath.filename().generic_string()); !pattern.empty())
                {
                    fs::path dir = (basepath / texpath.parent_path());

                    if (!fs::is_directory(dir))
                        return;

                    for (auto const& entry : fs::directory_iterator(dir))
                    {
                        int id;
                        if (sscanf_s(entry.path().filename().generic_string().c_str(), pattern.c_str(), &id) == 1)
                            udims.push_back(id);
                    }

                    // remove duplicates (ex. caused by dds version of texture)
                    std::sort(udims.begin(), udims.end());
                    udims.erase(std::unique(udims.begin(), udims.end()), udims.end());

                    if (udims.empty())
                        throw std::runtime_error(std::string("cannot find udims for: ") + texpath.generic_string());
                }
            }
        };

    std::apply([&search](auto const&... maps) { (search(maps), ...); }, materialMaps(mtl));

    return udims;
}

static std::unique_ptr<Shape::material> resolveUdim(fs::path const& basepath, Shape::material const& mtl, uint32_t udim)
{
    auto newMtl = std::make_unique<Shape::material>(mtl);

    auto resolve = [&basepath, &udim](std::string& texpath)
        {
            if (texpath.empty())
                return;

            texpath = udimPath(texpath, std::to_string(udim).c_str());

            if (!fs::is_regular_file(basepath / texpath))
                throw std::runtime_error(std::string("cannot find udim: ") + (basepath / texpath).generic_string().c_str());
        };

    std::apply([&resolve](auto&... maps) { (resolve(maps), ...); }, materialMaps(*newMtl));

    newMtl->udim = udim;

    return newMtl;
}

static void resolveUdims(Shape& shape)
{
    if (!hasUdims(shape) || shape.mtlbind.empty() || shape.faceuvs.empty())
        return;

    fs::path basepath = shape.filepath.parent_path();

    // generate new library where materials with udims are duplicated

    std::vector<std::unique_ptr<Shape::material>> mtls;
    mtls.reserve(shape.mtls.size() * 2);

    std::map<uint64_t, uint32_t> mtlsMap;

    auto makeKey = [](uint32_t mtlid, uint32_t udim) { return uint64_t(mtlid) << 32 | uint64_t(udim); };

    for (uint32_t i = 0; i < shape.mtls.size(); ++i)
    {
        auto mtl = std::move(shape.mtls[i]);

        if (hasUdims(*mtl))
        {
            std::vector<uint32_t> udims = findUdims(basepath, *mtl);

            for (uint32_t udim : udims)
            {
                //printf("material %s : %d udim: %d -> %d\n", mtl->name.c_str(), i, udim, (uint32_t)mtls.m_size());
                mtlsMap[makeKey(i, udim)] = static_cast<uint32_t>(mtls.size());
                mtls.emplace_back(resolveUdim(basepath, *mtl, udim));
            }
        }
        else
        {
            //printf("material %s : %d -> %d\n", mtl->name.c_str(), i, (uint32_t)mtls.m_size());
            mtlsMap[makeKey(i, 0)] = static_cast<uint32_t>(mtls.size());
            mtls.emplace_back(std::move(mtl));
        }
    }

    mtls.shrink_to_fit();

    // re-assign material bindings

    assert(shape.mtlbind.size() == shape.GetNumFaces());

    // see: https://learn.foundry.com/katana/Content/ug/checking_uvs/multi_tile_textures.html
    auto makeUdim = [](float2 uv) -> uint32_t
        {
            return 1001 + uint32_t(std::trunc(uv.x) + 10 * std::trunc(uv.y));
        };

    std::vector<unsigned short> mtlbind(shape.mtlbind.size());

    for (uint32_t face = 0, vertCount = 0; face < shape.GetNumFaces(); ++face)
    {
        uint32_t nverts = shape.nvertsPerFace[face];
        uint32_t mtlId = shape.mtlbind[face];

        auto it = mtlsMap.find(makeKey(mtlId, 0));

        if (it == mtlsMap.end())
        {
            float2 texcoord = shape.uvs[shape.faceuvs[vertCount]];

            uint32_t udim = makeUdim(texcoord);

            it = mtlsMap.find(makeKey(mtlId, udim));

            assert(it != mtlsMap.end() && mtls[it->second]->udim == udim);

            for (uint32_t vert = 1; vert < nverts; ++vert)
            {
                texcoord = shape.uvs[shape.faceuvs[vertCount + vert]];

                if (makeUdim(texcoord) != udim)
                    throw std::runtime_error(std::string("udim crosses bounds for face " + std::to_string(face)));
            }
        }
        else
            assert(mtls[it->second]->udim == 0);

        mtlbind[face] = static_cast<short>(it->second);

        vertCount += nverts;
    }

    shape.mtls = std::move(mtls);
    shape.mtlbind = std::move(mtlbind);
}


std::unique_ptr<Shape> Shape::LoadObjFile(const fs::path& m_filepath,
    bool parseMaterials, bool requireUVs)
{
    std::unique_ptr<Shape> shape = std::make_unique<Shape>();

    std::string filepathStr = m_filepath.lexically_normal().generic_string();

    constexpr bool isLeftHanded = false;

    if (!shape->ReadShape(filepathStr))
    {
        shape = parseObj(filepathStr.c_str(), Scheme::kCatmark, isLeftHanded,
            parseMaterials);

        if (!shape)
        {
            log::error("Error parsing obj file: %s", filepathStr.c_str());
            return nullptr;
        }

        shape->WriteShape(filepathStr);
    }

    shape->filepath = filepathStr;


    // Require texcoords, to simplify mesh processing later
    if (!shape->HasUV() && requireUVs)
    {
        log::fatal("OBJ file is missing texture coords");
    }

    if (parseMaterials && !shape->mtllib.empty())
    {
        fs::path p = shape->mtllib;
        if (!fs::is_regular_file(p))
        {
            p = shape->filepath.parent_path() / shape->mtllib;
        }

        if (fs::is_regular_file(p))
        {
            log::info("Loading mtl file from disk: %s", p.generic_string().c_str());
            shape->mtls = parseMtllib(p.generic_string().c_str());
        }
        else
        {
            log::error("Error loading mtl file: %s", p.generic_string().c_str());
        }

        resolveUdims(*shape);

        if (!shape->capslib.empty())
        {
        	fs::path p = shape->capslib;

            if (!fs::is_regular_file(p))
            {
                p = shape->filepath.parent_path() / shape->capslib;
            }
            if (fs::is_regular_file(p))
            {
                // TODO: load capsules
                //log::info("Loading caps file from disk: %s", p.generic_string().c_str());
                //shape->capsules.Load(p.generic_string());
            }
        }
    }

    return shape;
}
