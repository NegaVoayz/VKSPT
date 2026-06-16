// Stats overlay compute shader — renders ray tracing stats panel onto outputImage
// [numthreads(8,8,1)], dispatched at 35×23 → 280×184 px panel

import common.bindings;
import rt.payload;

// Encoding: glyphs[charIdx * 7 + row], lower 5 bits = pixel row (bit4=leftmost)
static const uint kGlyphRows = 7;
static const uint kGlyphCount = 40;

// Character order (index → char):
// 0:space 1-10:'0'-'9' 11-36:'A'-'Z' 37:'.' 38:':' 39:'/'
// Map: ASCII - offset
uint charIdx(uint c)
{
    if (c == 32u) return 0u;
    if (c >= 48u && c <= 57u) return c - 47u;  // '0'=48→1
    if (c >= 65u && c <= 90u) return c - 54u;  // 'A'=65→11
    if (c == 46u) return 37u;  // '.'
    if (c == 58u) return 38u;  // ':'
    if (c == 47u) return 39u;  // '/'
    return 0u;  // fallback to space
}

static const uint kFont[280] = {
// space (0)
0,0,0,0,0,0,0,
// 0 (1)
0x0E,0x11,0x13,0x15,0x19,0x11,0x0E,
// 1 (2)
0x04,0x0C,0x04,0x04,0x04,0x04,0x0E,
// 2 (3)
0x0E,0x11,0x01,0x02,0x04,0x08,0x1F,
// 3 (4)
0x0E,0x11,0x01,0x06,0x01,0x11,0x0E,
// 4 (5)
0x02,0x06,0x0A,0x12,0x1F,0x02,0x02,
// 5 (6)
0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E,
// 6 (7)
0x06,0x08,0x10,0x1E,0x11,0x11,0x0E,
// 7 (8)
0x1F,0x01,0x02,0x04,0x08,0x08,0x08,
// 8 (9)
0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E,
// 9 (10)
0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C,
// A (11)
0x04,0x0A,0x11,0x11,0x1F,0x11,0x11,
// B (12)
0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E,
// C (13)
0x0E,0x11,0x10,0x10,0x10,0x11,0x0E,
// D (14)
0x1C,0x12,0x11,0x11,0x11,0x12,0x1C,
// E (15)
0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F,
// F (16)
0x1F,0x10,0x10,0x1E,0x10,0x10,0x10,
// G (17)
0x0E,0x11,0x10,0x17,0x11,0x11,0x0E,
// H (18)
0x11,0x11,0x11,0x1F,0x11,0x11,0x11,
// I (19)
0x0E,0x04,0x04,0x04,0x04,0x04,0x0E,
// J (20)
0x07,0x02,0x02,0x02,0x02,0x12,0x0C,
// K (21)
0x11,0x12,0x14,0x18,0x14,0x12,0x11,
// L (22)
0x10,0x10,0x10,0x10,0x10,0x10,0x1F,
// M (23)
0x11,0x1B,0x15,0x15,0x11,0x11,0x11,
// N (24)
0x11,0x19,0x15,0x13,0x11,0x11,0x11,
// O (25)
0x0E,0x11,0x11,0x11,0x11,0x11,0x0E,
// P (26)
0x1E,0x11,0x11,0x1E,0x10,0x10,0x10,
// Q (27)
0x0E,0x11,0x11,0x11,0x15,0x12,0x0D,
// R (28)
0x1E,0x11,0x11,0x1E,0x14,0x12,0x11,
// S (29)
0x0E,0x11,0x10,0x0E,0x01,0x11,0x0E,
// T (30)
0x1F,0x04,0x04,0x04,0x04,0x04,0x04,
// U (31)
0x11,0x11,0x11,0x11,0x11,0x11,0x0E,
// V (32)
0x11,0x11,0x11,0x11,0x0A,0x0A,0x04,
// W (33)
0x11,0x11,0x11,0x15,0x15,0x1B,0x11,
// X (34)
0x11,0x11,0x0A,0x04,0x0A,0x11,0x11,
// Y (35)
0x11,0x11,0x0A,0x04,0x04,0x04,0x04,
// Z (36)
0x1F,0x01,0x02,0x04,0x08,0x10,0x1F,
// . (37)
0x00,0x00,0x00,0x00,0x00,0x0C,0x0C,
// : (38)
0x00,0x0C,0x0C,0x00,0x0C,0x0C,0x00,
// / (39)
0x01,0x02,0x02,0x04,0x08,0x08,0x10,
};

// Draw one glyph at (px, py) in given color, blending onto outputImage
void drawGlyph(uint gx, uint gy, uint ci, float4 color)
{
    for (uint row = 0; row < kGlyphRows; row++) {
        uint bits = kFont[ci * kGlyphRows + row];
        for (uint col = 0; col < 5; col++) {
            if ((bits >> (4u - col)) & 1u) {
                outputImage[int2(int(gx) + int(col), int(gy) + int(row))] = color;
            }
        }
    }
}

// Draw a text string at position, each char is 6px wide (5 glyph + 1 gap), 7px tall
void drawString(uint x, uint y, uint str[64], uint len, float4 color)
{
    for (uint i = 0; i < len; i++) {
        uint ci = charIdx(str[i]);
        drawGlyph(x + i * 6u, y, ci, color);
    }
}

// Format unsigned int with commas into buffer, return length
uint formatUint(uint val, inout uint buf[64], uint offset)
{
    if (val == 0u) { buf[offset] = 48u; return 1u; }
    uint tmp[10]; uint d = 0;
    uint v = val;
    while (v > 0u) { tmp[d] = 48u + (v % 10u); v /= 10u; d++; }
    uint ri = d;
    uint commas = 0u;
    for (uint i = 0; i < d; i++) {
        if (i > 0u && (i % 3u) == 0u) {
            buf[offset + ri + commas] = 44u; // ','
            commas++;
        }
        buf[offset + ri + commas] = tmp[i];
        ri--;
    }
    return d + commas;
}

// Format float with 1 decimal place
uint formatFloat1(float val, inout uint buf[64], uint offset)
{
    uint ipart = uint(val);
    uint fpart = uint((val - float(ipart)) * 10.0 + 0.5);
    uint len = formatUint(ipart, buf, offset);
    buf[offset + len] = 46u; // '.'
    len++;
    uint flen = formatUint(fpart, buf, offset + len);
    return len + flen;
}

// Draw histogram bar at given position + index
void drawBar(uint x, uint y, uint barW, uint barH, uint maxH)
{
    if (barH == 0u) return;
    for (uint r = 0; r < barH; r++) {
        for (uint c = 0; c < barW; c++) {
            outputImage[int2(int(x) + int(c), int(y) - int(r))] = float4(0.3, 0.7, 1.0, 0.9);
        }
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    int px = int(tid.x);
    int py = int(tid.y);

    // Panel background: 35×23 groups × 8×8 = 280×184 px
    if (px < 280 && py < 184) {
        // Text lines: full opaque clear (glyph height 7px + 1px margin each side)
        bool inText = (py >= 3  && py < 12)  || (py >= 13 && py < 22)
                   || (py >= 23 && py < 32)  || (py >= 39 && py < 48)
                   || (py >= 75 && py < 84)  || (py >= 103 && py < 112)
                   || (py >= 115 && py < 124);
        if (inText)
            outputImage[int2(px, py)] = float4(0.05, 0.05, 0.08, 1.0);
        else {
            float4 bg = outputImage[int2(px, py)];
            outputImage[int2(px, py)] = lerp(bg, float4(0.05, 0.05, 0.08, 1.0), 0.75);
        }
    }

    // Only thread 0 renders text
    if (px != 0 || py != 0) return;

    uint buf[64];
    uint len;

    uint prim  = rayStats[0];
    uint child = rayStats[1];
    uint avgCh = prim > 0u ? (child * 10u + prim / 2u) / prim : 0u;  // ×10 for 1 decimal

    {
        uint str[64] = {82u,65u,89u,83u,58u,32u}; // "RAYS: "
        len = 6u;
        len += formatUint(prim, str, len);
        str[len++] = 32u; // space
        len += formatUint(child, str, len);
        // " avg "
        str[len++] = 32u; str[len++] = 97u; str[len++] = 118u; str[len++] = 103u; str[len++] = 32u;
        // avg children
        len += formatUint(avgCh / 10u, str, len);
        str[len++] = 46u;
        len += formatUint(avgCh % 10u, str, len);
        drawString(4u, 4u, str, len, float4(0.2, 1.0, 0.4, 0.95));
    }

    // Line2: generations 0-7 ----
    {
        uint str[64] = {71u,69u,78u,58u,32u}; // "GEN: "
        len = 5u;
        for (uint g = 0; g < 8; g++) {
            len += formatUint(rayStats[2 + g], str, len);
            str[len++] = 32u;
        }
        drawString(4u, 14u, str, len, float4(0.6, 0.8, 1.0, 0.9));
    }

    // Line3: bounce histogram top entries ----
    {
        uint str[64] = {66u,78u,67u,58u,32u}; // "BNC: "
        len = 5u;
        uint maxBnc = 0u; uint bestBin = 0u;
        for (uint b = 0; b < 16; b++) {
            uint v = rayStats[34 + b];
            if (v > maxBnc) { maxBnc = v; bestBin = b; }
            if (v > 0u) {
                if (b < 10u) str[len++] = 48u + b;
                else { str[len++] = 49u; str[len++] = 48u + b - 10u; }
                str[len++] = 58u; // ':'
                len += formatUint(v, str, len);
                str[len++] = 32u;
            }
        }
        drawString(4u, 24u, str, len, float4(1.0, 0.8, 0.4, 0.9));
    }

    // Line4: Lambda width histogram bars ----
    {
        uint str[64] = {76u,87u,58u,32u}; // "LW: "
        len = 4u;
        // Find max for scaling
        uint lwMax = 1u;
        for (uint b = 0; b < 40; b++) { uint v = rayStats[98 + b]; if (v > lwMax) lwMax = v; }
        uint barMaxH = 30u;
        for (uint b = 0; b < 40; b++) {
            uint v = rayStats[98 + b];
            if (v > 0u) {
                uint h = (v * barMaxH + lwMax / 2u) / lwMax;
                if (h == 0u) h = 1u;
                drawBar(4u + b * 7u, 68u, 5u, h, barMaxH);
            }
        }
        // Scale label
        len += formatUint(lwMax, str, len);
        drawString(4u, 40u, str, len, float4(0.7, 0.7, 1.0, 0.9));
    }

    // Line5: Spread histogram bars ----
    {
        uint str[64] = {83u,80u,58u,32u}; // "SP: "
        len = 4u;
        uint spMax = 1u;
        for (uint b = 0; b < 64; b++) { uint v = rayStats[138 + b]; if (v > spMax) spMax = v; }
        uint barMaxH = 25u;
        for (uint b = 0; b < 64; b++) {
            uint v = rayStats[138 + b];
            if (v > 0u) {
                uint h = (v * barMaxH + spMax / 2u) / spMax;
                if (h == 0u) h = 1u;
                drawBar(4u + b * 4u, 96u, 3u, h, barMaxH);
            }
        }
        len += formatUint(spMax, str, len);
        str[len++] = 32u; str[len++] = 108u; str[len++] = 111u; str[len++] = 103u; str[len++] = 50u; str[len++] = 40u; str[len++] = 100u; str[len++] = 47u; str[len++] = 48u; // " log2(d/0.01)"
        str[len++] = 46u; str[len++] = 48u; str[len++] = 49u; str[len++] = 41u;
        drawString(4u, 76u, str, len, float4(1.0, 0.6, 0.8, 0.9));
    }

    // Line6: Accum count + FPS + GPU ms ----
    {
        uint str2[64] = {65u,67u,67u,58u,32u}; // "ACC: "
        uint len2 = 5u;
        len2 += formatUint(uint(max(0, params.frameIndex)), str2, len2);
        // "  FPS: "
        str2[len2++] = 32u; str2[len2++] = 70u; str2[len2++] = 80u; str2[len2++] = 83u; str2[len2++] = 58u; str2[len2++] = 32u;
        len2 += formatFloat1(max(0.0f, params.fps), str2, len2);
        // "  GPU: "
        str2[len2++] = 32u; str2[len2++] = 71u; str2[len2++] = 80u; str2[len2++] = 85u; str2[len2++] = 58u; str2[len2++] = 32u;
        len2 += formatFloat1(max(0.0f, params.gpuMilliseconds), str2, len2);
        str2[len2++] = 109u; str2[len2++] = 115u; // "ms"
        drawString(4u, 104u, str2, len2, float4(0.8, 0.8, 0.8, 0.9));
    }

    // Line7: Camera position ----
    {
        uint str3[64] = {80u,79u,83u,58u,32u}; // "POS: "
        uint len3 = 5u;
        len3 += formatFloat1(params.camOrigin[0], str3, len3);
        str3[len3++] = 32u;
        len3 += formatFloat1(params.camOrigin[1], str3, len3);
        str3[len3++] = 32u;
        len3 += formatFloat1(params.camOrigin[2], str3, len3);
        drawString(4u, 116u, str3, len3, float4(0.6, 0.8, 1.0, 0.9));
    }
}
