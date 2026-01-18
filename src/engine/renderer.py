from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from OpenGL import GL

from .math3d import Mat4
from .shader import Shader
from .texture import Texture


# -----------------------------
# Main forward shader (writes a small gbuffer)
# -----------------------------

VERTEX_SRC = """
#version 330 core
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform mat4 u_lightSpace;

out vec3 v_worldPos;
out vec3 v_normal;
out vec2 v_uv;
out vec4 v_lightSpacePos;

bool isnan3(vec3 v) {
    return (v.x != v.x) || (v.y != v.y) || (v.z != v.z);
}

void main() {
    vec4 world = u_model * vec4(a_position, 1.0);
    v_worldPos = world.xyz;

    // Models in this project use scale+translation (and occasional rotations). This is OK here.
    mat3 m = mat3(u_model);
    v_normal = normalize(m * a_normal);
    if (isnan3(v_normal) || length(v_normal) < 0.0001) {
        v_normal = vec3(0.0, 1.0, 0.0);
    }

    v_uv = a_uv;
    v_lightSpacePos = u_lightSpace * world;
    gl_Position = u_projection * u_view * world;
}
"""

FRAGMENT_SRC = """
#version 330 core

in vec3 v_worldPos;
in vec3 v_normal;
in vec2 v_uv;
in vec4 v_lightSpacePos;

layout(location = 0) out vec4 oColor;   // HDR linear color
layout(location = 1) out vec4 oNormal;  // view-space normal encoded 0..1
layout(location = 2) out float oMask;   // SSAO apply mask (1 = apply, 0 = skip)

uniform vec3 u_color;
uniform bool u_useTexture;
uniform sampler2D u_diffuseTex;
uniform float u_uvScale;

uniform vec3 u_viewPos;
uniform mat4 u_view;

uniform vec3 u_lightDir;
uniform vec3 u_lightColor;
uniform vec3 u_ambientColor;
uniform float u_specStrength;
uniform float u_shininess;

uniform float u_matSpecStrength;
uniform float u_matShininess;
uniform vec3 u_emissive;

uniform vec3 u_fogColor;
uniform float u_fogStart;
uniform float u_fogEnd;

// Shadows
uniform sampler2D u_shadowMap;
uniform float u_shadowStrength;
uniform float u_shadowSoftness;
uniform bool u_receiveShadows;

// God rays
uniform bool u_godRays;
uniform float u_godRaysStrength;
uniform int u_godRaysSteps;

// Transparency / sprites
uniform float u_alpha;
uniform float u_ssaoMask;
uniform int u_sprite; // 0=none, 1=muzzle flash, 2=smoke

bool isnan3(vec3 v) {
    return (v.x != v.x) || (v.y != v.y) || (v.z != v.z);
}

float softShadowPCF(vec4 lightSpacePos, float softness) {
    vec3 proj = lightSpacePos.xyz / max(lightSpacePos.w, 1e-6);
    proj = proj * 0.5 + 0.5;
    if (proj.z > 1.0) return 1.0;

    // Bias reduces acne.
    float bias = 0.0015;

    float texel = 1.0 / float(textureSize(u_shadowMap, 0).x);
    float r = clamp(softness, 0.0, 8.0) * texel;

    // 9-tap PCF
    float shadow = 0.0;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            vec2 off = vec2(float(x), float(y)) * r;
            float closest = texture(u_shadowMap, proj.xy + off).r;
            shadow += (proj.z - bias) > closest ? 0.0 : 1.0;
        }
    }
    shadow /= 9.0;
    return shadow;
}

vec3 applyFog(vec3 col, float dist) {
    float fogFactor = 1.0;
    if (u_fogEnd > u_fogStart + 0.001) {
        fogFactor = clamp((u_fogEnd - dist) / (u_fogEnd - u_fogStart), 0.0, 1.0);
    }
    vec3 fogCol = u_fogColor;
    if (isnan3(fogCol)) fogCol = vec3(0.55, 0.70, 0.95);
    return mix(fogCol, col, fogFactor);
}

float spriteMask(vec2 uv, int spriteType) {
    // uv is 0..1
    vec2 p = uv * 2.0 - 1.0;
    float r = length(p);

    // Soft circle by default
    float m = smoothstep(1.0, 0.0, r);

    if (spriteType == 1) {
        // Muzzle flash: sharper core + slight streak
        float core = smoothstep(0.65, 0.0, r);
        float streak = smoothstep(0.15, 0.0, abs(p.y)) * smoothstep(1.0, 0.0, abs(p.x));
        m = max(core, 0.35 * streak);
    } else if (spriteType == 2) {
        // Smoke: softer, with some pseudo-noise breakup
        float n = fract(sin(dot(uv * 128.0, vec2(12.9898, 78.233))) * 43758.5453);
        m *= smoothstep(0.15, 0.85, n);
        m = pow(m, 1.15);
    }

    return clamp(m, 0.0, 1.0);
}

void main() {
    vec3 N = normalize(v_normal);
    if (isnan3(N) || length(N) < 0.0001) N = vec3(0.0, 1.0, 0.0);

    vec3 albedo = u_color;
    if (u_useTexture) {
        vec3 t = texture(u_diffuseTex, v_uv * u_uvScale).rgb;
        if (isnan3(t)) t = vec3(1.0);
        albedo *= t;
    }

    vec3 L = normalize(-u_lightDir);
    vec3 V = normalize(u_viewPos - v_worldPos);
    vec3 H = normalize(L + V);

    float diff = max(dot(N, L), 0.0);
    float specPow = max(2.0, u_shininess * u_matShininess);
    float spec = pow(max(dot(N, H), 0.0), specPow) * (u_specStrength * u_matSpecStrength);

    float shadow = 1.0;
    if (u_receiveShadows) {
        shadow = softShadowPCF(v_lightSpacePos, u_shadowSoftness);
        shadow = mix(1.0, shadow, clamp(u_shadowStrength, 0.0, 1.0));
    }

    vec3 ambient = u_ambientColor * albedo;
    vec3 direct = (diff * albedo * u_lightColor + spec * u_lightColor) * shadow;
    vec3 lit = ambient + direct + u_emissive;

    // Volumetric sun rays (very cheap screen-space approx using the shadow term)
    if (u_godRays) {
        // Brighter when looking towards the sun and when not shadowed.
        float sunDot = clamp(dot(normalize(u_lightDir), normalize(v_worldPos - u_viewPos)), -1.0, 1.0);
        float facing = pow(clamp(-sunDot, 0.0, 1.0), 2.0);
        float ray = (1.0 - shadow) * facing;
        lit += ray * u_godRaysStrength * u_lightColor;
    }

    // Fog BEFORE tonemap (HDR)
    float dist = length(v_worldPos - u_viewPos);
    lit = applyFog(lit, dist);

    // Sprite alpha mask (for muzzle flash / smoke)
    float a = clamp(u_alpha, 0.0, 1.0);
    if (u_sprite != 0) {
        a *= spriteMask(v_uv, u_sprite);
    }

    // Output HDR color (linear) + alpha
    oColor = vec4(max(lit, vec3(0.0)), a);

    // Output view-space normal (encoded)
    vec3 nView = normalize(mat3(u_view) * N);
    oNormal = vec4(nView * 0.5 + 0.5, 1.0);

    // SSAO mask (1 = apply). Viewmodel/sprites can set this to 0.
    oMask = clamp(u_ssaoMask, 0.0, 1.0);
}
"""


# -----------------------------
# Shadow map shaders
# -----------------------------

DEPTH_VERTEX_SRC = """
#version 330 core
layout(location = 0) in vec3 a_position;

uniform mat4 u_model;
uniform mat4 u_lightSpace;

void main() {
    gl_Position = u_lightSpace * u_model * vec4(a_position, 1.0);
}
"""

DEPTH_FRAGMENT_SRC = """
#version 330 core
void main() {
    // Depth only
}
"""


# -----------------------------
# Screen-space shaders (SSAO, Bloom, Composite+FXAA)
# -----------------------------

SCREEN_VERT = """
#version 330 core
layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec2 a_uv;
out vec2 v_uv;
void main(){
    v_uv = a_uv;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
"""

SSAO_FRAG = """
#version 330 core
in vec2 v_uv;
out float FragAO;

uniform sampler2D u_depthTex;
uniform sampler2D u_normalTex;
uniform sampler2D u_noiseTex;

uniform mat4 u_projection;
uniform mat4 u_invProjection;

uniform vec3 u_samples[16];
uniform vec2 u_noiseScale;
uniform float u_radius;
uniform float u_bias;

vec3 reconstructViewPos(vec2 uv, float depth01) {
    // depth01 is [0..1]
    float z = depth01 * 2.0 - 1.0;
    vec4 clip = vec4(uv * 2.0 - 1.0, z, 1.0);
    vec4 view = u_invProjection * clip;
    view.xyz /= max(view.w, 1e-6);
    return view.xyz;
}

void main(){
    float depth = texture(u_depthTex, v_uv).r;
    if (depth >= 0.99999) { FragAO = 1.0; return; }

    vec3 pos = reconstructViewPos(v_uv, depth);

    vec3 n = texture(u_normalTex, v_uv).xyz * 2.0 - 1.0;
    n = normalize(n);

    // Random rotation from small noise texture
    vec3 rand = texture(u_noiseTex, v_uv * u_noiseScale).xyz * 2.0 - 1.0;
    rand = normalize(rand);

    vec3 tangent = normalize(rand - n * dot(rand, n));
    vec3 bitangent = cross(n, tangent);
    mat3 TBN = mat3(tangent, bitangent, n);

    float occlusion = 0.0;
    for (int i = 0; i < 16; i++) {
        vec3 samp = TBN * u_samples[i];
        vec3 sampPos = pos + samp * u_radius;

        // Project sample pos back to screen to sample depth
        vec4 offset = u_projection * vec4(sampPos, 1.0);
        offset.xyz /= max(offset.w, 1e-6);
        vec2 uv = offset.xy * 0.5 + 0.5;

        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) continue;

        float sampDepth = texture(u_depthTex, uv).r;
        if (sampDepth >= 0.99999) continue;

        vec3 sampView = reconstructViewPos(uv, sampDepth);

        // View space looks down -Z. More negative z is further.
        float rangeCheck = smoothstep(0.0, 1.0, u_radius / (abs(pos.z - sampView.z) + 1e-6));
        float occ = (sampView.z >= (sampPos.z + u_bias)) ? 1.0 : 0.0;
        occlusion += occ * rangeCheck;
    }

    occlusion = 1.0 - (occlusion / 16.0);
    FragAO = clamp(occlusion, 0.0, 1.0);
}
"""

BLUR_FRAG = """
#version 330 core
in vec2 v_uv;
out vec4 FragColor;

uniform sampler2D u_tex;
uniform vec2 u_dir; // (1,0)=horizontal, (0,1)=vertical
uniform vec2 u_texel; // 1/size

void main(){
    // 9-tap gaussian-ish weights
    float w0 = 0.227027;
    float w1 = 0.1945946;
    float w2 = 0.1216216;
    float w3 = 0.054054;
    float w4 = 0.016216;

    vec2 off = u_dir * u_texel;

    vec4 c = texture(u_tex, v_uv) * w0;
    c += texture(u_tex, v_uv + off * 1.0) * w1;
    c += texture(u_tex, v_uv - off * 1.0) * w1;
    c += texture(u_tex, v_uv + off * 2.0) * w2;
    c += texture(u_tex, v_uv - off * 2.0) * w2;
    c += texture(u_tex, v_uv + off * 3.0) * w3;
    c += texture(u_tex, v_uv - off * 3.0) * w3;
    c += texture(u_tex, v_uv + off * 4.0) * w4;
    c += texture(u_tex, v_uv - off * 4.0) * w4;

    FragColor = c;
}
"""

BLOOM_EXTRACT_FRAG = """
#version 330 core
in vec2 v_uv;
out vec4 FragColor;

uniform sampler2D u_hdr;
uniform float u_threshold;

void main(){
    vec3 c = texture(u_hdr, v_uv).rgb;
    float lum = dot(c, vec3(0.2126, 0.7152, 0.0722));
    vec3 outc = (lum > u_threshold) ? c : vec3(0.0);
    FragColor = vec4(outc, 1.0);
}
"""

COMPOSITE_FXAA_FRAG = """
#version 330 core
in vec2 v_uv;
out vec4 FragColor;

uniform sampler2D u_hdr;
uniform sampler2D u_bloom;
uniform sampler2D u_ao;
uniform sampler2D u_mask;

uniform float u_exposure;
uniform float u_bloomStrength;
uniform float u_aoStrength;
uniform vec2 u_invRes;

// FXAA controls
uniform int u_fxaa;

vec3 tonemap(vec3 hdr){
    float e = clamp(u_exposure, 0.25, 4.0);
    vec3 c = vec3(1.0) - exp(-hdr * e);
    // gamma
    c = pow(max(c, vec3(0.0)), vec3(1.0 / 2.2));
    return clamp(c, 0.0, 1.0);
}

vec3 compositeAt(vec2 uv){
    vec3 base = texture(u_hdr, uv).rgb;
    vec3 bloom = texture(u_bloom, uv).rgb;
    float ao = texture(u_ao, uv).r;
    float mask = texture(u_mask, uv).r;

    // 1=no AO, 0=dark
    float aoFactor = mix(1.0, mix(1.0, ao, clamp(u_aoStrength, 0.0, 1.0)), mask);

    vec3 hdr = base * aoFactor + bloom * u_bloomStrength;
    return tonemap(hdr);
}

float luma(vec3 c){
    return dot(c, vec3(0.299, 0.587, 0.114));
}

void main(){
    vec3 c = compositeAt(v_uv);

    if (u_fxaa == 0) {
        FragColor = vec4(c, 1.0);
        return;
    }

    // Lightweight FXAA (does compositeAt for neighbors as well)
    vec2 px = u_invRes;

    vec3 cN = compositeAt(v_uv + vec2(0.0, px.y));
    vec3 cS = compositeAt(v_uv - vec2(0.0, px.y));
    vec3 cE = compositeAt(v_uv + vec2(px.x, 0.0));
    vec3 cW = compositeAt(v_uv - vec2(px.x, 0.0));

    float lC = luma(c);
    float lN = luma(cN);
    float lS = luma(cS);
    float lE = luma(cE);
    float lW = luma(cW);

    float lMin = min(lC, min(min(lN, lS), min(lE, lW)));
    float lMax = max(lC, max(max(lN, lS), max(lE, lW)));

    // If no edge, keep original
    float contrast = lMax - lMin;
    if (contrast < 0.06) {
        FragColor = vec4(c, 1.0);
        return;
    }

    vec2 dir;
    dir.x = -((lN + lS) - 2.0 * lC);
    dir.y =  ((lE + lW) - 2.0 * lC);

    // Normalize and clamp
    float dirReduce = max((lN + lS + lE + lW) * 0.25 * 0.5, 1e-5);
    float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = clamp(dir * rcpDirMin, vec2(-8.0), vec2(8.0)) * px;

    // Sample along edge
    vec3 cA = 0.5 * (compositeAt(v_uv + dir * (1.0/3.0 - 0.5)) + compositeAt(v_uv + dir * (2.0/3.0 - 0.5)));
    vec3 cB = cA * 0.5 + 0.25 * (compositeAt(v_uv + dir * -0.5) + compositeAt(v_uv + dir * 0.5));

    float lB = luma(cB);
    vec3 outc = (lB < lMin || lB > lMax) ? cA : cB;
    FragColor = vec4(outc, 1.0);
}
"""


@dataclass
class RenderItem:
    mesh: object
    model: Mat4
    color: Iterable[float]
    texture: Optional[Texture] = None
    uv_scale: float = 1.0

    # Material controls
    spec_strength: float = 1.0     # multiplier over renderer.spec_strength
    shininess: float = 32.0        # 2..128
    emissive: Iterable[float] = (0.0, 0.0, 0.0)

    # Shadows
    cast_shadows: bool = True
    receive_shadows: bool = True

    # Depth controls (viewmodel etc.)
    depth_test: bool = True
    depth_write: bool = True

    # Transparency / sprites
    alpha: float = 1.0
    blend_mode: int = 0  # 0=opaque, 1=alpha, 2=additive
    sprite: int = 0      # 0=none, 1=muzzle flash, 2=smoke

    # SSAO apply mask (0 for viewmodel/sprites)
    ssao_mask: float = 1.0


class Renderer:
    def __init__(self) -> None:
        # Main shaders
        self.shader = Shader(VERTEX_SRC, FRAGMENT_SRC)
        self.depth_shader = Shader(DEPTH_VERTEX_SRC, DEPTH_FRAGMENT_SRC)

        # Post shaders
        self.ssao_shader = Shader(SCREEN_VERT, SSAO_FRAG)
        self.blur_shader = Shader(SCREEN_VERT, BLUR_FRAG)
        self.bloom_extract_shader = Shader(SCREEN_VERT, BLOOM_EXTRACT_FRAG)
        self.composite_shader = Shader(SCREEN_VERT, COMPOSITE_FXAA_FRAG)

        # Main uniform locations
        p = self.shader.program
        self._loc_model = GL.glGetUniformLocation(p, b"u_model")
        self._loc_view = GL.glGetUniformLocation(p, b"u_view")
        self._loc_proj = GL.glGetUniformLocation(p, b"u_projection")
        self._loc_light_space = GL.glGetUniformLocation(p, b"u_lightSpace")

        self._loc_color = GL.glGetUniformLocation(p, b"u_color")
        self._loc_use_tex = GL.glGetUniformLocation(p, b"u_useTexture")
        self._loc_uv_scale = GL.glGetUniformLocation(p, b"u_uvScale")
        self._loc_view_pos = GL.glGetUniformLocation(p, b"u_viewPos")

        self._loc_light_dir = GL.glGetUniformLocation(p, b"u_lightDir")
        self._loc_light_color = GL.glGetUniformLocation(p, b"u_lightColor")
        self._loc_ambient_color = GL.glGetUniformLocation(p, b"u_ambientColor")
        self._loc_spec_strength = GL.glGetUniformLocation(p, b"u_specStrength")
        self._loc_shininess = GL.glGetUniformLocation(p, b"u_shininess")

        self._loc_mat_spec = GL.glGetUniformLocation(p, b"u_matSpecStrength")
        self._loc_mat_shininess = GL.glGetUniformLocation(p, b"u_matShininess")
        self._loc_emissive = GL.glGetUniformLocation(p, b"u_emissive")

        self._loc_shadow_strength = GL.glGetUniformLocation(p, b"u_shadowStrength")
        self._loc_shadow_softness = GL.glGetUniformLocation(p, b"u_shadowSoftness")
        self._loc_receive_shadows = GL.glGetUniformLocation(p, b"u_receiveShadows")

        self._loc_godrays = GL.glGetUniformLocation(p, b"u_godRays")
        self._loc_godrays_strength = GL.glGetUniformLocation(p, b"u_godRaysStrength")
        self._loc_godrays_steps = GL.glGetUniformLocation(p, b"u_godRaysSteps")

        self._loc_fog_color = GL.glGetUniformLocation(p, b"u_fogColor")
        self._loc_fog_start = GL.glGetUniformLocation(p, b"u_fogStart")
        self._loc_fog_end = GL.glGetUniformLocation(p, b"u_fogEnd")

        self._loc_alpha = GL.glGetUniformLocation(p, b"u_alpha")
        self._loc_ssao_mask = GL.glGetUniformLocation(p, b"u_ssaoMask")
        self._loc_sprite = GL.glGetUniformLocation(p, b"u_sprite")

        # Samplers
        self._loc_diffuse_sampler = GL.glGetUniformLocation(p, b"u_diffuseTex")
        self._loc_shadow_sampler = GL.glGetUniformLocation(p, b"u_shadowMap")

        # Depth shader uniforms
        dp = self.depth_shader.program
        self._dloc_model = GL.glGetUniformLocation(dp, b"u_model")
        self._dloc_light_space = GL.glGetUniformLocation(dp, b"u_lightSpace")

        # Lighting defaults
        self.light_dir = (-0.25, -1.0, -0.35)
        self.light_color = (1.0, 0.98, 0.92)
        self.ambient_color = (0.28, 0.28, 0.32)
        self.spec_strength = 0.30
        self.shininess = 32.0
        self.exposure = 1.05

        # Fog defaults
        self.fog_color = (0.55, 0.70, 0.95)
        self.fog_start = 12.0
        self.fog_end = 38.0

        # God rays
        self.godrays_enabled = True
        self.godrays_strength = 0.75
        self.godrays_steps = 14

        # Shadows
        self.shadow_strength = 0.85
        self.shadow_size = 2048
        self.shadow_softness = 2.8
        self._shadow_fbo = 0
        self._shadow_tex = 0
        self._init_shadow_map()

        # Post effects toggles & params
        self.ssao_enabled = True
        self.ssao_radius = 1.25
        self.ssao_bias = 0.025
        self.ssao_strength = 1.0

        self.bloom_enabled = True
        self.bloom_threshold = 1.35
        self.bloom_strength = 0.9

        self.fxaa_enabled = True

        # Internal: gbuffer & post resources
        self._screen_w = 0
        self._screen_h = 0
        self._gbuffer_fbo = 0
        self._g_color = 0
        self._g_normal = 0
        self._g_mask = 0
        self._g_depth = 0

        self._ssao_fbo = 0
        self._ssao_tex = 0
        self._ssao_blur_fbo = 0
        self._ssao_blur_tex = 0

        self._bloom_extract_fbo = 0
        self._bloom_extract_tex = 0
        self._bloom_pingpong_fbo = [0, 0]
        self._bloom_pingpong_tex = [0, 0]

        # SSAO kernel + noise
        self._ssao_samples = self._make_ssao_kernel(16)
        self._noise_tex = self._make_noise_texture(4)

        # Screen quad
        self._quad_vao, self._quad_vbo = self._create_screen_quad()

        # Keep last matrices for SSAO
        self._last_projection = Mat4.identity()
        self._last_inv_projection = Mat4.identity()

        # MSAA request (for window). Offscreen is single-sample; FXAA handles aliasing.
        self.msaa_enabled = True
        GL.glEnable(GL.GL_MULTISAMPLE)
        GL.glEnable(GL.GL_DEPTH_TEST)

    # ---------------------
    # Resource creation
    # ---------------------

    def _create_screen_quad(self) -> tuple[int, int]:
        # Fullscreen quad (triangle strip) with UVs
        vao = GL.glGenVertexArrays(1)
        vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)

        data = np.array([
            # pos      uv
            -1.0, -1.0,  0.0, 0.0,
             1.0, -1.0,  1.0, 0.0,
            -1.0,  1.0,  0.0, 1.0,
             1.0,  1.0,  1.0, 1.0,
        ], dtype=np.float32)

        GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 4 * 4, GL.ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, 4 * 4, GL.ctypes.c_void_p(2 * 4))

        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        return vao, vbo

    def _make_ssao_kernel(self, count: int) -> list[list[float]]:
        rng = np.random.default_rng(1337)
        samples: list[list[float]] = []
        for i in range(count):
            v = rng.random(3).astype(np.float32) * 2.0 - 1.0
            v[2] = rng.random(1).astype(np.float32)[0]  # hemisphere
            v = v / (np.linalg.norm(v) + 1e-6)
            scale = i / float(count)
            scale = 0.1 + 0.9 * (scale * scale)
            v *= scale
            samples.append([float(v[0]), float(v[1]), float(v[2])])
        return samples

    def _make_noise_texture(self, size: int) -> int:
        rng = np.random.default_rng(2024)
        noise = rng.random((size, size, 3), dtype=np.float32) * 2.0 - 1.0
        noise[:, :, 2] = 0.0

        tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGB16F,
            size,
            size,
            0,
            GL.GL_RGB,
            GL.GL_FLOAT,
            noise,
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        return tex

    def _create_tex(self, w: int, h: int, internal: int, fmt: int, typ: int, filter_linear: bool = True) -> int:
        tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, internal, w, h, 0, fmt, typ, None)
        if filter_linear:
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        else:
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        return tex

    def _ensure_framebuffers(self, w: int, h: int) -> None:
        if w == self._screen_w and h == self._screen_h and self._gbuffer_fbo != 0:
            return

        self._screen_w = int(w)
        self._screen_h = int(h)

        # Cleanup old
        def _del_tex(t: int) -> None:
            if t:
                GL.glDeleteTextures([t])

        def _del_fbo(f: int) -> None:
            if f:
                GL.glDeleteFramebuffers(1, [f])

        for t in [self._g_color, self._g_normal, self._g_mask, self._g_depth, self._ssao_tex, self._ssao_blur_tex, self._bloom_extract_tex, *self._bloom_pingpong_tex]:
            _del_tex(t)
        for f in [self._gbuffer_fbo, self._ssao_fbo, self._ssao_blur_fbo, self._bloom_extract_fbo, *self._bloom_pingpong_fbo]:
            _del_fbo(f)

        # GBuffer
        self._gbuffer_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._gbuffer_fbo)

        self._g_color = self._create_tex(w, h, GL.GL_RGBA16F, GL.GL_RGBA, GL.GL_FLOAT, filter_linear=True)
        self._g_normal = self._create_tex(w, h, GL.GL_RGBA16F, GL.GL_RGBA, GL.GL_FLOAT, filter_linear=False)
        self._g_mask = self._create_tex(w, h, GL.GL_R8, GL.GL_RED, GL.GL_UNSIGNED_BYTE, filter_linear=False)
        self._g_depth = self._create_tex(w, h, GL.GL_DEPTH_COMPONENT24, GL.GL_DEPTH_COMPONENT, GL.GL_UNSIGNED_INT, filter_linear=False)

        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self._g_color, 0)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT1, GL.GL_TEXTURE_2D, self._g_normal, 0)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT2, GL.GL_TEXTURE_2D, self._g_mask, 0)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_TEXTURE_2D, self._g_depth, 0)

        bufs = [GL.GL_COLOR_ATTACHMENT0, GL.GL_COLOR_ATTACHMENT1, GL.GL_COLOR_ATTACHMENT2]
        GL.glDrawBuffers(len(bufs), bufs)

        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"GBuffer framebuffer incomplete: {status}")

        # SSAO
        self._ssao_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._ssao_fbo)
        self._ssao_tex = self._create_tex(w, h, GL.GL_R8, GL.GL_RED, GL.GL_UNSIGNED_BYTE, filter_linear=False)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self._ssao_tex, 0)
        GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0)

        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"SSAO framebuffer incomplete: {status}")

        self._ssao_blur_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._ssao_blur_fbo)
        self._ssao_blur_tex = self._create_tex(w, h, GL.GL_R8, GL.GL_RED, GL.GL_UNSIGNED_BYTE, filter_linear=False)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self._ssao_blur_tex, 0)
        GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0)

        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"SSAO blur framebuffer incomplete: {status}")

        # Bloom extract
        self._bloom_extract_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._bloom_extract_fbo)
        self._bloom_extract_tex = self._create_tex(w, h, GL.GL_RGBA16F, GL.GL_RGBA, GL.GL_FLOAT, filter_linear=True)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self._bloom_extract_tex, 0)
        GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0)

        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Bloom extract framebuffer incomplete: {status}")

        # Bloom blur ping-pong
        for i in range(2):
            self._bloom_pingpong_fbo[i] = GL.glGenFramebuffers(1)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._bloom_pingpong_fbo[i])
            self._bloom_pingpong_tex[i] = self._create_tex(w, h, GL.GL_RGBA16F, GL.GL_RGBA, GL.GL_FLOAT, filter_linear=True)
            GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self._bloom_pingpong_tex[i], 0)
            GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0)
            status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
            if status != GL.GL_FRAMEBUFFER_COMPLETE:
                raise RuntimeError(f"Bloom pingpong framebuffer incomplete: {status}")

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    # ---------------------
    # Shadow map
    # ---------------------

    def _init_shadow_map(self) -> None:
        self._shadow_fbo = GL.glGenFramebuffers(1)
        self._shadow_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._shadow_tex)

        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_DEPTH_COMPONENT,
            self.shadow_size,
            self.shadow_size,
            0,
            GL.GL_DEPTH_COMPONENT,
            GL.GL_FLOAT,
            None,
        )

        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_BORDER)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_BORDER)
        border = (1.0, 1.0, 1.0, 1.0)
        GL.glTexParameterfv(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_BORDER_COLOR, border)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._shadow_fbo)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_DEPTH_ATTACHMENT,
            GL.GL_TEXTURE_2D,
            self._shadow_tex,
            0,
        )
        GL.glDrawBuffer(GL.GL_NONE)
        GL.glReadBuffer(GL.GL_NONE)

        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Shadow framebuffer incomplete: {status}")

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def render_shadow_map(self, items: list[RenderItem], light_space: Mat4, screen_w: int, screen_h: int) -> None:
        GL.glViewport(0, 0, self.shadow_size, self.shadow_size)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._shadow_fbo)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)

        self.depth_shader.use()
        GL.glUniformMatrix4fv(self._dloc_light_space, 1, GL.GL_FALSE, light_space.to_list())

        GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
        GL.glPolygonOffset(2.0, 4.0)

        for item in items:
            if not getattr(item, "cast_shadows", True):
                continue
            GL.glUniformMatrix4fv(self._dloc_model, 1, GL.GL_FALSE, item.model.to_list())
            item.mesh.draw()

        GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glViewport(0, 0, screen_w, screen_h)

    # ---------------------
    # Frame begin/end
    # ---------------------

    @staticmethod
    def _invert_mat4(m: Mat4) -> Mat4:
        a = np.array(m.to_list(), dtype=np.float32).reshape((4, 4), order="F")
        inv = np.linalg.inv(a)
        return Mat4(inv.reshape(16, order="F").tolist())

    def begin(
        self,
        clear_color: Iterable[float],
        view: Mat4,
        projection: Mat4,
        view_pos: Iterable[float],
        light_space: Mat4,
        screen_w: int,
        screen_h: int,
    ) -> None:
        self._ensure_framebuffers(screen_w, screen_h)

        # Store matrices for SSAO
        self._last_projection = projection
        self._last_inv_projection = self._invert_mat4(projection)

        GL.glViewport(0, 0, screen_w, screen_h)

        # Hard reset of key OpenGL states (HUD safety)
        GL.glDisable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_ONE, GL.GL_ZERO)
        GL.glDisable(GL.GL_COLOR_LOGIC_OP)
        GL.glDisable(GL.GL_SCISSOR_TEST)
        GL.glDisable(GL.GL_STENCIL_TEST)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glColorMask(True, True, True, True)
        GL.glDepthMask(True)
        GL.glEnable(GL.GL_DEPTH_TEST)

        # Offscreen render
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._gbuffer_fbo)

        r, g, b = clear_color
        GL.glClearColor(float(r), float(g), float(b), 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        self.shader.use()

        # Per-frame uniforms
        GL.glUniformMatrix4fv(self._loc_view, 1, GL.GL_FALSE, view.to_list())
        GL.glUniformMatrix4fv(self._loc_proj, 1, GL.GL_FALSE, projection.to_list())
        GL.glUniformMatrix4fv(self._loc_light_space, 1, GL.GL_FALSE, light_space.to_list())

        vp = list(view_pos)
        GL.glUniform3f(self._loc_view_pos, float(vp[0]), float(vp[1]), float(vp[2]))

        ld = self.light_dir
        lc = self.light_color
        ac = self.ambient_color
        GL.glUniform3f(self._loc_light_dir, float(ld[0]), float(ld[1]), float(ld[2]))
        GL.glUniform3f(self._loc_light_color, float(lc[0]), float(lc[1]), float(lc[2]))
        GL.glUniform3f(self._loc_ambient_color, float(ac[0]), float(ac[1]), float(ac[2]))
        GL.glUniform1f(self._loc_spec_strength, float(self.spec_strength))
        GL.glUniform1f(self._loc_shininess, float(self.shininess))

        fc = self.fog_color
        GL.glUniform3f(self._loc_fog_color, float(fc[0]), float(fc[1]), float(fc[2]))
        GL.glUniform1f(self._loc_fog_start, float(self.fog_start))
        GL.glUniform1f(self._loc_fog_end, float(self.fog_end))

        # Shadow map on unit 1
        GL.glUniform1f(self._loc_shadow_strength, float(self.shadow_strength))
        GL.glUniform1f(self._loc_shadow_softness, float(self.shadow_softness))
        GL.glUniform1i(self._loc_shadow_sampler, 1)
        GL.glActiveTexture(GL.GL_TEXTURE0 + 1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._shadow_tex)

        # Diffuse sampler is unit 0
        GL.glUniform1i(self._loc_diffuse_sampler, 0)

        # God rays
        GL.glUniform1i(self._loc_godrays, 1 if self.godrays_enabled else 0)
        GL.glUniform1f(self._loc_godrays_strength, float(self.godrays_strength))
        GL.glUniform1i(self._loc_godrays_steps, int(self.godrays_steps))

    def draw_item(self, item: RenderItem) -> None:
        self.shader.use()

        # Depth state
        prev_depth = GL.glIsEnabled(GL.GL_DEPTH_TEST)
        prev_depth_mask = GL.glGetBooleanv(GL.GL_DEPTH_WRITEMASK)

        if getattr(item, "depth_test", True):
            GL.glEnable(GL.GL_DEPTH_TEST)
        else:
            GL.glDisable(GL.GL_DEPTH_TEST)

        GL.glDepthMask(GL.GL_TRUE if getattr(item, "depth_write", True) else GL.GL_FALSE)

        # Blending
        # Note: we try to use per-draw-buffer blending so that the normal/mask
        # attachments are not blended (prevents SSAO mask artifacts on sprites).
        blend_mode = int(getattr(item, "blend_mode", 0))
        if blend_mode == 0:
            GL.glDisable(GL.GL_BLEND)
            # Best-effort disable on MRT
            if hasattr(GL, "glDisablei"):
                try:
                    GL.glDisablei(GL.GL_BLEND, 0)
                    GL.glDisablei(GL.GL_BLEND, 1)
                    GL.glDisablei(GL.GL_BLEND, 2)
                except Exception:
                    pass
        else:
            GL.glEnable(GL.GL_BLEND)

            # Enable blending only for color attachment 0; keep others unblended.
            if hasattr(GL, "glEnablei"):
                try:
                    GL.glEnablei(GL.GL_BLEND, 0)
                    GL.glDisablei(GL.GL_BLEND, 1)
                    GL.glDisablei(GL.GL_BLEND, 2)
                except Exception:
                    pass

            if blend_mode == 2:
                # additive
                if hasattr(GL, "glBlendFunci"):
                    try:
                        GL.glBlendFunci(0, GL.GL_ONE, GL.GL_ONE)
                    except Exception:
                        GL.glBlendFunc(GL.GL_ONE, GL.GL_ONE)
                else:
                    GL.glBlendFunc(GL.GL_ONE, GL.GL_ONE)
            else:
                if hasattr(GL, "glBlendFunci"):
                    try:
                        GL.glBlendFunci(0, GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
                    except Exception:
                        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
                else:
                    GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

        # Uniforms
        GL.glUniformMatrix4fv(self._loc_model, 1, GL.GL_FALSE, item.model.to_list())
        c = list(item.color)
        GL.glUniform3f(self._loc_color, float(c[0]), float(c[1]), float(c[2]))

        use_tex = 1 if item.texture is not None else 0
        GL.glUniform1i(self._loc_use_tex, int(use_tex))
        GL.glUniform1f(self._loc_uv_scale, float(item.uv_scale))

        GL.glUniform1f(self._loc_mat_spec, float(getattr(item, "spec_strength", 1.0)))
        GL.glUniform1f(self._loc_mat_shininess, float(getattr(item, "shininess", 32.0)))
        e = list(getattr(item, "emissive", (0.0, 0.0, 0.0)))
        GL.glUniform3f(self._loc_emissive, float(e[0]), float(e[1]), float(e[2]))

        recv = 1 if getattr(item, "receive_shadows", True) else 0
        GL.glUniform1i(self._loc_receive_shadows, int(recv))

        GL.glUniform1f(self._loc_alpha, float(getattr(item, "alpha", 1.0)))
        GL.glUniform1f(self._loc_ssao_mask, float(getattr(item, "ssao_mask", 1.0)))
        GL.glUniform1i(self._loc_sprite, int(getattr(item, "sprite", 0)))

        if item.texture is not None:
            item.texture.bind(0)
        else:
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        item.mesh.draw()

        # Restore depth
        if prev_depth:
            GL.glEnable(GL.GL_DEPTH_TEST)
        else:
            GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(prev_depth_mask)

    def end(self) -> None:
        """Run post-processing and present to the default framebuffer."""
        w, h = self._screen_w, self._screen_h
        if w <= 0 or h <= 0:
            return

        # SSAO pass
        if self.ssao_enabled:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._ssao_fbo)
            GL.glViewport(0, 0, w, h)
            GL.glDisable(GL.GL_DEPTH_TEST)
            GL.glDisable(GL.GL_BLEND)
            GL.glClearColor(1.0, 1.0, 1.0, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)

            self.ssao_shader.use()

            # textures
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._g_depth)
            GL.glUniform1i(GL.glGetUniformLocation(self.ssao_shader.program, b"u_depthTex"), 0)

            GL.glActiveTexture(GL.GL_TEXTURE0 + 1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._g_normal)
            GL.glUniform1i(GL.glGetUniformLocation(self.ssao_shader.program, b"u_normalTex"), 1)

            GL.glActiveTexture(GL.GL_TEXTURE0 + 2)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._noise_tex)
            GL.glUniform1i(GL.glGetUniformLocation(self.ssao_shader.program, b"u_noiseTex"), 2)

            GL.glUniformMatrix4fv(
                GL.glGetUniformLocation(self.ssao_shader.program, b"u_projection"),
                1,
                GL.GL_FALSE,
                self._last_projection.to_list(),
            )
            GL.glUniformMatrix4fv(
                GL.glGetUniformLocation(self.ssao_shader.program, b"u_invProjection"),
                1,
                GL.GL_FALSE,
                self._last_inv_projection.to_list(),
            )
            GL.glUniform2f(
                GL.glGetUniformLocation(self.ssao_shader.program, b"u_noiseScale"),
                float(w / 4.0),
                float(h / 4.0),
            )
            GL.glUniform1f(GL.glGetUniformLocation(self.ssao_shader.program, b"u_radius"), float(self.ssao_radius))
            GL.glUniform1f(GL.glGetUniformLocation(self.ssao_shader.program, b"u_bias"), float(self.ssao_bias))

            # samples
            loc_samples = GL.glGetUniformLocation(self.ssao_shader.program, b"u_samples")
            for i, s in enumerate(self._ssao_samples):
                GL.glUniform3f(loc_samples + i, float(s[0]), float(s[1]), float(s[2]))

            self._draw_fullscreen_quad()

            # Blur AO
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._ssao_blur_fbo)
            GL.glClearColor(1.0, 1.0, 1.0, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)

            self.blur_shader.use()
            GL.glUniform2f(GL.glGetUniformLocation(self.blur_shader.program, b"u_dir"), 1.0, 0.0)
            GL.glUniform2f(GL.glGetUniformLocation(self.blur_shader.program, b"u_texel"), 1.0 / float(w), 1.0 / float(h))
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._ssao_tex)
            GL.glUniform1i(GL.glGetUniformLocation(self.blur_shader.program, b"u_tex"), 0)
            self._draw_fullscreen_quad()

        # Bloom extract + blur
        bloom_tex = 0
        if self.bloom_enabled:
            # extract
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._bloom_extract_fbo)
            GL.glViewport(0, 0, w, h)
            GL.glDisable(GL.GL_DEPTH_TEST)
            GL.glDisable(GL.GL_BLEND)
            GL.glClearColor(0.0, 0.0, 0.0, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)

            self.bloom_extract_shader.use()
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._g_color)
            GL.glUniform1i(GL.glGetUniformLocation(self.bloom_extract_shader.program, b"u_hdr"), 0)
            GL.glUniform1f(GL.glGetUniformLocation(self.bloom_extract_shader.program, b"u_threshold"), float(self.bloom_threshold))
            self._draw_fullscreen_quad()

            # blur ping-pong
            self.blur_shader.use()
            GL.glUniform2f(GL.glGetUniformLocation(self.blur_shader.program, b"u_texel"), 1.0 / float(w), 1.0 / float(h))

            horizontal = True
            first = True
            passes = 6
            for i in range(passes):
                idx = 1 if horizontal else 0
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._bloom_pingpong_fbo[idx])
                GL.glClear(GL.GL_COLOR_BUFFER_BIT)
                GL.glUniform2f(
                    GL.glGetUniformLocation(self.blur_shader.program, b"u_dir"),
                    1.0 if horizontal else 0.0,
                    0.0 if horizontal else 1.0,
                )

                GL.glActiveTexture(GL.GL_TEXTURE0)
                if first:
                    GL.glBindTexture(GL.GL_TEXTURE_2D, self._bloom_extract_tex)
                else:
                    GL.glBindTexture(GL.GL_TEXTURE_2D, self._bloom_pingpong_tex[0 if horizontal else 1])
                GL.glUniform1i(GL.glGetUniformLocation(self.blur_shader.program, b"u_tex"), 0)
                self._draw_fullscreen_quad()

                horizontal = not horizontal
                first = False

            bloom_tex = self._bloom_pingpong_tex[0 if horizontal else 1]
        else:
            bloom_tex = 0

        # Composite + FXAA to default framebuffer
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glViewport(0, 0, w, h)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_BLEND)
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        self.composite_shader.use()

        # base hdr
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._g_color)
        GL.glUniform1i(GL.glGetUniformLocation(self.composite_shader.program, b"u_hdr"), 0)

        # bloom
        GL.glActiveTexture(GL.GL_TEXTURE0 + 1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, bloom_tex if bloom_tex else 0)
        GL.glUniform1i(GL.glGetUniformLocation(self.composite_shader.program, b"u_bloom"), 1)

        # ao
        GL.glActiveTexture(GL.GL_TEXTURE0 + 2)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._ssao_blur_tex if self.ssao_enabled else 0)
        GL.glUniform1i(GL.glGetUniformLocation(self.composite_shader.program, b"u_ao"), 2)

        # mask
        GL.glActiveTexture(GL.GL_TEXTURE0 + 3)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._g_mask)
        GL.glUniform1i(GL.glGetUniformLocation(self.composite_shader.program, b"u_mask"), 3)

        GL.glUniform1f(GL.glGetUniformLocation(self.composite_shader.program, b"u_exposure"), float(self.exposure))
        GL.glUniform1f(GL.glGetUniformLocation(self.composite_shader.program, b"u_bloomStrength"), float(self.bloom_strength if self.bloom_enabled else 0.0))
        GL.glUniform1f(GL.glGetUniformLocation(self.composite_shader.program, b"u_aoStrength"), float(self.ssao_strength if self.ssao_enabled else 0.0))
        GL.glUniform2f(GL.glGetUniformLocation(self.composite_shader.program, b"u_invRes"), 1.0 / float(w), 1.0 / float(h))
        GL.glUniform1i(GL.glGetUniformLocation(self.composite_shader.program, b"u_fxaa"), 1 if self.fxaa_enabled else 0)

        self._draw_fullscreen_quad()

        # Restore for HUD
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(True)
        GL.glDisable(GL.GL_BLEND)

    def _draw_fullscreen_quad(self) -> None:
        GL.glBindVertexArray(self._quad_vao)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        GL.glBindVertexArray(0)
