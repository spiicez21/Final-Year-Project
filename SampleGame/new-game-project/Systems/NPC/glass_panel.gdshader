shader_type canvas_item;

// Frosted-glass panel backing for the NPC UI.
//
// Plain alpha over a busy 3D scene makes text hard to read: the concrete
// corridor shows straight through and fights the letterforms. Blurring what is
// behind the panel keeps the translucency while giving the text a calm ground
// to sit on.
//
// The blur is free: `filter_linear_mipmap` on the screen texture means a
// higher `lod` samples a smaller mip, so there is no multi-tap loop here.
//
// Corners are rounded in the shader rather than by a StyleBox, because the
// blur has to be clipped to the same shape or it bleeds past the panel edge.

uniform sampler2D screen_tex : hint_screen_texture, filter_linear_mipmap;

// Panel size in pixels, pushed from GDScript on resize — a shader cannot ask
// its Control how big it is.
uniform vec2 panel_size = vec2(600.0, 200.0);
uniform float radius : hint_range(0.0, 40.0) = 14.0;
uniform float blur : hint_range(0.0, 6.0) = 2.6;

// Tint colour, with alpha used as "how much tint over the blur".
uniform vec4 tint : source_color = vec4(0.05, 0.06, 0.09, 0.62);
uniform float border_alpha : hint_range(0.0, 1.0) = 0.14;
uniform float border_width : hint_range(0.0, 4.0) = 1.0;

// Signed distance to a rounded rectangle: negative inside, positive outside.
float rounded_box(vec2 p, vec2 half_size, float r) {
	vec2 q = abs(p) - half_size + r;
	return min(max(q.x, q.y), 0.0) + length(max(q, vec2(0.0))) - r;
}

void fragment() {
	vec2 p = (UV - 0.5) * panel_size;
	float d = rounded_box(p, panel_size * 0.5, radius);

	// One pixel of feathering, so the edge is not stair-stepped.
	float inside = 1.0 - smoothstep(-1.0, 0.0, d);
	if (inside <= 0.001) {
		discard;
	}

	vec3 behind = textureLod(screen_tex, SCREEN_UV, blur).rgb;
	vec3 col = mix(behind, tint.rgb, tint.a);

	// A hairline rim, brightest exactly on the edge, fading inward.
	float rim = smoothstep(-border_width, 0.0, d) * inside;
	col = mix(col, vec3(1.0), rim * border_alpha);

	COLOR = vec4(col, inside);
}
