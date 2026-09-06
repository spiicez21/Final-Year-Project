shader_type spatial;
render_mode unshaded, blend_add, depth_test_disabled, depth_draw_never, cull_disabled, fog_disabled;

// Screen-space crepuscular rays.
//
// Marches from each pixel back towards the sun's screen position, accumulating
// wherever the depth buffer says "sky". Geometry between the pixel and the sun
// therefore carves shadow into the streaks for free -- no extra shadow pass.
//
// This is a spatial (not canvas_item) shader because it needs the depth buffer.
// The vertex stage rewrites POSITION directly so the mesh covers the screen
// regardless of where the node actually sits in the world.
//
// This handles the sun-in-frame case. Shafts through windows and doorways come
// from the volumetric fog instead, which is genuinely 3D and keeps working when
// the sun itself is off screen.

uniform sampler2D depth_tex : hint_depth_texture, filter_nearest;
uniform vec2 sun_pos = vec2(0.5, 0.5);
uniform float ray_density : hint_range(0.1, 1.5) = 0.62;
uniform float decay : hint_range(0.8, 1.0) = 0.965;
uniform float weight : hint_range(0.0, 2.0) = 0.5;
uniform float exposure : hint_range(0.0, 2.0) = 0.42;
uniform vec3 tint : source_color = vec3(1.0, 0.88, 0.68);
/// Faded by the driver script when the sun is behind or off screen.
uniform float visibility : hint_range(0.0, 1.0) = 0.0;

const int SAMPLES = 48;

float hash12(vec2 p) {
	vec3 p3 = fract(vec3(p.xyx) * 0.1031);
	p3 += dot(p3, p3.yzx + 33.33);
	return fract((p3.x + p3.y) * p3.z);
}

void vertex() {
	// full-screen triangle/quad, independent of the node's transform
	POSITION = vec4(VERTEX.xy * 2.0, 1.0, 1.0);
}

void fragment() {
	if (visibility <= 0.001) {
		ALBEDO = vec3(0.0);
	} else {
		vec2 uv = SCREEN_UV;
		vec2 delta = (uv - sun_pos) * (ray_density / float(SAMPLES));
		// Jitter the start so the march does not band into visible rings.
		vec2 coord = uv - delta * hash12(uv * 917.0);

		float illum = 1.0;
		float accum = 0.0;
		for (int i = 0; i < SAMPLES; i++) {
			coord -= delta;
			// Godot uses reverse-Z, so untouched pixels (the sky) sit at 0.
			float d = texture(depth_tex, clamp(coord, vec2(0.0), vec2(1.0))).r;
			accum += step(d, 0.000001) * illum;
			illum *= decay;
		}
		accum = accum / float(SAMPLES) * weight;

		// Fade towards the frame edge so nothing pops as the sun leaves view.
		float edge = smoothstep(1.35, 0.35, length(uv - sun_pos));
		ALBEDO = tint * accum * exposure * visibility * edge;
	}
}
