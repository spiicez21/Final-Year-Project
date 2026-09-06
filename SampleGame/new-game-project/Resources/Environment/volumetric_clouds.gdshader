shader_type sky;

// Layered cloud sky.
//
// DESIGN NOTE -- this is a projected 2.5D cloud layer, not a full volumetric
// raymarch. The march version cost 300+ noise samples per pixel and tanked the
// frame rate. This projects the view ray onto two cloud planes and shades them
// from an offset sun-side sample, which fakes self-shadowing convincingly for
// about 8 samples per pixel. Two decks at different heights and speeds restore
// most of the parallax the march was providing.
//
// All noise is PROCEDURAL on purpose. Sampler uniforms fall back to a blank
// texture if the material's parameters are ever stripped, which silently
// produces an empty sky; hash noise has no such failure mode.

uniform vec3 sky_top_color : source_color = vec3(0.24, 0.44, 0.84);
uniform vec3 sky_horizon_color : source_color = vec3(0.76, 0.85, 0.94);
uniform vec3 ground_color : source_color = vec3(0.17, 0.16, 0.15);
uniform float horizon_falloff : hint_range(0.5, 12.0) = 3.4;

uniform float sun_disc_size : hint_range(0.0, 0.2) = 0.022;
uniform float sun_glow : hint_range(1.0, 512.0) = 120.0;

uniform vec3 cloud_color : source_color = vec3(1.0, 0.99, 0.96);
uniform vec3 cloud_dark : source_color = vec3(0.44, 0.48, 0.58);
uniform float coverage : hint_range(0.0, 1.0) = 0.5;
uniform float cloud_scale : hint_range(0.1, 8.0) = 1.6;
uniform float wind_speed : hint_range(0.0, 0.2) = 0.012;
uniform float shade_offset : hint_range(0.0, 1.0) = 0.28;
uniform float silver : hint_range(0.0, 3.0) = 1.2;
uniform float upper_deck : hint_range(0.0, 1.0) = 0.45;


float hash21(vec2 p) {
	vec3 p3 = fract(vec3(p.xyx) * 0.1031);
	p3 += dot(p3, p3.yzx + 33.33);
	return fract((p3.x + p3.y) * p3.z);
}

float vnoise(vec2 p) {
	vec2 i = floor(p);
	vec2 f = fract(p);
	f = f * f * (3.0 - 2.0 * f);           // smoothstep weights
	float a = hash21(i);
	float b = hash21(i + vec2(1.0, 0.0));
	float c = hash21(i + vec2(0.0, 1.0));
	float d = hash21(i + vec2(1.0, 1.0));
	return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// 4 octaves is the sweet spot: enough structure to look like weather, cheap
// enough to run per pixel without a raymarch.
float fbm(vec2 p) {
	float v = 0.0;
	float amp = 0.5;
	for (int i = 0; i < 4; i++) {
		v += amp * vnoise(p);
		p *= 2.03;
		amp *= 0.5;
	}
	return v;
}

// One cloud deck. Returns rgb premultiplied by its own alpha, plus alpha.
vec4 deck(vec3 dir, vec3 sun_dir, vec3 sun_col, float height, float scale,
		float drift, float cover, float phase) {
	// project the view ray onto a horizontal plane at `height`
	float t = height / max(dir.y, 0.015);
	vec2 p = dir.xz * t * scale * 0.001;
	p += vec2(drift, drift * 0.4) * TIME * wind_speed;

	float d = fbm(p);
	float a = smoothstep(1.0 - cover, 1.0 - cover * 0.35, d);
	if (a <= 0.001) {
		return vec4(0.0);
	}

	// Sample again shifted towards the sun. Where the sun-side sample is
	// thinner than here, this bit of cloud is facing the light -- that
	// difference is what reads as self-shadowing and gives the decks depth.
	vec2 sp = p + sun_dir.xz * shade_offset;
	float ds = fbm(sp);
	float lit = clamp((d - ds) * 2.2 + 0.55, 0.0, 1.0);

	vec3 col = mix(cloud_dark, cloud_color, lit) * sun_col;
	// forward scattering: rim towards the sun picks up a bright edge
	col += cloud_color * phase * silver * (1.0 - lit) * 0.5;

	// fade the deck out as it approaches the horizon, where the projection
	// stretches to infinity and the noise would smear
	a *= smoothstep(0.015, 0.22, dir.y);
	return vec4(col * a, a);
}

void sky() {
	vec3 dir = EYEDIR;

	vec3 sun_dir = vec3(0.0, 1.0, 0.0);
	vec3 sun_col = vec3(1.0);
	float sun_energy = 1.0;
	if (LIGHT0_ENABLED) {
		// LIGHT0_DIRECTION is where the light points, so the sun is opposite.
		sun_dir = -LIGHT0_DIRECTION;
		sun_col = LIGHT0_COLOR;
		sun_energy = LIGHT0_ENERGY;
	}
	vec3 sun_lit = sun_col * max(sun_energy, 0.35);

	// sky gradient
	float up = clamp(dir.y, -1.0, 1.0);
	float grad = pow(clamp(up, 0.0, 1.0), 1.0 / horizon_falloff);
	vec3 col = mix(sky_horizon_color, sky_top_color, grad);
	col = mix(col, ground_color, smoothstep(0.0, -0.1, up));

	// sun disc and glow
	float cd = dot(dir, sun_dir);
	float disc = smoothstep(1.0 - sun_disc_size, 1.0 - sun_disc_size * 0.4, cd);
	float glow = pow(max(cd, 0.0), sun_glow);
	col += sun_col * (disc * 8.0 + glow * 0.9) * max(sun_energy, 0.0);

	// Henyey-Greenstein-ish forward lobe, evaluated once and shared by both decks
	float g = 0.72;
	float phase = (1.0 - g * g) / (4.0 * PI * pow(1.0 + g * g - 2.0 * g * cd, 1.5));
	phase = clamp(phase, 0.0, 2.0);

	if (dir.y > 0.0) {
		// lower deck: bigger, slower, denser
		vec4 low = deck(dir, sun_dir, sun_lit, 1200.0, cloud_scale, 1.0, coverage, phase);
		col = col * (1.0 - low.a) + low.rgb;
		// upper deck: finer and faster, adds parallax against the lower one
		if (upper_deck > 0.001) {
			vec4 high = deck(dir, sun_dir, sun_lit, 2600.0, cloud_scale * 2.3, 1.7,
				coverage * upper_deck, phase);
			col = col * (1.0 - high.a) + high.rgb;
		}
	}

	COLOR = col;
}
