@file:Suppress("UnusedImport")

package com.auto_fe.auto_fe

import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.blur
import androidx.compose.ui.draw.scale
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.*
import androidx.compose.ui.unit.dp
import com.auto_fe.auto_fe.ui.theme.*
import kotlin.math.*
import kotlin.random.Random
import kotlinx.coroutines.delay

/**
 * Enhanced 3D Aura Sphere with advanced visual effects
 * 
 * Features:
 * - True 3D rotation with accurate perspective
 * - Multi-layered particle system with depth sorting
 * - Dynamic expansion waves responding to voice input
 * - Advanced glassmorphism effects with realistic light diffusion
 * - Ultra-smooth animations with optimized transitions
 * - Harmonic color blending with dynamic luminosity
 */
@Composable
fun Rotating3DSphere(
    voiceLevel: Int,
    isNightMode: Boolean,
    modifier: Modifier = Modifier
) {
    // ==== Enhanced Color System ====
    // Base colors with better luminosity balance
    val baseA = if (isNightMode) DarkWaveColor2 else WaveColor3
    val baseB = if (isNightMode) DarkWaveColor4 else WaveColor5
    val baseC = if (isNightMode) DarkWaveColor3 else WaveColor2
    
    // HARDCODE VOICE LEVEL FOR TESTING - Remove this in production
    val testVoiceLevel = remember { mutableStateOf(0) }
    LaunchedEffect(Unit) {
        while (true) {
            delay(2000) // Change every 2 seconds
            testVoiceLevel.value = (testVoiceLevel.value + 1) % 4 // 0,1,2,3
        }
    }
    val actualVoiceLevel = testVoiceLevel.value
    
    // Dynamic accent colors with harmonic transitions
    val accent = when (actualVoiceLevel) {
        0 -> baseC
        1 -> VoiceLowColor
        2 -> VoiceMediumColor
        else -> VoiceHighColor
    }
    
    // Secondary accent for color harmony
    val accentSecondary = lerp(accent, 
                              if (isNightMode) Color.White else Color(0xFF8B4CFF), 
                              0.35f)
    
    // ==== Enhanced Animation System ====
    val transition = rememberInfiniteTransition(label = "master")

    // Breathing animation with improved natural rhythm
    val breathScale by transition.animateFloat(
        initialValue = 0.95f,  // Slightly higher min scale for better presence
        targetValue = if (actualVoiceLevel == 0) 1.07f else 1.15f,
        animationSpec = infiniteRepeatable(
            animation = tween(
                durationMillis = if (actualVoiceLevel == 0) 3800 else 1500,
                // Natural breathing curve with slight asymmetry
                easing = CubicBezierEasing(0.4f, 0.0f, 0.2f, 1.0f)
            ),
            repeatMode = RepeatMode.Reverse
        ),
        label = "breath"
    )

    // Enhanced glow intensity with subtle variation
    val glowIntensity by transition.animateFloat(
        initialValue = 0.28f,
        targetValue = if (actualVoiceLevel == 0) 0.38f else 0.52f,
        animationSpec = infiniteRepeatable(
            animation = tween(
                durationMillis = if (actualVoiceLevel == 0) 3200 else 1300,
                easing = EaseInOutCubic
            ),
            repeatMode = RepeatMode.Reverse
        ),
        label = "glow"
    )
    
    // Micro variation for subtle organic movement
    val microVar by transition.animateFloat(
        initialValue = 0f, targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(2300, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "microVar"
    )

    // Enhanced 3D Rotation with improved variability
    // X axis slower for stability
    val rotationX by transition.animateFloat(
        0f, 2f * PI.toFloat(),
        animationSpec = infiniteRepeatable(
            tween(26000, easing = LinearEasing) 
        ),
        label = "rotX"
    )

    // Y axis medium speed for primary rotation
    val rotationY by transition.animateFloat(
        0f, 2f * PI.toFloat(),
        animationSpec = infiniteRepeatable(
            tween(23000, easing = LinearEasing) 
        ),
        label = "rotY"
    )

    // Z axis faster for dynamic feel
    val rotationZ by transition.animateFloat(
        0f, 2f * PI.toFloat(),
        animationSpec = infiniteRepeatable(
            tween(30000, easing = LinearEasing) 
        ),
        label = "rotZ"
    )
    
    // Pulse wave animation
    val pulseWave by transition.animateFloat(
        initialValue = 0f, targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(
                durationMillis = 3000,
                easing = LinearEasing
            )
        ),
        label = "pulse"
    )

    // ==== Enhanced Particle System ====
    // Increased particle count for richer visual
    val MAX_COUNT = 4200  // Increased from 3400
    
    // Use Fibonacci sphere distribution with subtle variation for more natural look
    val particles = remember { 
        createEnhancedFibonacciSphere(MAX_COUNT, variation = 0.03f) 
    }
    
    // Enhanced smooth level interpolation
    val lv by animateFloatAsState(
        targetValue = (actualVoiceLevel / 3f).coerceIn(0f, 1f),
        // Spring physics for more natural response
        animationSpec = spring(
            dampingRatio = Spring.DampingRatioLowBouncy,
            stiffness = Spring.StiffnessLow
        ),
        label = "lv"
    )

    // ==== Enhanced Expansion System with Wave Propagation ====
    // Track expansion points with improved wave physics
    var expansionPoints by remember { mutableStateOf<List<EnhancedExpansionPoint>>(emptyList()) }
    var band by remember { mutableStateOf(0) }
    
    // Add ripple effect system
    var ripples by remember { mutableStateOf<List<RippleWave>>(emptyList()) }

    // Improved hysteresis bands with smoother transitions
    fun bandOf(x: Float): Int = when {
        x < 0.28f -> 0
        x < 0.62f -> 1
        else -> 2
    }

    // Enhanced expansion point creation with better spatial distribution
    fun spawnExpansionPoints(b: Int) {
        // Create new expansion points with varied parameters
        val count = 2 + b * 3  // 2,5,8 - increased for higher levels
        expansionPoints = List(count) {
            EnhancedExpansionPoint(
                theta = Random.nextFloat() * 2f * PI.toFloat(),
                phi = Random.nextFloat() * PI.toFloat(),
                strength = 0.6f + Random.nextFloat() * 0.4f,  // Varied strength
                startTime = System.currentTimeMillis(),
                // More varied durations for natural feel
                duration = 800L + Random.nextLong(800L)
            )
        }
        
        // Spawn a ripple wave with each level change
        if (b > 0) {
            ripples = ripples + RippleWave(
                startTime = System.currentTimeMillis(),
                duration = 1200L + b * 400L,
                strength = 0.4f + b * 0.3f
            )
        }
    }

    // Debounce expansion point creation for smoother transitions
    LaunchedEffect(lv) {
        val newBand = bandOf(lv)
        if (newBand != band) {
            band = newBand
            // Slightly longer delay for better debouncing
            kotlinx.coroutines.delay(150)
            spawnExpansionPoints(newBand)
        }
    }

    // Continuous clean up of expired effects
    LaunchedEffect(Unit) {
        while (true) {
            kotlinx.coroutines.delay(100)
            val now = System.currentTimeMillis()
            
            // Clean expired expansion points
            expansionPoints = expansionPoints.filter {
                now - it.startTime < it.duration
            }
            
            // Clean expired ripples
            ripples = ripples.filter {
                now - it.startTime < it.duration
            }
        }
    }
    
    // Generate ambient particles for outer atmosphere
    val ambientParticles = remember {
        List(80) {
            AmbientParticle(
                radius = 0.6f + Random.nextFloat() * 0.6f,
                theta = Random.nextFloat() * 2f * PI.toFloat(),
                phi = Random.nextFloat() * PI.toFloat(),
                size = 0.8f + Random.nextFloat() * 1.2f,
                speed = 0.2f + Random.nextFloat() * 0.3f
            )
        }
    }

    Box(modifier = modifier.fillMaxSize()) {
        // Layer 1: Enhanced Background Glow with improved light diffusion
        Canvas(
            modifier = Modifier
                .fillMaxSize()
                .alpha(1.15f)
                .blur(if (isNightMode) 32.dp else 40.dp)
        ) {
            // Draw enhanced background glow with dynamic color and size
            drawEnhancedBackgroundGlow(
                center = center,
                baseRadius = size.minDimension * 0.38f,
                scale = breathScale,
                colors = listOf(baseA, baseB, accent, accentSecondary),
                intensity = glowIntensity,
                isNightMode = isNightMode,
                pulseWave = pulseWave,
                voiceLevel = actualVoiceLevel
            )
            
            // Draw ripple waves if present
            ripples.forEach { ripple ->
                val progress = ((System.currentTimeMillis() - ripple.startTime).toFloat() / 
                               ripple.duration).coerceIn(0f, 1f)
                
                // Eased progress for natural wave motion
                val easedProgress = Easing.EaseOutCubic.transform(progress)
                
                // Ripple radius grows outward
                val rippleRadius = size.minDimension * 0.38f * 
                                   (0.4f + easedProgress * 1.5f)
                
                // Fade out as it expands
                val rippleAlpha = (1f - easedProgress) * ripple.strength * 0.4f
                
                drawCircle(
                    brush = Brush.radialGradient(
                        colors = listOf(
                            Color.Transparent,
                            accent.copy(alpha = rippleAlpha * 0.7f),
                            Color.Transparent
                        ),
                        center = center,
                        radius = rippleRadius
                    ),
                    radius = rippleRadius,
                    center = center
                )
            }
        }

        // Layer 2: Enhanced Glassmorphism Core with improved light refraction
        Canvas(
            modifier = Modifier
                .fillMaxSize()
                .alpha(1.1f)
                .blur(6.dp)
                .scale(breathScale * (1f + microVar * 0.01f))  // Subtle micro-variation
        ) {
            drawEnhancedGlassmorphismCore(
                center = center,
                radius = size.minDimension * 0.38f,
                colors = listOf(baseA, baseB, accent, accentSecondary),
                intensity = glowIntensity,
                isNightMode = isNightMode,
                voiceLevel = actualVoiceLevel,
                microVariation = microVar
            )
        }
        
        // Layer 2.5: Ambient atmosphere particles for depth
        Canvas(
            modifier = Modifier
                .fillMaxSize()
                .alpha(0.7f)
                .scale(breathScale * (1f + microVar * 0.02f))
        ) {
            val c = center
            val R = size.minDimension * 0.38f
            val now = System.currentTimeMillis()
            
            ambientParticles.forEach { particle ->
                // Animate position over time
                val timeOffset = (now / 1000f * particle.speed) % (2f * PI.toFloat())
                
                val theta = particle.theta + timeOffset
                
                // Convert to Cartesian coordinates
                val x = particle.radius * sin(particle.phi) * cos(theta)
                val y = particle.radius * cos(particle.phi)
                val z = particle.radius * sin(particle.phi) * sin(theta)
                
                // Apply 3D rotation
                val (rx, ry, rz) = rotate3D(x, y, z, rotationX, rotationY, rotationZ)
                
                // Apply perspective projection
                val perspective = 1.5f + 0.5f * rz
                val px = c.x + (rx * R * 1.2f) / perspective
                val py = c.y + (ry * R * 1.2f) / perspective
                
                // Calculate opacity based on z position (depth)
                val depth = ((rz + 1f) * 0.5f).coerceIn(0f, 1f)
                val particleAlpha = 0.15f + 0.3f * depth * (0.6f + 0.4f * lv)
                
                // Draw ambient particle with glow
                val particleSize = particle.size * (0.8f + depth * 0.4f) * 
                                  (0.8f + 0.4f * lv)
                
                drawCircle(
                    brush = Brush.radialGradient(
                        colors = listOf(
                            Color.White.copy(alpha = particleAlpha),
                            accent.copy(alpha = particleAlpha * 0.7f),
                            Color.Transparent
                        ),
                        center = Offset(px, py),
                        radius = particleSize * 2f
                    ),
                    radius = particleSize,
                    center = Offset(px, py)
                )
            }
        }

        // Layer 3: Enhanced 3D Particles with improved depth and lighting
        Canvas(
            modifier = Modifier
                .fillMaxSize()
                .alpha(1.05f)
                .scale(breathScale * (1f + microVar * 0.01f))
        ) {
            val c = center
            val R = size.minDimension * 0.38f
            val now = System.currentTimeMillis()

            // Enhanced density calculation - smoother transition
            val density = 0.6f + 0.4f * lv      // 60%..100%
            val visibleCount = (MAX_COUNT * density).toInt()
            
            // Transform, enhance, and sort particles by depth
            val transformedParticles = particles.take(visibleCount).map { point ->
                // Apply 3D rotation with micro-variation
                val (x, y, z) = rotate3D(
                    point[0], point[1], point[2],
                    rotationX, rotationY + point[3] * 0.05f, rotationZ
                )

                // Calculate expansion with improved wave physics
                val expansionFactor = calculateEnhancedExpansion(
                    x, y, z, expansionPoints, now
                )
                
                // Calculate distance from center for radial effects
                val distFromCenter = sqrt(x*x + y*y + z*z)

                EnhancedParticle(
                    x = x, y = y, z = z,
                    expansion = expansionFactor,
                    distFromCenter = distFromCenter,
                    variation = point[3]  // Use 4th value for variation
                )
            }.sortedBy { it.z } // Sort by depth (back to front)

            // Draw enhanced particles with improved visual quality
            transformedParticles.forEach { p ->
                // Apply dynamic expansion based on voice level
                val expandFactor = 1f + p.expansion * (0.3f + 0.2f * lv)
                val expandedX = p.x * expandFactor
                val expandedY = p.y * expandFactor
                val expandedZ = p.z * expandFactor

                // Enhanced perspective with subtle depth cues
                val perspective = 1.6f + 0.4f * expandedZ
                val px = c.x + (expandedX * R) / perspective
                val py = c.y + (expandedY * R) / perspective

                // Enhanced depth mapping with smoother gradients
                val depth = ((expandedZ + 1f) * 0.5f).coerceIn(0f, 1f)
                
                // Enhanced particle size with natural variation
                val baseSizePx = (0.9f + 2.2f * depth) * 
                                 (0.9f + 0.4f * lv) * 
                                 (1f + p.expansion * 0.9f) *
                                 (0.85f + p.variation * 0.3f) // Natural variation
                
                // Micro-variation for organic feel
                val sizePx = baseSizePx * (1f + sin(p.variation * 10f + microVar * 6.28f) * 0.05f)

                // Enhanced color mixing with better luminosity
                val colorMix = 0.3f + 0.7f * depth  
                val expansionGlow = p.expansion * 0.85f
                
                // Radial glow effect for particles near center
                val coreEffect = (1f - p.distFromCenter.coerceIn(0f, 1f)) * 0.5f
                
                // Dynamic alpha based on depth and voice level
                val particleAlpha = (0.4f + 0.6f * depth) * (0.85f + 0.3f * lv)
                
                // Base color with improved blending
                val baseColor = lerp(
                    lerp(baseB, accent, colorMix),
                    Color.White,
                    expansionGlow + coreEffect
                ).copy(alpha = particleAlpha)
                
                // Enhanced particle rendering with inner glow
                drawCircle(
                    color = baseColor,
                    radius = sizePx,
                    center = Offset(px, py)
                )
                
                // Add subtle inner highlight for particles near expansion points
                if (p.expansion > 0.2f) {
                    drawCircle(
                        color = Color.White.copy(
                            alpha = p.expansion * 0.3f * particleAlpha
                        ),
                        radius = sizePx * 0.4f,
                        center = Offset(px, py)
                    )
                }
            }

            // Enhanced nucleus with improved energy visualization
            drawEnhancedNucleus(
                center = c,
                radius = R,
                accent = accent,
                accentSecondary = accentSecondary,
                baseA = baseA,
                intensity = glowIntensity,
                voiceLevel = actualVoiceLevel,
                microVar = microVar,
                pulseWave = pulseWave
            )
        }

        // Layer 4: Enhanced Expansion Point Highlights with improved glow
        if (expansionPoints.isNotEmpty()) {
            Canvas(
                modifier = Modifier
                    .fillMaxSize()
                    .alpha(0.9f)
                    .blur(14.dp)
                    .scale(breathScale)
            ) {
                val c = center
                val R = size.minDimension * 0.38f
                val now = System.currentTimeMillis()

                // Draw each expansion point with enhanced visual effects
                expansionPoints.forEach { ep ->
                    val progress = ((now - ep.startTime).toFloat() / ep.duration).coerceIn(0f, 1f)
                    
                    // Improved easing for expansion animation
                    val easedProgress = Easing.EaseOutQuad.transform(progress)
                    
                    // Fade based on strength and progress
                    val alpha = (1f - easedProgress) * 0.6f * ep.strength

                    // Convert spherical to 3D coordinates
                    val x = sin(ep.phi) * cos(ep.theta)
                    val y = cos(ep.phi)
                    val z = sin(ep.phi) * sin(ep.theta)

                    // Apply 3D rotation
                    val (rx, ry, rz) = rotate3D(x, y, z, rotationX, rotationY, rotationZ)

                    // Enhanced projection with better perspective
                    val perspective = 1.5f + 0.5f * rz
                    val px = c.x + (rx * R * 1.3f) / perspective
                    val py = c.y + (ry * R * 1.3f) / perspective

                    // Improved glow with color blending
                    val baseRadius = R * 0.15f * (1f + easedProgress * 0.8f) * ep.strength
                    
                    // Draw primary glow
                    drawCircle(
                        brush = Brush.radialGradient(
                            colors = listOf(
                                Color.White.copy(alpha = alpha * 0.7f),
                                accent.copy(alpha = alpha * 0.8f),
                                Color.Transparent
                            ),
                            center = Offset(px, py),
                            radius = baseRadius
                        ),
                        radius = baseRadius,
                        center = Offset(px, py)
                    )
                    
                    // Draw secondary outer glow for depth
                    drawCircle(
                        brush = Brush.radialGradient(
                            colors = listOf(
                                Color.Transparent,
                                accentSecondary.copy(alpha = alpha * 0.4f),
                                Color.Transparent
                            ),
                            center = Offset(px, py),
                            radius = baseRadius * 1.8f
                        ),
                        radius = baseRadius * 1.8f,
                        center = Offset(px, py)
                    )
                }
            }
        }
    }
}

// ==== Enhanced Data Classes ====

data class EnhancedExpansionPoint(
    val theta: Float,    // 0 to 2π
    val phi: Float,      // 0 to π
    val strength: Float, // 0 to 1 - varies intensity
    val startTime: Long,
    val duration: Long
)

data class EnhancedParticle(
    val x: Float,
    val y: Float,
    val z: Float,
    val expansion: Float,       // 0 to 1
    val distFromCenter: Float,  // 0 to 1
    val variation: Float        // Natural variation parameter
)

data class RippleWave(
    val startTime: Long,
    val duration: Long,
    val strength: Float  // 0 to 1
)

data class AmbientParticle(
    val radius: Float,   // Distance from center
    val theta: Float,    // Angular position
    val phi: Float,      // Angular position
    val size: Float,     // Particle size
    val speed: Float     // Movement speed
)

// ==== Enhanced Helper Functions ====

private fun createEnhancedFibonacciSphere(count: Int, variation: Float = 0.02f): List<FloatArray> {
    return List(count) { i ->
        val t = i.toFloat() / (count - 1).coerceAtLeast(1)
        val y = 1f - 2f * t
        val r = sqrt(max(0f, 1f - y * y))
        val theta = PI.toFloat() * (1f + sqrt(5f)) * i
        
        // Add subtle variation for more organic look
        val vr = 1f + (Random.nextFloat() * 2f - 1f) * variation
        val x = r * cos(theta) * vr
        val z = r * sin(theta) * vr
        
        // Add 4th parameter for natural variation
        val naturalVar = Random.nextFloat()
        
        floatArrayOf(x, y, z, naturalVar)
    }
}

private fun rotate3D(
    x: Float, y: Float, z: Float,
    rotX: Float, rotY: Float, rotZ: Float
): Triple<Float, Float, Float> {
    // Rotation around X axis
    var x1 = x
    var y1 = y * cos(rotX) - z * sin(rotX)
    var z1 = y * sin(rotX) + z * cos(rotX)

    // Rotation around Y axis
    val x2 = x1 * cos(rotY) + z1 * sin(rotY)
    val y2 = y1
    val z2 = -x1 * sin(rotY) + z1 * cos(rotY)

    // Rotation around Z axis
    val x3 = x2 * cos(rotZ) - y2 * sin(rotZ)
    val y3 = x2 * sin(rotZ) + y2 * cos(rotZ)
    val z3 = z2

    return Triple(x3, y3, z3)
}

private fun calculateEnhancedExpansion(
    x: Float, y: Float, z: Float,
    expansionPoints: List<EnhancedExpansionPoint>,
    currentTime: Long
): Float {
    if (expansionPoints.isEmpty()) return 0f

    var maxExpansion = 0f

    expansionPoints.forEach { ep ->
        // Convert expansion point to Cartesian
        val epX = sin(ep.phi) * cos(ep.theta)
        val epY = cos(ep.phi)
        val epZ = sin(ep.phi) * sin(ep.theta)

        // Calculate distance
        val dx = x - epX
        val dy = y - epY
        val dz = z - epZ
        val distance = sqrt(dx * dx + dy * dy + dz * dz)

        // Enhanced wave physics with improved falloff
        val progress = ((currentTime - ep.startTime).toFloat() / ep.duration).coerceIn(0f, 1f)
        
        // Improved easing curve for more natural wave propagation
        val timeEffect = 1f - Easing.EaseOutQuint.transform(progress)
        
        // Improved distance falloff with smoother gradient
        val falloffThreshold = 0.65f * ep.strength
        val distanceEffect = max(0f, 1f - distance / falloffThreshold)
        val smoothDistEffect = Easing.EaseOutCubic.transform(distanceEffect)
        
        // Final expansion with strength factor
        val expansion = timeEffect * smoothDistEffect * ep.strength
        maxExpansion = max(maxExpansion, expansion)
    }

    return maxExpansion
}

private object Easing {
    val EaseOutCubic = CubicBezierEasing(0.33f, 1f, 0.68f, 1f)
    val EaseOutQuad = CubicBezierEasing(0.25f, 0.46f, 0.45f, 0.94f)
    val EaseOutQuint = CubicBezierEasing(0.23f, 1f, 0.32f, 1f)
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawEnhancedBackgroundGlow(
    center: Offset,
    baseRadius: Float,
    scale: Float,
    colors: List<Color>,
    intensity: Float,
    isNightMode: Boolean,
    pulseWave: Float,
    voiceLevel: Int
) {
    val R = baseRadius * scale
    
    // Pulse wave effect for more dynamic background
    val pulseOffset = kotlin.math.sin(pulseWave * 2f * kotlin.math.PI).toFloat() * 0.05f * voiceLevel
    val pulseScale = 1f + pulseOffset

    // Outermost aura with subtle color transition
    drawCircle(
        brush = Brush.radialGradient(
            colors = listOf(
                lerp(colors[0], colors[2], 0.15f).copy(alpha = intensity * 0.5f),
                lerp(colors[1], colors[3], 0.08f).copy(alpha = intensity * 0.25f),
                Color.Transparent
            ),
            center = center,
            radius = R * 2.2f * pulseScale
        ),
        radius = R * 2.2f * pulseScale,
        center = center
    )

    // Middle aura with improved color blending
    drawCircle(
        brush = Brush.radialGradient(
            colors = listOf(
                lerp(colors[1], colors[2], 0.3f).copy(alpha = intensity * 0.8f),
                lerp(colors[2], colors[3], 0.2f).copy(alpha = intensity * 0.4f),
                Color.Transparent
            ),
            center = center,
            radius = R * 1.5f * pulseScale
        ),
        radius = R * 1.5f * pulseScale,
        center = center
    )

    // Inner aura with voice-responsive glow
    drawCircle(
        brush = Brush.radialGradient(
            colors = listOf(
                lerp(colors[2], Color.White, 0.2f).copy(
                    alpha = intensity * (0.9f + pulseOffset * 0.4f)
                ),
                lerp(colors[1], colors[2], 0.6f).copy(
                    alpha = intensity * (0.5f + pulseOffset * 0.2f)
                ),
                Color.Transparent
            ),
            center = center,
            radius = R * 1.1f * pulseScale
        ),
        radius = R * 1.1f * pulseScale,
        center = center
    )
    
    // Additional energy halo for higher voice levels
    if (voiceLevel > 1) {
        drawCircle(
            brush = Brush.radialGradient(
                colors = listOf(
                    Color.Transparent,
                    colors[2].copy(alpha = intensity * 0.2f * voiceLevel / 3f),
                    colors[3].copy(alpha = intensity * 0.4f * voiceLevel / 3f),
                    Color.Transparent
                ),
                center = center,
                radius = R * 1.8f * (0.9f + 0.2f * kotlin.math.sin(pulseWave * 4f * kotlin.math.PI).toFloat())
            ),
            radius = R * 1.8f * (0.9f + 0.2f * kotlin.math.sin(pulseWave * 4f * kotlin.math.PI).toFloat()),
            center = center
        )
    }
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawEnhancedGlassmorphismCore(
    center: Offset,
    radius: Float,
    colors: List<Color>,
    intensity: Float,
    isNightMode: Boolean,
    voiceLevel: Int,
    microVariation: Float
) {
    // Calculate dynamic variations
    val microOffset = sin(microVariation * 2f * PI.toFloat()) * 0.01f
    
    // Outer glass layer with improved refraction
    drawCircle(
        brush = Brush.radialGradient(
            colors = listOf(
                colors[2].copy(alpha = if (isNightMode) 0.18f else 0.12f),
                lerp(colors[0], colors[1], 0.5f).copy(
                    alpha = if (isNightMode) 0.1f else 0.06f
                ),
                Color.Transparent
            ),
            center = center,
            radius = radius * (0.9f + microOffset)
        ),
        radius = radius * (0.9f + microOffset),
        center = center
    )

    // Enhanced inner glass highlight with dynamic positioning
    val highlightOffset = Offset(
        center.x - radius * (0.2f + microOffset * 0.5f),
        center.y - radius * (0.2f + microOffset * 0.5f)
    )
    
    drawCircle(
        brush = Brush.radialGradient(
            colors = listOf(
                Color.White.copy(alpha = if (isNightMode) 0.11f else 0.16f),
                Color.Transparent
            ),
            center = highlightOffset,
            radius = radius * (0.4f - microOffset * 0.2f)
        ),
        radius = radius * (0.4f - microOffset * 0.2f),
        center = highlightOffset
    )
    
    // Secondary highlight for depth
    if (voiceLevel > 0) {
        val secondaryOffset = Offset(
            center.x + radius * 0.15f * microOffset,
            center.y + radius * 0.15f * microOffset
        )
        
        drawCircle(
            brush = Brush.radialGradient(
                colors = listOf(
                    colors[3].copy(alpha = 0.08f * voiceLevel / 3f),
                    Color.Transparent
                ),
                center = secondaryOffset,
                radius = radius * 0.3f
            ),
            radius = radius * 0.3f,
            center = secondaryOffset
        )
    }
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawEnhancedNucleus(
    center: Offset,
    radius: Float,
    accent: Color,
    accentSecondary: Color,
    baseA: Color,
    intensity: Float,
    voiceLevel: Int,
    microVar: Float,
    pulseWave: Float
) {
    // Dynamic pulse effect
    val pulse = 0.4f + 0.6f * (0.5f + 0.5f * kotlin.math.sin(pulseWave * 2f * kotlin.math.PI).toFloat())
    val microPulse = 0.8f + 0.2f * kotlin.math.sin(microVar * 8f * kotlin.math.PI).toFloat()
    
    // Expanded voice level response
    val voiceResponse = 0.6f + 0.4f * voiceLevel / 3f
    
    // Outer nucleus glow with improved energy visualization
    drawCircle(
        brush = Brush.radialGradient(
            colors = listOf(
                lerp(accent, Color.White, 0.3f).copy(
                    alpha = intensity * 0.6f * pulse * voiceResponse
                ),
                lerp(accent, accentSecondary, 0.3f).copy(
                    alpha = intensity * 0.35f * pulse
                ),
                accent.copy(alpha = intensity * 0.2f),
                Color.Transparent
            ),
            center = center,
            radius = radius * 0.48f * microPulse
        ),
        radius = radius * 0.48f * microPulse,
        center = center
    )

    // Enhanced inner core with dynamic energy
    drawCircle(
        brush = Brush.radialGradient(
            colors = listOf(
                Color.White.copy(alpha = intensity * 0.4f * pulse),
                lerp(accent, Color.White, 0.5f).copy(alpha = intensity * 0.25f * pulse),
                accentSecondary.copy(alpha = intensity * 0.15f),
                Color.Transparent
            ),
            center = center,
            radius = radius * 0.25f * microPulse
        ),
        radius = radius * 0.25f * microPulse,
        center = center
    )

    // Dynamic energy center
    if (voiceLevel > 0) {
        // Primary core
        drawCircle(
            brush = Brush.radialGradient(
                colors = listOf(
                    Color.White.copy(alpha = intensity * 0.85f * pulse),
                    lerp(accent, Color.White, 0.7f).copy(
                        alpha = intensity * 0.7f * pulse
                    ),
                    Color.Transparent
                ),
                center = center,
                radius = (1.4f + voiceLevel * 0.6f) * microPulse
            ),
            radius = (1.4f + voiceLevel * 0.6f) * microPulse,
            center = center
        )
        
        // Inner core highlight
        drawCircle(
            color = Color.White.copy(
                alpha = intensity * 0.9f * pulse * voiceResponse
            ),
            radius = (0.7f + voiceLevel * 0.3f) * microPulse,
            center = center
        )
    }
    
    // Add energy rays for high voice levels
    if (voiceLevel > 1) {
        val rayCount = 6 + voiceLevel * 2
        val rayAngleStep = 2f * kotlin.math.PI / rayCount
        
        // Calculate ray properties
        val rayLength = radius * (0.1f + 0.1f * voiceLevel / 3f) * pulse
        val rayWidth = 1.5f + voiceLevel * 0.5f
        
        // Add rotation for dynamic effect
        val rayRotation = microVar * kotlin.math.PI * 2f
        
        for (i in 0 until rayCount) {
            val angle = i * rayAngleStep + rayRotation
            
            val startX = center.x
            val startY = center.y
            val endX = center.x + kotlin.math.cos(angle).toFloat() * rayLength
            val endY = center.y + kotlin.math.sin(angle).toFloat() * rayLength
            
            // Draw ray with gradient
            drawLine(
                brush = Brush.linearGradient(
                    colors = listOf(
                        Color.White.copy(alpha = 0.6f * pulse),
                        accent.copy(alpha = 0.3f * pulse),
                        Color.Transparent
                    ),
                    start = androidx.compose.ui.geometry.Offset(startX, startY),
                    end = androidx.compose.ui.geometry.Offset(endX, endY)
                ),
                start = androidx.compose.ui.geometry.Offset(startX, startY),
                end = androidx.compose.ui.geometry.Offset(endX, endY),
                strokeWidth = rayWidth,
                cap = StrokeCap.Round
            )
        }
    }
}

/**
 * Utility extension for linear interpolation between colors.
 */
private fun lerp(start: Color, stop: Color, fraction: Float): Color {
    return Color(
        red = start.red + (stop.red - start.red) * fraction,
        green = start.green + (stop.green - start.green) * fraction,
        blue = start.blue + (stop.blue - start.blue) * fraction,
        alpha = start.alpha + (stop.alpha - start.alpha) * fraction
    )
}