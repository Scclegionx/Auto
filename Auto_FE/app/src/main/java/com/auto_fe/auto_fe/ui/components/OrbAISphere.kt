@file:Suppress("UnusedImport")

package com.auto_fe.auto_fe.ui.components

import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.clickable
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

// ======= SAFETY UTILITIES =======
private inline fun Float.safeAlpha() = this.coerceIn(0.04f, 0.98f)
private inline fun Float.safeStroke() = this.coerceIn(0.8f, 12f)

/** radius dùng để VẼ (stroke/solid) */
private inline fun Float.safeDrawRadius(min: Float = 0.75f) = this.coerceAtLeast(min)

/** radius dùng cho BRUSH.radialGradient (shader yêu cầu >0) */
private inline fun Float.safeShaderRadius(min: Float = 1.2f) = this.coerceAtLeast(min)

// ==== SATURN RINGS DRAWING FUNCTION ====
private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawSaturnRings(
    center: Offset,
    sphereRadius: Float,
    ringRotation1: Float,
    ringRotation2: Float,
    ringRotation3: Float,
    isDarkMode: Boolean,
    voiceIntensity: Float
) {
    val ringColors = if (isDarkMode) {
        listOf(
            Color(0xFFFFD700), // GOLD - Ultra bright ring
            Color(0xFFFF8C00), // DARK ORANGE - Vibrant ring
            Color(0xFFFF4500)  // ORANGE RED - Intense ring
        )
    } else {
        listOf(
            Color(0xFFE91E63), // PINK - Ultra bright ring
            Color(0xFF9C27B0), // PURPLE - Vibrant ring
            Color(0xFF673AB7)  // DEEP PURPLE - Intense ring
        )
    }
    
    // Ring 1: Horizontal ring (consistent plane) - ULTRA VIBRANT
    val ring1Radius = sphereRadius * 1.4f
    val ring1Thickness = (8f + voiceIntensity * 5f).safeStroke() // MUCH THICKER
    val ring1Alpha = (0.8f + voiceIntensity * 0.3f).safeAlpha() // MUCH BRIGHTER
    
    // Use consistent angle units (radians throughout calculation, degrees for drawing)
    val ring1StartAngle = (ringRotation1 * 57.3f) % 360f
    
    // Draw complete ellipse with height reduced to create perspective
    drawOval(
        color = ringColors[0].copy(alpha = ring1Alpha),
        topLeft = Offset(
            center.x - ring1Radius,
            center.y - ring1Radius * 0.3f // Flatten for perspective
        ),
        size = androidx.compose.ui.geometry.Size(
            width = ring1Radius * 2f, 
            height = ring1Radius * 0.6f // Flatten for perspective
        ),
        style = androidx.compose.ui.graphics.drawscope.Stroke(width = ring1Thickness)
    )
    
    // Ring 2: Tilted 30 degrees (middle ring) - ULTRA VIBRANT
    val ring2Radius = sphereRadius * 1.6f
    val ring2Thickness = (7f + voiceIntensity * 4f).safeStroke() // MUCH THICKER
    val ring2Alpha = (0.7f + voiceIntensity * 0.25f).safeAlpha() // MUCH BRIGHTER
    
    val ring2StartAngle = (ringRotation2 * 57.3f) % 360f
    
    // Ring 2: Tilted 30 degrees (middle ring) - Simplified approach
    val ring2CenterX = center.x + (ring2Radius * 0.3f * cos(ringRotation2))
    val ring2CenterY = center.y + (ring2Radius * 0.15f * sin(ringRotation2))
    
    drawOval(
        color = ringColors[1].copy(alpha = ring2Alpha),
        topLeft = Offset(
            ring2CenterX - ring2Radius,
            ring2CenterY - ring2Radius * 0.25f
        ),
        size = androidx.compose.ui.geometry.Size(
            width = ring2Radius * 2f, 
            height = ring2Radius * 0.5f
        ),
        style = androidx.compose.ui.graphics.drawscope.Stroke(width = ring2Thickness)
    )
    
    // Ring 3: Tilted -45 degrees (outer ring) - ULTRA VIBRANT
    val ring3Radius = sphereRadius * 1.8f
    val ring3Thickness = (6f + voiceIntensity * 3f).safeStroke() // MUCH THICKER
    val ring3Alpha = (0.6f + voiceIntensity * 0.2f).safeAlpha() // MUCH BRIGHTER
    
    val ring3StartAngle = (ringRotation3 * 57.3f) % 360f
    
    // Ring 3: Tilted -45 degrees (outer ring) - Simplified approach
    val ring3CenterX = center.x + (ring3Radius * 0.2f * cos(ringRotation3 + PI.toFloat() / 4f))
    val ring3CenterY = center.y + (ring3Radius * 0.1f * sin(ringRotation3 + PI.toFloat() / 4f))
    
    drawOval(
        color = ringColors[2].copy(alpha = ring3Alpha),
        topLeft = Offset(
            ring3CenterX - ring3Radius,
            ring3CenterY - ring3Radius * 0.2f
        ),
        size = androidx.compose.ui.geometry.Size(
            width = ring3Radius * 2f, 
            height = ring3Radius * 0.4f
        ),
        style = androidx.compose.ui.graphics.drawscope.Stroke(width = ring3Thickness)
    )
}

// ==== SATURN RINGS BEHIND SPHERE ====
private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawSaturnRingsBehind(
    center: Offset,
    sphereRadius: Float,
    ringRotation1: Float,
    ringRotation2: Float,
    ringRotation3: Float,
    isDarkMode: Boolean,
    voiceIntensity: Float
) {
    val ringColors = if (isDarkMode) {
        listOf(
            Color(0x60FFD700), // GOLD - Behind sphere (dimmer)
            Color(0x50FF8C00), // DARK ORANGE - Behind sphere
            Color(0x40FF4500)  // ORANGE RED - Behind sphere
        )
    } else {
        listOf(
            Color(0x70E91E63), // PINK - Behind sphere (dimmer)
            Color(0x609C27B0), // PURPLE - Behind sphere
            Color(0x50673AB7)  // DEEP PURPLE - Behind sphere
        )
    }
    
    // Ring 1: Behind sphere (bottom half only)
    val ring1Radius = sphereRadius * 1.4f
    val ring1Thickness = (6f + voiceIntensity * 3f).safeStroke()
    val ring1Alpha = (0.4f + voiceIntensity * 0.2f).safeAlpha()
    
    // Only draw bottom half (behind sphere)
    drawArc(
        color = ringColors[0].copy(alpha = ring1Alpha),
        startAngle = 0f,
        sweepAngle = 180f, // Bottom half only
        useCenter = false,
        topLeft = Offset(
            center.x - ring1Radius,
            center.y - ring1Radius * 0.3f
        ),
        size = androidx.compose.ui.geometry.Size(
            width = ring1Radius * 2f, 
            height = ring1Radius * 0.6f
        ),
        style = androidx.compose.ui.graphics.drawscope.Stroke(width = ring1Thickness)
    )
    
    // Ring 2: Behind sphere (bottom portion)
    val ring2Radius = sphereRadius * 1.6f
    val ring2Thickness = (5f + voiceIntensity * 2f).safeStroke()
    val ring2Alpha = (0.3f + voiceIntensity * 0.15f).safeAlpha()
    
    val ring2CenterX = center.x + (ring2Radius * 0.2f * cos(ringRotation2))
    val ring2CenterY = center.y + (ring2Radius * 0.1f * sin(ringRotation2))
    
    // Only draw bottom portion (behind sphere)
    drawArc(
        color = ringColors[1].copy(alpha = ring2Alpha),
        startAngle = 0f,
        sweepAngle = 120f, // Bottom portion only
        useCenter = false,
        topLeft = Offset(
            ring2CenterX - ring2Radius,
            ring2CenterY - ring2Radius * 0.25f
        ),
        size = androidx.compose.ui.geometry.Size(
            width = ring2Radius * 2f, 
            height = ring2Radius * 0.5f
        ),
        style = androidx.compose.ui.graphics.drawscope.Stroke(width = ring2Thickness)
    )
    
    // Ring 3: Behind sphere (bottom portion)
    val ring3Radius = sphereRadius * 1.8f
    val ring3Thickness = (4f + voiceIntensity * 2f).safeStroke()
    val ring3Alpha = (0.2f + voiceIntensity * 0.1f).safeAlpha()
    
    val ring3CenterX = center.x + (ring3Radius * 0.15f * cos(ringRotation3 + PI.toFloat() / 4f))
    val ring3CenterY = center.y + (ring3Radius * 0.08f * sin(ringRotation3 + PI.toFloat() / 4f))
    
    // Only draw bottom portion (behind sphere)
    drawArc(
        color = ringColors[2].copy(alpha = ring3Alpha),
        startAngle = 0f,
        sweepAngle = 100f, // Bottom portion only
        useCenter = false,
        topLeft = Offset(
            ring3CenterX - ring3Radius,
            ring3CenterY - ring3Radius * 0.2f
        ),
        size = androidx.compose.ui.geometry.Size(
            width = ring3Radius * 2f, 
            height = ring3Radius * 0.4f
        ),
        style = androidx.compose.ui.graphics.drawscope.Stroke(width = ring3Thickness)
    )
}

// ==== SATURN RINGS IN FRONT OF SPHERE ====
private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawSaturnRingsFront(
    center: Offset,
    sphereRadius: Float,
    ringRotation1: Float,
    ringRotation2: Float,
    ringRotation3: Float,
    isDarkMode: Boolean,
    voiceIntensity: Float
) {
    val ringColors = if (isDarkMode) {
        listOf(
            Color(0x80FFD700), // GOLD - In front of sphere (brighter)
            Color(0x70FF8C00), // DARK ORANGE - In front of sphere
            Color(0x60FF4500)  // ORANGE RED - In front of sphere
        )
    } else {
        listOf(
            Color(0x90E91E63), // PINK - In front of sphere (brighter)
            Color(0x809C27B0), // PURPLE - In front of sphere
            Color(0x70673AB7)  // DEEP PURPLE - In front of sphere
        )
    }
    
    // Ring 1: In front of sphere (top half only)
    val ring1Radius = sphereRadius * 1.4f
    val ring1Thickness = (6f + voiceIntensity * 3f).safeStroke()
    val ring1Alpha = (0.6f + voiceIntensity * 0.3f).safeAlpha()
    
    // Only draw top half (in front of sphere)
    drawArc(
        color = ringColors[0].copy(alpha = ring1Alpha),
        startAngle = 180f,
        sweepAngle = 180f, // Top half only
        useCenter = false,
        topLeft = Offset(
            center.x - ring1Radius,
            center.y - ring1Radius * 0.3f
        ),
        size = androidx.compose.ui.geometry.Size(
            width = ring1Radius * 2f, 
            height = ring1Radius * 0.6f
        ),
        style = androidx.compose.ui.graphics.drawscope.Stroke(width = ring1Thickness)
    )
    
    // Ring 2: In front of sphere (top portion)
    val ring2Radius = sphereRadius * 1.6f
    val ring2Thickness = (5f + voiceIntensity * 2f).safeStroke()
    val ring2Alpha = (0.5f + voiceIntensity * 0.25f).safeAlpha()
    
    val ring2CenterX = center.x + (ring2Radius * 0.2f * cos(ringRotation2))
    val ring2CenterY = center.y + (ring2Radius * 0.1f * sin(ringRotation2))
    
    // Only draw top portion (in front of sphere)
    drawArc(
        color = ringColors[1].copy(alpha = ring2Alpha),
        startAngle = 180f,
        sweepAngle = 120f, // Top portion only
        useCenter = false,
        topLeft = Offset(
            ring2CenterX - ring2Radius,
            ring2CenterY - ring2Radius * 0.25f
        ),
        size = androidx.compose.ui.geometry.Size(
            width = ring2Radius * 2f, 
            height = ring2Radius * 0.5f
        ),
        style = androidx.compose.ui.graphics.drawscope.Stroke(width = ring2Thickness)
    )
    
    // Ring 3: In front of sphere (top portion)
    val ring3Radius = sphereRadius * 1.8f
    val ring3Thickness = (4f + voiceIntensity * 2f).safeStroke()
    val ring3Alpha = (0.4f + voiceIntensity * 0.2f).safeAlpha()
    
    val ring3CenterX = center.x + (ring3Radius * 0.15f * cos(ringRotation3 + PI.toFloat() / 4f))
    val ring3CenterY = center.y + (ring3Radius * 0.08f * sin(ringRotation3 + PI.toFloat() / 4f))
    
    // Only draw top portion (in front of sphere)
    drawArc(
        color = ringColors[2].copy(alpha = ring3Alpha),
        startAngle = 180f,
        sweepAngle = 100f, // Top portion only
        useCenter = false,
        topLeft = Offset(
            ring3CenterX - ring3Radius,
            ring3CenterY - ring3Radius * 0.2f
        ),
        size = androidx.compose.ui.geometry.Size(
            width = ring3Radius * 2f, 
            height = ring3Radius * 0.4f
        ),
        style = androidx.compose.ui.graphics.drawscope.Stroke(width = ring3Thickness)
    )
}

// --- Local easing (Compose không có sẵn 3 easing này) ---
private val EaseInOutSine = CubicBezierEasing(0.37f, 0.00f, 0.63f, 1.00f)
private val EaseInOutQuad = CubicBezierEasing(0.45f, 0.00f, 0.55f, 1.00f)
private val EaseInOutCubic = CubicBezierEasing(0.65f, 0.00f, 0.35f, 1.00f)

/**
 * Enhanced Orb AI Motion Sphere - Frame-by-Frame Timeline Animation
 * 
 * Modernized timeline animation cycle (3s):
 * 0.00-0.10s: Idle state - Subtle glow, gentle ripples, minimal movement
 * 0.10-0.27s: Listening activation - Core brightens, surface waves initiate, aura expands
 * 0.27-0.47s: Processing/Thinking - Dynamic surface deformations, energy particles orbit, swirls form
 * 0.47-0.67s: Responding - Energy waves emanate outward, halo intensity peaks, surface ripples
 * 0.67-1.00s: Return to idle - Gradual calming, energy dissipation, smooth transition to idle
 * 
 * Enhanced Layer System:
 * 1. Deep Core - Pulsating center with brightness variations and energy particles
 * 2. Inner Core - Vibrant gradients with responsive breathing animation
 * 3. Fluid Surface - Liquid plasma with dynamic wave patterns and deformations
 * 4. Energy Field - Orbiting particles and swirls that respond to voice/activity
 * 5. Luminous Aura - Ethereal glow with subtle pulsing and expanding waves
 * 6. Ambient Background - Soft environmental illumination that enhances depth
 */
@Composable
fun OrbAISphere(
    voiceLevel: Int,
    isListening: Boolean = false,
    isDarkMode: Boolean = true,
    onClick: () -> Unit = {},
    performancePreset: String = "lite", // "lite" | "balanced" | "ultra"
    modifier: Modifier = Modifier
) {
    // ==== PERFORMANCE PRESET SYSTEM ====
    val isLite = performancePreset == "lite"
    val isUltra = performancePreset == "ultra"
    
    // Giới hạn blur theo preset (QUAN TRỌNG: blur to quá dễ crash trên 1080x1920)
    val bgBlurDp = when {
        isLite -> 22.dp
        isUltra -> 48.dp
        else -> 32.dp
    }
    val haloBlurDp = when {
        isLite -> 16.dp
        isUltra -> 40.dp
        else -> 28.dp
    }
    
    // Ẩn bớt layer nặng theo preset
    val enableEnergyField = !isLite && isUltra     // chỉ ultra
    val enableComplexSurface = !isLite             // lite dùng surface đơn giản
    val enableRingsAndWaves = true                 // giữ lại nhưng giảm count ở dưới
    val enableHighlights = !isLite
    
    // Biên an toàn cho stroke/alpha
    fun safeAlpha(a: Float) = a.coerceIn(0.04f, 0.98f)
    fun safeStroke(w: Float) = w.coerceIn(0.8f, 12f)
    // ==== ENHANCED COLORS FOR LIGHT/DARK MODE ====
    
    val borderColors = if (isDarkMode) {
        listOf(
            Color(0xFFFFD700), // ULTRA BRIGHT GOLD - +80% stronger
            Color(0xFFFF8C00), // ULTRA ORANGE - +80% stronger
            Color(0xFFFF4500), // ULTRA DARK ORANGE - +80% stronger
            Color(0xFFFF0000)  // ULTRA RED - +80% stronger
        )
    } else {
        listOf(
            Color(0xFF0000CC), // ULTRA DARK BLUE - +90% stronger
            Color(0xFF0033CC), // ULTRA DARK DODGER BLUE - +90% stronger
            Color(0xFF0066CC), // ULTRA DARK SKY BLUE - +90% stronger
            Color(0xFF0099CC)  // ULTRA DARK CYAN - +90% stronger
        )
    }
    
    val waveColors = if (isDarkMode) {
        listOf(
            Color(0xFFFFD700), // ULTRA BRIGHT GOLD - +80% stronger waves
            Color(0xFFFF8C00), // ULTRA ORANGE - +80% stronger waves
            Color(0xFFFF4500), // ULTRA DARK ORANGE - +80% stronger waves
            Color(0xFFFF0000)  // ULTRA RED - +80% stronger waves
        )
    } else {
        listOf(
            Color(0xFF0000CC), // ULTRA DARK BLUE - +90% stronger waves
            Color(0xFF0033CC), // ULTRA DARK DODGER BLUE - +90% stronger waves
            Color(0xFF0066CC), // ULTRA DARK SKY BLUE - +90% stronger waves
            Color(0xFF0099CC)  // ULTRA DARK CYAN - +90% stronger waves
        )
    }
    
    val backgroundColors = if (isDarkMode) {
        listOf(
            Color(0x00FFFFFF), // Fully transparent
            Color(0x08FFFFFF), // More visible white
            Color(0x15FFFFFF), // More visible white
            Color(0x00FFFFFF)  // Fully transparent
        )
    } else {
        listOf(
            Color(0x30FFFFFF), // More opaque white
            Color(0x40FFFFFF), // Even more opaque white
            Color(0x50FFFFFF), // Very opaque white
            Color(0x30FFFFFF)  // More opaque white
        )
    }
    
    // ==== ENHANCED ANIMATION SYSTEM ====
    val transition = rememberInfiniteTransition(label = "orb_timeline")
    
    // Master timeline - 3 second cycle with improved easing
    val waveTimeline by transition.animateFloat(
        initialValue = 0f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(
                durationMillis = 3000,
                easing = LinearEasing
            )
        ),
        label = "waveTimeline"
    )
    
    // Voice intensity - CẢI THIỆN RESPONSIVENESS cho real voice
    val voiceIntensity by animateFloatAsState(
        targetValue = (voiceLevel / 3f).coerceIn(0f, 1f),
        animationSpec = spring(
            dampingRatio = Spring.DampingRatioLowBouncy, // Bouncy hơn
            stiffness = Spring.StiffnessHigh // Phản ứng nhanh hơn
        ),
        label = "voiceIntensity"
    )
    
    // ==== SATURN RINGS ANIMATION ====
    val ringRotation1 by transition.animateFloat(
        initialValue = 0f,
        targetValue = 2f * PI.toFloat(),
        animationSpec = infiniteRepeatable(
            animation = tween(8000, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "ringRotation1"
    )
    
    val ringRotation2 by transition.animateFloat(
        initialValue = 0f,
        targetValue = 2f * PI.toFloat(),
        animationSpec = infiniteRepeatable(
            animation = tween(12000, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "ringRotation2"
    )
    
    val ringRotation3 by transition.animateFloat(
        initialValue = 0f,
        targetValue = 2f * PI.toFloat(),
        animationSpec = infiniteRepeatable(
            animation = tween(10000, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "ringRotation3"
    )
    
    // Deep core pulse - Faster rhythm (1.5 second)
    val deepCorePulse by transition.animateFloat(
        initialValue = 0f,
        targetValue = 2f * PI.toFloat(),
        animationSpec = infiniteRepeatable(
            animation = tween(
                durationMillis = 1500,
                easing = EaseInOutSine
            )
        ),
        label = "deepCorePulse"
    )
    
    // Breathing cycle - Slower rhythm (2.5 second)
    val breathCycle by transition.animateFloat(
        initialValue = 0f,
        targetValue = 2f * PI.toFloat(),
        animationSpec = infiniteRepeatable(
            animation = tween(
                durationMillis = 2500,
                easing = EaseInOutQuad
            )
        ),
        label = "breathCycle"
    )
    
    // Energy swirl rotation - Variable speed
    val energySwirl by transition.animateFloat(
        initialValue = 0f,
        targetValue = 2f * PI.toFloat(),
        animationSpec = infiniteRepeatable(
            animation = tween(
                durationMillis = 8000,
                easing = LinearEasing
            )
        ),
        label = "energySwirl"
    )
    
    // Micro particles - Fast movement
    val microParticles by transition.animateFloat(
        initialValue = 0f,
        targetValue = 2f * PI.toFloat(),
        animationSpec = infiniteRepeatable(
            animation = tween(
                durationMillis = 4000,
                easing = LinearEasing
            )
        ),
        label = "microParticles"
    )
    
    // Surface waves - Medium rhythm
    val surfaceWaves by transition.animateFloat(
        initialValue = 0f,
        targetValue = 2f * PI.toFloat(),
        animationSpec = infiniteRepeatable(
            animation = tween(
                durationMillis = 5000,
                easing = EaseInOutSine
            )
        ),
        label = "surfaceWaves"
    )
    
    // Halo pulse - Gentle rhythm
    val haloPulse by transition.animateFloat(
        initialValue = 0f,
        targetValue = 2f * PI.toFloat(),
        animationSpec = infiniteRepeatable(
            animation = tween(
                durationMillis = 3500,
                easing = EaseInOutCubic
            )
        ),
        label = "haloPulse"
    )
    
    // 3D Sphere Rotation - X, Y, Z axes for true 3D floating sphere
    val liquidRotationX by transition.animateFloat(
        initialValue = 0f,
        targetValue = 2f * PI.toFloat(),
        animationSpec = infiniteRepeatable(
            animation = tween(
                durationMillis = 15000, // X-axis rotation: 15 seconds
                easing = LinearEasing
            )
        ),
        label = "liquidRotationX"
    )
    
    val liquidRotationY by transition.animateFloat(
        initialValue = 0f,
        targetValue = 2f * PI.toFloat(),
        animationSpec = infiniteRepeatable(
            animation = tween(
                durationMillis = 12000, // Y-axis rotation: 12 seconds
                easing = LinearEasing
            )
        ),
        label = "liquidRotationY"
    )
    
    val liquidRotationZ by transition.animateFloat(
        initialValue = 0f,
        targetValue = 2f * PI.toFloat(),
        animationSpec = infiniteRepeatable(
            animation = tween(
                durationMillis = 18000, // Z-axis rotation: 18 seconds
                easing = LinearEasing
            )
        ),
        label = "liquidRotationZ"
    )
    
    // Earth-like surface waves - SLOW & SMOOTH movement
    val liquidWaves by transition.animateFloat(
        initialValue = 0f,
        targetValue = 2f * PI.toFloat(),
        animationSpec = infiniteRepeatable(
            animation = tween(
                durationMillis = 8000, // SLOW: 8 seconds for full rotation like Earth's surface
                easing = EaseInOutSine
            )
        ),
        label = "liquidWaves"
    )
    
    // Free-floating sphere turbulence - SLOW & SMOOTH movement
    val liquidTurbulence by transition.animateFloat(
        initialValue = 0f,
        targetValue = 2f * PI.toFloat(),
        animationSpec = infiniteRepeatable(
            animation = tween(
                durationMillis = 10000, // SLOW: 10 seconds for full rotation like free-floating sphere
                easing = EaseInOutQuad
            )
        ),
        label = "liquidTurbulence"
    )
    
    // ==== ENHANCED PHASE SYSTEM ====
    val phase = when {
        waveTimeline < 0.1f -> OrbPhase.IDLE
        waveTimeline < 0.27f -> OrbPhase.LISTENING
        waveTimeline < 0.47f -> OrbPhase.THINKING
        waveTimeline < 0.67f -> OrbPhase.RESPONDING
        else -> OrbPhase.IDLE
    }
    
    // Dynamic phase progress (0-1 within each phase)
    val phaseProgress = when (phase) {
        OrbPhase.IDLE -> if (waveTimeline < 0.1f) waveTimeline / 0.1f else (1f - waveTimeline) / 0.33f
        OrbPhase.LISTENING -> (waveTimeline - 0.1f) / 0.17f
        OrbPhase.THINKING -> (waveTimeline - 0.27f) / 0.2f
        OrbPhase.RESPONDING -> (waveTimeline - 0.47f) / 0.2f
    }.coerceIn(0f, 1f)
    
    // ==== ENHANCED WAVE PARAMETERS - CẢI THIỆN VOICE RESPONSIVENESS ====
    val waveAmplitude = when (phase) {
        OrbPhase.IDLE -> 0.02f + voiceIntensity * 0.02f // Thêm voice response ngay cả khi idle
        OrbPhase.LISTENING -> 0.06f + voiceIntensity * 0.08f + phaseProgress * 0.03f // Tăng amplitude
        OrbPhase.THINKING -> 0.10f + voiceIntensity * 0.08f + sin(phaseProgress * PI.toFloat()) * 0.04f // Tăng amplitude
        OrbPhase.RESPONDING -> 0.12f + voiceIntensity * 0.10f + (1f - phaseProgress) * 0.05f // Tăng amplitude
    }
    
    val waveFrequency = 1.2f + voiceIntensity * 1.0f // Tăng frequency cho responsive hơn
    val surfaceRoughness = when (phase) {
        OrbPhase.IDLE -> 0.15f + voiceIntensity * 0.05f // Thêm voice response
        OrbPhase.LISTENING -> 0.20f + phaseProgress * 0.08f + voiceIntensity * 0.10f // Tăng roughness
        OrbPhase.THINKING -> 0.30f + voiceIntensity * 0.20f // Tăng roughness
        OrbPhase.RESPONDING -> 0.30f + (1f - phaseProgress) * 0.18f + voiceIntensity * 0.15f // Tăng roughness
    }
    
    // ==== ENHANCED BRIGHTNESS SYSTEM - CẢI THIỆN VOICE RESPONSIVENESS ====
    val baseBrightness = 0.70f + voiceIntensity * 0.35f // Tăng base brightness và voice response
    val phaseBrightness = when (phase) {
        OrbPhase.IDLE -> 0.7f + voiceIntensity * 0.2f // Thêm voice response ngay cả khi idle
        OrbPhase.LISTENING -> 0.85f + phaseProgress * 0.25f + voiceIntensity * 0.15f // Tăng brightness
        OrbPhase.THINKING -> 1.1f + sin(phaseProgress * PI.toFloat() * 2) * 0.15f + voiceIntensity * 0.2f // Tăng brightness
        OrbPhase.RESPONDING -> 1.1f + (1f - phaseProgress) * 0.15f + voiceIntensity * 0.25f // Tăng brightness
    }
    val currentBrightness = baseBrightness * phaseBrightness
    
    // ==== ENHANCED HALO SYSTEM - CẢI THIỆN VOICE RESPONSIVENESS ====
    val baseHaloRadius = 1.4f + voiceIntensity * 0.3f // Tăng base radius và voice response
    val phaseHaloRadius = when (phase) {
        OrbPhase.IDLE -> 1.0f + voiceIntensity * 0.1f // Thêm voice response ngay cả khi idle
        OrbPhase.LISTENING -> 1.1f + phaseProgress * 0.25f + voiceIntensity * 0.15f // Tăng radius
        OrbPhase.THINKING -> 1.3f + sin(phaseProgress * PI.toFloat()) * 0.15f + voiceIntensity * 0.2f // Tăng radius
        OrbPhase.RESPONDING -> 1.3f + (1f - phaseProgress) * 0.35f + voiceIntensity * 0.25f // Tăng radius
    }
    val currentHaloRadius = baseHaloRadius * phaseHaloRadius
    
    // ==== ENHANCED RENDERING SYSTEM ====
    Box(
        modifier = modifier
            .fillMaxSize()
            .clickable { onClick() }
    ) {
        // Layer 1 & 2: DISABLED - Bỏ hiệu ứng phát sáng và xoay
        // Disabled để sphere nổi bật hơn
        
        // Layer 3: Energy Field - DISABLED to prevent crash
        // if (phase == OrbPhase.THINKING) {
        //     // Disabled for stability
        // }
        
        // Layer 4: Surface - Performance-aware
        if (enableComplexSurface) {
            Canvas(
                modifier = Modifier
                    .fillMaxSize()
                    .alpha(0.7f) // Reduced alpha
                    .scale(1f + waveAmplitude * 0.3f) // Reduced scale
            ) {
                // Simple fluid surface without complex calculations
                val surfaceRadius = size.minDimension * 0.35f
                val surfaceAlpha = currentBrightness * 0.3f
                
                drawCircle(
                    brush = Brush.radialGradient(
                        colors = listOf(
                            backgroundColors[0].copy(alpha = surfaceAlpha),
                            backgroundColors[1].copy(alpha = surfaceAlpha * 0.5f),
                            backgroundColors[2].copy(alpha = surfaceAlpha * 0.2f),
                            Color.Transparent
                        ),
                        center = center,
                        radius = surfaceRadius
                    ),
                    radius = surfaceRadius,
                    center = center
                )
            }
        } else {
            // surface nhẹ: chỉ một radial fill
            Canvas(
                modifier = Modifier
                    .fillMaxSize()
                    .alpha(0.55f)
            ) {
                val r = size.minDimension * 0.33f
                drawCircle(
                    brush = Brush.radialGradient(
                        colors = listOf(
                            backgroundColors[1].copy(alpha = 0.35f),
                            backgroundColors[2].copy(alpha = 0.18f),
                            Color.Transparent
                        ),
                        center = center,
                        radius = r
                    ),
                    radius = r,
                    center = center
                )
            }
        }
        
        // Layer 5: Liquid 3D Sphere with Rotation - CẢI THIỆN VOICE RESPONSIVENESS
        Canvas(
            modifier = Modifier
                .fillMaxSize()
                .alpha(0.95f)
                .scale(1f + sin(breathCycle) * (0.03f + voiceIntensity * 0.02f)) // Thêm voice response vào breathing
        ) {
            val sphereRadius = (size.minDimension * 0.35f * 0.9f).safeDrawRadius()
            
        // Draw Liquid 3D Sphere
        drawLiquid3DSphere(
            center = center,
            radius = sphereRadius,
            borderColors = borderColors,
            waveColors = waveColors,
            intensity = currentBrightness,
            liquidRotationX = liquidRotationX,
            liquidRotationY = liquidRotationY,
            liquidRotationZ = liquidRotationZ,
            liquidWaves = liquidWaves,
            liquidTurbulence = liquidTurbulence,
            breathPhase = breathCycle,
            phase = phase,
            phaseProgress = phaseProgress,
            voiceIntensity = voiceIntensity,
            isListening = isListening,
            isDarkMode = isDarkMode
        )
        }
        
        // Layer 6 & 7: Saturn Rings - DISABLED for better user experience
        // Removed rings for cleaner, more accessible design
        
        // Layer 6: Deep Core - DISABLED to prevent crash
        // Disabled for stability
        
        // Layer 7: Energy Rays - DISABLED to prevent crash
        // Disabled for stability
        
        // Layer 8: Expanding Waves - DISABLED to prevent crash
        // Disabled for stability
        
        // Layer 9: Surface Highlights - DISABLED to prevent crash
        // Disabled for stability
    }
}

// ==== ENHANCED PHASE SYSTEM ====

enum class OrbPhase {
    IDLE,       // Gentle ambient state
    LISTENING,  // Activation and input capture
    THINKING,   // Processing and analysis
    RESPONDING  // Output and communication
}
// ==== ENHANCED DRAWING FUNCTIONS ====

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawAmbientBackground(
    center: Offset,
    radius: Float,
    colors: List<Color>,
    intensity: Float,
    phase: OrbPhase,
    phaseProgress: Float
) {
    // Ambient glow based on phase
    val glowRadius = radius * (1.5f + when(phase) {
        OrbPhase.IDLE -> 0.2f
        OrbPhase.LISTENING -> 0.3f + phaseProgress * 0.2f
        OrbPhase.THINKING -> 0.5f + sin(phaseProgress * PI.toFloat()) * 0.2f
        OrbPhase.RESPONDING -> 0.5f + (1f - phaseProgress) * 0.3f
    })
    
    // Main ambient glow
    drawCircle(
        brush = Brush.radialGradient(
            colors = listOf(
                colors[1].copy(alpha = intensity * 0.15f),
                colors[2].copy(alpha = intensity * 0.10f),
                colors[3].copy(alpha = intensity * 0.05f),
                Color.Transparent
            ),
            center = center,
            radius = glowRadius
        ),
        radius = glowRadius,
        center = center
    )
    
    // Secondary ambient pulses
    if (phase != OrbPhase.IDLE) {
        val pulseCount = 2
        for (i in 0 until pulseCount) {
            val pulseProgress = (phaseProgress + i * 0.5f) % 1f
            val pulseRadius = glowRadius * (0.6f + pulseProgress * 0.6f)
            val pulseAlpha = intensity * 0.08f * (1f - pulseProgress)
            
            drawCircle(
                brush = Brush.radialGradient(
                    colors = listOf(
                        colors[2].copy(alpha = pulseAlpha),
                        colors[3].copy(alpha = pulseAlpha * 0.5f),
                        Color.Transparent
                    ),
                    center = center,
                    radius = pulseRadius
                ),
                radius = pulseRadius,
                center = center
            )
        }
    }
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawExpandedAura(
    center: Offset,
    radius: Float,
    colors: List<Color>,
    coreColors: List<Color>,
    intensity: Float,
    haloRadius: Float,
    pulsePhase: Float,
    phase: OrbPhase,
    phaseProgress: Float
) {
    val currentRadius = (radius * haloRadius).safeDrawRadius()
    val pulseIntensity = (0.7f + 0.3f * (sin(pulsePhase) + 1f) / 2f).safeAlpha()
    val phaseIntensity = when (phase) {
        OrbPhase.IDLE -> 0.7f
        OrbPhase.LISTENING -> 0.8f + phaseProgress * 0.2f
        OrbPhase.THINKING -> 0.9f + sin(phaseProgress * PI.toFloat() * 2) * 0.1f
        OrbPhase.RESPONDING -> 1.0f
    }.safeAlpha()
    
    // Enhanced main halo with multiple layers
    drawCircle(
        brush = Brush.radialGradient(
            colors = listOf(
                coreColors[3].copy(alpha = (intensity * 0.3f * pulseIntensity).safeAlpha()),
                coreColors[2].copy(alpha = (intensity * 0.25f * pulseIntensity).safeAlpha()),
                coreColors[1].copy(alpha = (intensity * 0.2f * pulseIntensity).safeAlpha()),
                coreColors[0].copy(alpha = (intensity * 0.15f * pulseIntensity).safeAlpha()),
                Color.Transparent // Fixed: thay vì colors[4]
            ),
            center = center,
            radius = currentRadius.safeShaderRadius()
        ),
        radius = currentRadius,
        center = center
    )
    
    // Inner aura glow
    drawCircle(
        brush = Brush.radialGradient(
            colors = listOf(
                Color.White.copy(alpha = intensity * 0.15f * pulseIntensity),
                coreColors[3].copy(alpha = intensity * 0.1f * pulseIntensity),
                Color.Transparent
            ),
            center = center,
            radius = currentRadius * 0.7f
        ),
        radius = currentRadius * 0.7f,
        center = center
    )
    
    // Dynamic aura waves for active phases
    if (phase != OrbPhase.IDLE) {
        val waveCount = 3
        for (i in 0 until waveCount) {
            val waveProgress = (phaseProgress + i * (1f / waveCount)) % 1f
            val waveRadius = currentRadius * (0.7f + waveProgress * 0.5f)
            val waveAlpha = intensity * 0.15f * phaseIntensity * (1f - waveProgress)
            
            drawCircle(
                brush = Brush.radialGradient(
                    colors = listOf(
                        coreColors[1].copy(alpha = waveAlpha),
                        coreColors[0].copy(alpha = waveAlpha * 0.7f),
                        Color.Transparent
                    ),
                    center = center,
                    radius = waveRadius
                ),
                radius = waveRadius,
                center = center
            )
        }
    }
    
    // Outer halo flares for thinking/responding phases
    if (phase == OrbPhase.THINKING || phase == OrbPhase.RESPONDING) {
        val flareCount = 6
        for (i in 0 until flareCount) {
            val flareAngle = i * (2f * PI.toFloat() / flareCount) + phaseProgress * PI.toFloat()
            val flareDistance = currentRadius * 0.9f
            val flareSize = currentRadius * 0.4f
            
            val flareX = center.x + cos(flareAngle) * flareDistance
            val flareY = center.y + sin(flareAngle) * flareDistance
            
            val flareAlpha = intensity * 0.1f * phaseIntensity
            
            drawCircle(
                brush = Brush.radialGradient(
                    colors = listOf(
                        coreColors[2].copy(alpha = flareAlpha),
                        coreColors[1].copy(alpha = flareAlpha * 0.7f),
                        Color.Transparent
                    ),
                    center = Offset(flareX, flareY),
                    radius = flareSize
                ),
                radius = flareSize,
                center = Offset(flareX, flareY)
            )
        }
    }
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawEnergyField(
    center: Offset,
    radius: Float,
    colors: List<Color>,
    intensity: Float,
    swirlPhase: Float,
    microPhase: Float,
    voiceIntensity: Float,
    phase: OrbPhase,
    phaseProgress: Float
) {
    // Simplified parameters to reduce complexity
    val fieldRadius = radius * 1.0f
    val particleDensity = when (phase) {
        OrbPhase.THINKING -> 4 + (voiceIntensity * 2).toInt() // Reduced from 8+4
        OrbPhase.RESPONDING -> 3 + (voiceIntensity * 1).toInt() // Reduced from 6+3
        else -> 2 // Reduced from 4
    }
    
    // Simplified energy field swirls (reduced from 3 to 2)
    for (i in 0 until 2) {
        val orbitRadius = fieldRadius * (0.6f + i * 0.2f)
        val orbitSpeed = 1f - i * 0.2f
        val orbitPhase = swirlPhase * orbitSpeed
        
        for (j in 0 until particleDensity) {
            val angle = j * (2f * PI.toFloat() / particleDensity) + orbitPhase
            
            // Add some variation to particle positions
            val radiusVariation = sin(angle * 3f + microPhase) * 0.1f
            val currentRadius = orbitRadius * (1f + radiusVariation)
            
            val x = center.x + cos(angle) * currentRadius
            val y = center.y + sin(angle) * currentRadius
            
            // Particle size and opacity variation
            val particleSize = (4f + voiceIntensity * 3f + sin(angle * 5f + microPhase) * 2f) * 
                when (phase) {
                    OrbPhase.THINKING -> 1f
                    OrbPhase.RESPONDING -> (1f - phaseProgress) * 0.8f + 0.2f
                    else -> 0.5f
                }
            
            val particleAlpha = intensity * (0.3f + voiceIntensity * 0.4f) * 
                (1f - radiusVariation * 2f).coerceIn(0.5f, 1f)
            
            // Draw the particle
            drawCircle(
                brush = Brush.radialGradient(
                    colors = listOf(
                        colors[0].copy(alpha = particleAlpha),
                        colors[1].copy(alpha = particleAlpha * 0.7f),
                        Color.Transparent
                    ),
                    center = Offset(x, y),
                    radius = particleSize
                ),
                radius = particleSize,
                center = Offset(x, y)
            )
            
            // Draw connecting lines between adjacent particles for energy streams
            if (j < particleDensity - 1 && phase == OrbPhase.THINKING) {
                val nextAngle = (j + 1) * (2f * PI.toFloat() / particleDensity) + orbitPhase
                val nextRadiusVariation = sin(nextAngle * 3f + microPhase) * 0.1f
                val nextRadius = orbitRadius * (1f + nextRadiusVariation)
                
                val nextX = center.x + cos(nextAngle) * nextRadius
                val nextY = center.y + sin(nextAngle) * nextRadius
                
                val lineAlpha = particleAlpha * 0.4f
                val lineWidth = 1f + voiceIntensity * 1f
                
                drawLine(
                    brush = Brush.linearGradient(
                        colors = listOf(
                            colors[1].copy(alpha = lineAlpha),
                            colors[2].copy(alpha = lineAlpha * 0.7f)
                        ),
                        start = Offset(x, y),
                        end = Offset(nextX, nextY)
                    ),
                    start = Offset(x, y),
                    end = Offset(nextX, nextY),
                    strokeWidth = lineWidth
                )
            }
        }
    }
    
    // Micro particles - creates a dust-like effect around the field
    if (phase == OrbPhase.THINKING) {
        val microCount = (20 + (voiceIntensity * 15).toInt()).coerceAtMost(40)
        for (i in 0 until microCount) {
            val angle = i * (2f * PI.toFloat() / microCount) + microPhase * 1.5f
            val distance = fieldRadius * (0.5f + Random.nextFloat() * 0.6f)
            
            val x = center.x + cos(angle) * distance
            val y = center.y + sin(angle) * distance
            
            val microSize = 1f + Random.nextFloat() * 2f
            val microAlpha = intensity * (0.2f + Random.nextFloat() * 0.3f) * phaseProgress
            
            drawCircle(
                color = colors[1].copy(alpha = microAlpha),
                radius = microSize,
                center = Offset(x, y)
            )
        }
    }
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawFluidSurface(
    center: Offset,
    radius: Float,
    colors: List<Color>,
    waveColors: List<Color>,
    intensity: Float,
    wavePhase: Float,
    waveAmplitude: Float,
    waveFrequency: Float,
    surfaceRoughness: Float,
    phase: OrbPhase,
    phaseProgress: Float,
    voiceIntensity: Float
) {
    // Base surface with enhanced fluid effect
    val baseRadius = radius * 0.95f
    
    // Main fluid surface - outer layer
    drawCircle(
        brush = Brush.radialGradient(
            colors = listOf(
                colors[3].copy(alpha = intensity * 0.8f),
                colors[2].copy(alpha = intensity * 0.7f),
                colors[1].copy(alpha = intensity * 0.6f),
                colors[0].copy(alpha = intensity * 0.4f),
                Color.Transparent
            ),
            center = center,
            radius = baseRadius
        ),
        radius = baseRadius,
        center = center
    )
    
    // Primary wave patterns - creates fluid-like deformations
    val waveCount = 9
    for (i in 0 until waveCount) {
        val waveAngle = i * (2f * PI.toFloat() / waveCount)
        val wavePhaseOffset = wavePhase + waveAngle
        
        val waveOffset = sin(wavePhaseOffset) * waveAmplitude * baseRadius
        val waveRadius = baseRadius + waveOffset
        
        // Wave opacity based on phase and intensity
        val waveAlpha = intensity * (0.3f + voiceIntensity * 0.3f) * when (phase) {
            OrbPhase.IDLE -> 0.6f
            OrbPhase.LISTENING -> 0.7f + phaseProgress * 0.3f
            OrbPhase.THINKING -> 0.8f + sin(phaseProgress * PI.toFloat() * 2) * 0.2f
            OrbPhase.RESPONDING -> 0.9f + (1f - phaseProgress) * 0.1f
        }
        
        // Wave ring with gradient
        drawCircle(
            brush = Brush.radialGradient(
                colors = listOf(
                    colors[3].copy(alpha = waveAlpha),
                    colors[2].copy(alpha = waveAlpha * 0.8f),
                    colors[1].copy(alpha = waveAlpha * 0.6f),
                    Color.Transparent
                ),
                center = center,
                radius = waveRadius
            ),
            radius = waveRadius,
            center = center
        )
    }
    
    // Secondary turbulence - adds complexity to the surface
    if (phase != OrbPhase.IDLE) {
        val turbulenceCount = 12
        for (i in 0 until turbulenceCount) {
            val turbAngle = i * (2f * PI.toFloat() / turbulenceCount)
            val turbPhase = wavePhase * 1.5f + turbAngle
            
            // Turbulence moves at a different frequency
            val turbOffset = sin(turbPhase) * waveAmplitude * baseRadius * 0.7f
            
            // Position on surface where deformation occurs
            val deformAngle = turbAngle + sin(wavePhase) * 0.2f
            val deformRadius = baseRadius * (0.7f + sin(turbPhase * 2) * 0.15f)
            val deformX = center.x + cos(deformAngle) * deformRadius
            val deformY = center.y + sin(deformAngle) * deformRadius
            
            // Deformation size and alpha
            val deformSize = baseRadius * surfaceRoughness * (0.1f + sin(turbPhase) * 0.05f)
            val deformAlpha = intensity * (0.2f + voiceIntensity * 0.3f) * phaseProgress
            
            // Draw the surface deformation
            drawCircle(
                brush = Brush.radialGradient(
                    colors = listOf(
                        colors[2].copy(alpha = deformAlpha),
                        colors[1].copy(alpha = deformAlpha * 0.7f),
                        Color.Transparent
                    ),
                    center = Offset(deformX, deformY),
                    radius = deformSize
                ),
                radius = deformSize,
                center = Offset(deformX, deformY)
            )
        }
    }
    
    // Energy currents - visible during active phases
    if (phase == OrbPhase.THINKING || phase == OrbPhase.RESPONDING) {
        val currentCount = 4
        for (i in 0 until currentCount) {
            val currentAngle = i * (2f * PI.toFloat() / currentCount) + wavePhase
            
            // Create a flowing path across the surface
            val pathPoints = 5
            val pathWidth = 2f + voiceIntensity * 2f
            val pathAlpha = intensity * (0.3f + voiceIntensity * 0.3f) * phaseProgress
            
            for (j in 0 until pathPoints - 1) {
                val startAngle = currentAngle + j * 0.2f
                val endAngle = currentAngle + (j + 1) * 0.2f
                
                val startRadius = baseRadius * (0.5f + j * 0.1f)
                val endRadius = baseRadius * (0.5f + (j + 1) * 0.1f)
                
                val startX = center.x + cos(startAngle) * startRadius
                val startY = center.y + sin(startAngle) * startRadius
                val endX = center.x + cos(endAngle) * endRadius
                val endY = center.y + sin(endAngle) * endRadius
                
                // Draw energy current segment
                drawLine(
                    brush = Brush.linearGradient(
                        colors = listOf(
                            waveColors[1].copy(alpha = pathAlpha),
                            waveColors[2].copy(alpha = pathAlpha * 0.7f)
                        ),
                        start = Offset(startX, startY),
                        end = Offset(endX, endY)
                    ),
                    start = Offset(startX, startY),
                    end = Offset(endX, endY),
                    strokeWidth = pathWidth,
                    cap = StrokeCap.Round
                )
            }
        }
    }
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawInnerCore(
    center: Offset,
    radius: Float,
    colors: List<Color>,
    waveColors: List<Color>,
    intensity: Float,
    breathPhase: Float,
    phase: OrbPhase,
    phaseProgress: Float,
    voiceIntensity: Float
) {
    // Core pulsation effect
    val coreRadius = radius * (0.65f + voiceIntensity * 0.1f)
    val breathEffect = 0.8f + 0.2f * (sin(breathPhase) + 1f) / 2f
    val currentIntensity = intensity * breathEffect
    
    // Enhanced core gradient with multiple layers
    drawCircle(
        brush = Brush.radialGradient(
            colors = listOf(
                waveColors[0].copy(alpha = currentIntensity * 0.9f),
                colors[3].copy(alpha = currentIntensity * 0.85f),
                colors[2].copy(alpha = currentIntensity * 0.8f),
                colors[1].copy(alpha = currentIntensity * 0.7f),
                colors[0].copy(alpha = currentIntensity * 0.5f),
                Color.Transparent
            ),
            center = center,
            radius = coreRadius
        ),
        radius = coreRadius,
        center = center
    )
    
    // Inner glow - creates depth and luminosity
    val innerGlowRadius = coreRadius * 0.8f
    drawCircle(
        brush = Brush.radialGradient(
            colors = listOf(
                waveColors[0].copy(alpha = currentIntensity * 0.95f),
                waveColors[1].copy(alpha = currentIntensity * 0.9f),
                colors[3].copy(alpha = currentIntensity * 0.8f),
                Color.Transparent
            ),
            center = center,
            radius = innerGlowRadius
        ),
        radius = innerGlowRadius,
        center = center
    )
    
    // Energy currents within core for active phases
    if (phase != OrbPhase.IDLE) {
        val swirls = 5
        for (i in 0 until swirls) {
            val swirlAngle = i * (2f * PI.toFloat() / swirls) + breathPhase
            val swirlRadius = coreRadius * (0.3f + i * 0.1f)
            
            // Dynamic positioning based on phase
            val offsetX = center.x + cos(swirlAngle) * swirlRadius * 0.2f
            val offsetY = center.y + sin(swirlAngle) * swirlRadius * 0.2f
            
            // Swirl opacity based on phase
            val swirlAlpha = currentIntensity * (0.2f + phaseProgress * 0.3f)
            
            // Energy swirl gradient
            drawCircle(
                brush = Brush.radialGradient(
                    colors = listOf(
                        waveColors[0].copy(alpha = swirlAlpha),
                        waveColors[1].copy(alpha = swirlAlpha * 0.8f),
                        Color.Transparent
                    ),
                    center = Offset(offsetX, offsetY),
                    radius = swirlRadius * 0.5f
                ),
                radius = swirlRadius * 0.5f,
                center = Offset(offsetX, offsetY)
            )
        }
    }
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawDeepCore(
    center: Offset,
    radius: Float,
    colors: List<Color>,
    intensity: Float,
    pulsePhase: Float,
    phase: OrbPhase,
    phaseProgress: Float,
    voiceIntensity: Float
) {
    // Core energy center with intense brightness
    val deepRadius = radius * (0.7f + sin(pulsePhase) * 0.1f)
    val pulseIntensity = 0.9f + 0.1f * (sin(pulsePhase) + 1f) / 2f
    
    // Phase-specific intensity modulation
    val phaseIntensity = when (phase) {
        OrbPhase.IDLE -> 0.7f
        OrbPhase.LISTENING -> 0.8f + phaseProgress * 0.2f
        OrbPhase.THINKING -> 0.9f + sin(phaseProgress * PI.toFloat() * 3) * 0.1f
        OrbPhase.RESPONDING -> 1.0f
    }
    
    val currentIntensity = intensity * pulseIntensity * phaseIntensity
    
    // Deep core with brilliant center
    drawCircle(
        brush = Brush.radialGradient(
            colors = listOf(
                Color.White.copy(alpha = currentIntensity),
                colors[0].copy(alpha = currentIntensity * 0.95f),
                colors[1].copy(alpha = currentIntensity * 0.85f),
                Color.Transparent
            ),
            center = center,
            radius = deepRadius
        ),
        radius = deepRadius,
        center = center
    )
    
    // Central bright point - pure energy source
    drawCircle(
        color = Color.White.copy(alpha = currentIntensity),
        radius = deepRadius * 0.4f,
        center = center
    )
    
    // Energy particles emanating from center during active phases
    if (phase != OrbPhase.IDLE) {
        val particleCount = (6 + (voiceIntensity * 6).toInt()).coerceAtMost(12)
        
        for (i in 0 until particleCount) {
            val particleAngle = i * (2f * PI.toFloat() / particleCount) + pulsePhase * 2
            
            // Particle distance varies with phase
            val particleDistance = deepRadius * when (phase) {
                OrbPhase.LISTENING -> 0.6f + phaseProgress * 0.4f
                OrbPhase.THINKING -> 0.8f + sin(phaseProgress * PI.toFloat() * 2) * 0.2f
                OrbPhase.RESPONDING -> 0.8f + (1f - phaseProgress) * 0.4f
                else -> 0.7f
            }
            
            val particleX = center.x + cos(particleAngle) * particleDistance
            val particleY = center.y + sin(particleAngle) * particleDistance
            
            // Particle size and brightness
            val particleSize = 3f + voiceIntensity * 2f
            val particleAlpha = currentIntensity * (0.7f + voiceIntensity * 0.3f) * phaseIntensity
            
            // Draw the energy particle
            drawCircle(
                brush = Brush.radialGradient(
                    colors = listOf(
                        Color.White.copy(alpha = particleAlpha),
                        colors[0].copy(alpha = particleAlpha * 0.8f),
                        Color.Transparent
                    ),
                    center = Offset(particleX, particleY),
                    radius = particleSize
                ),
                radius = particleSize,
                center = Offset(particleX, particleY)
            )
        }
    }
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawEnergyRays(
    center: Offset,
    radius: Float,
    colors: List<Color>,
    intensity: Float,
    rayPhase: Float,
    phase: OrbPhase,
    phaseProgress: Float,
    voiceIntensity: Float
) {
    // Ray count and properties based on phase
    val rayCount = when (phase) {
        OrbPhase.IDLE -> 0
        OrbPhase.LISTENING -> 6
        OrbPhase.THINKING -> 8
        OrbPhase.RESPONDING -> 10
    }
    
    if (rayCount > 0) {
        val rayIntensity = intensity * when (phase) {
            OrbPhase.LISTENING -> 0.4f + phaseProgress * 0.3f
            OrbPhase.THINKING -> 0.6f + sin(phaseProgress * PI.toFloat() * 2) * 0.2f
            OrbPhase.RESPONDING -> 0.7f + (1f - phaseProgress) * 0.3f
            else -> 0.4f
        }
        
        // Draw energy rays emanating from center
        for (i in 0 until rayCount) {
            val baseAngle = i * (2f * PI.toFloat() / rayCount)
            val angle = baseAngle + sin(rayPhase * 0.5f) * 0.1f
            
            // Ray length varies with phase and voice intensity
            val rayLength = radius * (0.8f + voiceIntensity * 0.3f) * when (phase) {
                OrbPhase.LISTENING -> 0.7f + phaseProgress * 0.3f
                OrbPhase.THINKING -> 0.8f + sin(phaseProgress * PI.toFloat() * 2) * 0.2f
                OrbPhase.RESPONDING -> 0.9f + (1f - phaseProgress) * 0.2f
                else -> 0.7f
            }
            
            // Ray width modulation
            val rayWidth = 2f + voiceIntensity * 3f + sin(angle * 3f + rayPhase) * 1f
            
            // Calculate ray endpoints with slight curve
            val curveOffset = sin(rayPhase + baseAngle) * radius * 0.1f
            val curveAngle = angle + curveOffset * 0.02f
            
            val innerRadius = radius * 0.2f
            val startX = center.x + cos(angle) * innerRadius
            val startY = center.y + sin(angle) * innerRadius
            val endX = center.x + cos(curveAngle) * rayLength
            val endY = center.y + sin(curveAngle) * rayLength
            
            // Ray color gradient
            drawLine(
                brush = Brush.linearGradient(
                    colors = listOf(
                        colors[0].copy(alpha = rayIntensity),
                        colors[1].copy(alpha = rayIntensity * 0.8f),
                        colors[2].copy(alpha = rayIntensity * 0.5f),
                        colors[3].copy(alpha = rayIntensity * 0.2f)
                    ),
                    start = Offset(startX, startY),
                    end = Offset(endX, endY)
                ),
                start = Offset(startX, startY),
                end = Offset(endX, endY),
                strokeWidth = rayWidth,
                cap = StrokeCap.Round
            )
            
            // Add glow point at end of ray
            if (phase == OrbPhase.THINKING || phase == OrbPhase.RESPONDING) {
                drawCircle(
                    brush = Brush.radialGradient(
                        colors = listOf(
                            colors[1].copy(alpha = rayIntensity * 0.7f),
                            colors[2].copy(alpha = rayIntensity * 0.4f),
                            Color.Transparent
                        ),
                        center = Offset(endX, endY),
                        radius = rayWidth * 2f
                    ),
                    radius = rayWidth * 2f,
                    center = Offset(endX, endY)
                )
            }
        }
    }
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawExpandingWaves(
    center: Offset,
    radius: Float,
    colors: List<Color>,
    intensity: Float,
    phaseProgress: Float,
    voiceIntensity: Float
) {
    // Multiple expanding wave rings
    val waveCount = 3
    val baseWaveSpacing = 0.3f
    
    for (i in 0 until waveCount) {
        // Calculate wave progression - each wave starts at a different time
        val waveOffset = (i * baseWaveSpacing) % 1f
        val waveProgress = (1f - phaseProgress + waveOffset) % 1f
        
        // Only draw waves that are within the visible expansion phase
        if (waveProgress < 0.8f) {
            // Radius grows as wave expands outward
            val waveRadius = radius * (1f + waveProgress * 2f)
            
            // Opacity fades as wave expands
            val waveAlpha = intensity * (1f - waveProgress) * (0.5f + voiceIntensity * 0.3f)
            
            // Wave thickness varies with voice intensity
            val waveThickness = (3f + voiceIntensity * 2f) * (1f - waveProgress * 0.5f)
            
            // Draw the expanding wave ring
            drawCircle(
                style = androidx.compose.ui.graphics.drawscope.Stroke(width = waveThickness),
                brush = Brush.radialGradient(
                    colors = listOf(
                        colors[3].copy(alpha = waveAlpha),
                        colors[2].copy(alpha = waveAlpha * 0.8f),
                        colors[1].copy(alpha = waveAlpha * 0.5f),
                        colors[0].copy(alpha = waveAlpha * 0.3f),
                        Color.Transparent
                    ),
                    center = center,
                    radius = waveRadius + waveThickness / 2
                ),
                radius = waveRadius,
                center = center
            )
        }
    }
    
    // Pulse ripples - smaller, faster waves
    val rippleCount = 5
    for (i in 0 until rippleCount) {
        val rippleOffset = (i * 0.2f) % 1f
        val rippleProgress = (1f - phaseProgress * 1.5f + rippleOffset) % 1f
        
        if (rippleProgress < 0.6f) {
            val rippleRadius = radius * (1f + rippleProgress * 1.2f)
            val rippleAlpha = intensity * (1f - rippleProgress) * 0.3f * (0.5f + voiceIntensity * 0.5f)
            val rippleThickness = 1.5f * (1f - rippleProgress * 0.7f)
            
            drawCircle(
                style = androidx.compose.ui.graphics.drawscope.Stroke(width = rippleThickness),
                color = colors[2].copy(alpha = rippleAlpha),
                radius = rippleRadius,
                center = center
            )
        }
    }
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawSurfaceHighlights(
    center: Offset,
    radius: Float,
    intensity: Float,
    phase: OrbPhase,
    phaseProgress: Float,
    breathPhase: Float,
    voiceIntensity: Float
) {
    // Primary highlight - creates a glass-like reflection effect
    val highlightRadius = radius * 0.8f
    val highlightOffsetScale = 0.15f * radius
    
    // Highlight position shifts with breathing
    val highlightOffsetX = cos(breathPhase) * highlightOffsetScale
    val highlightOffsetY = sin(breathPhase) * highlightOffsetScale
    
    // Highlight intensity varies with phase
    val highlightAlpha = intensity * when (phase) {
        OrbPhase.IDLE -> 0.3f
        OrbPhase.LISTENING -> 0.3f + phaseProgress * 0.2f
        OrbPhase.THINKING -> 0.4f + sin(phaseProgress * PI.toFloat() * 2) * 0.1f
        OrbPhase.RESPONDING -> 0.4f + (1f - phaseProgress) * 0.1f
    }
    
    // Main highlight
    drawCircle(
        brush = Brush.radialGradient(
            colors = listOf(
                Color.White.copy(alpha = highlightAlpha * 0.7f),
                Color.White.copy(alpha = highlightAlpha * 0.3f),
                Color.Transparent
            ),
            center = Offset(center.x + highlightOffsetX, center.y + highlightOffsetY),
            radius = highlightRadius * 0.6f
        ),
        radius = highlightRadius * 0.6f,
        center = Offset(center.x + highlightOffsetX, center.y + highlightOffsetY)
    )
    
    // Secondary highlights - adds depth and realism
    if (phase != OrbPhase.IDLE) {
        val secondaryHighlightCount = 3
        for (i in 0 until secondaryHighlightCount) {
            val angle = i * (2f * PI.toFloat() / secondaryHighlightCount) + breathPhase
            
            val highlightX = center.x + cos(angle) * highlightRadius * 0.6f
            val highlightY = center.y + sin(angle) * highlightRadius * 0.6f
            
            val secondarySize = radius * (0.1f + voiceIntensity * 0.05f) * (1f - i * 0.2f)
            val secondaryAlpha = highlightAlpha * 0.5f * (1f - i * 0.2f) * phaseProgress
            
            drawCircle(
                brush = Brush.radialGradient(
                    colors = listOf(
                        Color.White.copy(alpha = secondaryAlpha),
                        Color.White.copy(alpha = secondaryAlpha * 0.5f),
                        Color.Transparent
                    ),
                    center = Offset(highlightX, highlightY),
                    radius = secondarySize
                ),
                radius = secondarySize,
                center = Offset(highlightX, highlightY)
            )
        }
    }
    
    // Edge highlight - creates a subtle rim light effect
    if (phase == OrbPhase.THINKING || phase == OrbPhase.RESPONDING) {
        val edgeWidth = 2f + voiceIntensity * 1f
        val edgeAlpha = intensity * 0.4f * phaseProgress
        
        drawCircle(
            style = androidx.compose.ui.graphics.drawscope.Stroke(width = edgeWidth),
            brush = Brush.radialGradient(
                colors = listOf(
                    Color.White.copy(alpha = edgeAlpha),
                    Color.White.copy(alpha = edgeAlpha * 0.7f),
                    Color.White.copy(alpha = edgeAlpha * 0.3f),
                    Color.Transparent
                ),
                center = center,
                radius = radius + edgeWidth
            ),
            radius = radius,
            center = center
        )
    }
}

// ==== TRANSPARENT SPHERE WITH COLORED BORDER ====

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawTransparentSphereWithBorder(
    center: Offset,
    radius: Float,
    borderColors: List<Color>,
    waveColors: List<Color>,
    intensity: Float,
    breathPhase: Float,
    phase: OrbPhase,
    phaseProgress: Float,
    voiceIntensity: Float,
    wavePhase: Float
) {
    val sphereRadius = radius * 0.9f
    val borderWidth = 8f + voiceIntensity * 4f
    
    // 1. Draw transparent sphere interior (gần như background)
    drawCircle(
        brush = Brush.radialGradient(
            colors = listOf(
                Color.Transparent,
                Color.Transparent,
                Color.Transparent,
                Color.Transparent
            ),
            center = center,
            radius = sphereRadius
        ),
        radius = sphereRadius,
        center = center
    )
    
    // 2. Draw colored border rings
    val borderIntensity = intensity * (0.8f + voiceIntensity * 0.2f)
    
    // Outer border ring
    drawCircle(
        style = androidx.compose.ui.graphics.drawscope.Stroke(width = borderWidth),
        brush = Brush.radialGradient(
            colors = listOf(
                borderColors[0].copy(alpha = borderIntensity * 0.8f),
                borderColors[1].copy(alpha = borderIntensity * 0.6f),
                borderColors[2].copy(alpha = borderIntensity * 0.4f),
                Color.Transparent
            ),
            center = center,
            radius = sphereRadius
        ),
        radius = sphereRadius,
        center = center
    )
    
    // Inner border ring
    drawCircle(
        style = androidx.compose.ui.graphics.drawscope.Stroke(width = borderWidth * 0.6f),
        brush = Brush.radialGradient(
            colors = listOf(
                borderColors[1].copy(alpha = borderIntensity * 0.6f),
                borderColors[2].copy(alpha = borderIntensity * 0.4f),
                Color.Transparent
            ),
            center = center,
            radius = sphereRadius * 0.8f
        ),
        radius = sphereRadius * 0.8f,
        center = center
    )
    
    // 3. Draw simplified wave patterns (reduced complexity)
    val waveCount = 6 // Reduced from 12
    for (i in 0 until waveCount) {
        val waveAngle = i * (2f * PI.toFloat() / waveCount) + wavePhase
        val waveRadius = sphereRadius * (0.4f + sin(waveAngle + wavePhase) * 0.3f)
        val waveAlpha = intensity * (0.2f + voiceIntensity * 0.3f) * 0.5f
        
        // Wave ring
        drawCircle(
            style = androidx.compose.ui.graphics.drawscope.Stroke(width = 2f + voiceIntensity * 1f),
            color = waveColors[i % waveColors.size].copy(alpha = waveAlpha),
            radius = waveRadius,
            center = center
        )
    }
    
    // 4. Draw simplified concentric rings (reduced complexity)
    val ringCount = 3 // Reduced from 6
    for (i in 1 until ringCount) {
        val ringRadius = sphereRadius * (0.3f + i * 0.2f)
        val ringAlpha = intensity * (0.15f + voiceIntensity * 0.2f) * (1f - i.toFloat() / ringCount)
        
        drawCircle(
            style = androidx.compose.ui.graphics.drawscope.Stroke(width = 1f + voiceIntensity * 0.5f),
            color = waveColors[i % waveColors.size].copy(alpha = ringAlpha),
            radius = ringRadius,
            center = center
        )
    }
    
    // 5. Draw simplified ripple waves (only when active)
    if (phase == OrbPhase.LISTENING || phase == OrbPhase.RESPONDING) {
        val rippleCount = 4 // Reduced from 8
        for (i in 0 until rippleCount) {
            val rippleAngle = i * (2f * PI.toFloat() / rippleCount) + wavePhase
            val rippleRadius = sphereRadius * (0.5f + sin(rippleAngle * 2f) * 0.2f)
            val rippleAlpha = intensity * (0.3f + voiceIntensity * 0.3f) * phaseProgress
            
            drawCircle(
                style = androidx.compose.ui.graphics.drawscope.Stroke(width = 2f + voiceIntensity * 1f),
                color = waveColors[i % waveColors.size].copy(alpha = rippleAlpha),
                radius = rippleRadius,
                center = center
            )
        }
    }
    
    // 6. Draw breathing pulse on border
    val breathingIntensity = 0.7f + 0.3f * (sin(breathPhase) + 1f) / 2f
    val pulseRadius = sphereRadius * (1f + breathingIntensity * 0.05f)
    
    drawCircle(
        style = androidx.compose.ui.graphics.drawscope.Stroke(width = borderWidth * 0.3f),
        color = borderColors[0].copy(alpha = breathingIntensity * intensity * 0.6f),
        radius = pulseRadius,
        center = center
    )
}

// ==== LIQUID 3D SPHERE WITH ROTATION ====

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawLiquid3DSphere(
    center: Offset,
    radius: Float,
    borderColors: List<Color>,
    waveColors: List<Color>,
    intensity: Float,
    liquidRotationX: Float,
    liquidRotationY: Float,
    liquidRotationZ: Float,
    liquidWaves: Float,
    liquidTurbulence: Float,
    breathPhase: Float,
    phase: OrbPhase,
    phaseProgress: Float,
    voiceIntensity: Float,
    isListening: Boolean,
    isDarkMode: Boolean
) {
    val sphereRadius = radius * 0.9f
    val borderWidth = (8f + voiceIntensity * 3f).safeStroke()
    
    // ==== 3D ROTATION CALCULATIONS ====
    // Combine all 3 rotation axes for true 3D floating sphere
    val combinedRotation = liquidRotationX + liquidRotationY * 0.7f + liquidRotationZ * 0.5f
    val rotationX = liquidRotationX
    val rotationY = liquidRotationY
    val rotationZ = liquidRotationZ
    
    // 3D position calculations for highlights and effects
    val highlightAngleX = rotationX + PI.toFloat() / 4f
    val highlightAngleY = rotationY + PI.toFloat() / 6f
    val highlightAngleZ = rotationZ + PI.toFloat() / 8f
    
    // 1. Draw Ultra-Soft Liquid Core - BEAUTIFUL & FLUID - CẢI THIỆN VOICE RESPONSIVENESS
    val coreRadius = sphereRadius * (0.8f + voiceIntensity * 0.05f) // Thêm voice response vào radius
    val coreIntensity = intensity * (1.8f + voiceIntensity * 0.6f) // Tăng voice response
    
    // Ultra-soft gradient based on combined rotation
    val gradientOffset = combinedRotation * 0.3f
    val coreAlpha = coreIntensity * (1.08f + sin(breathPhase) * 0.36f) // +80% STRONGER
    
    // Multiple soft layers for liquid effect
    val liquidLayers = 5
    for (layer in 0 until liquidLayers) {
        val layerRadius = coreRadius * (0.6f + layer * 0.1f)
        val layerAlpha = coreAlpha * (0.8f - layer * 0.15f)
        val layerOffset = gradientOffset + layer * 0.2f
        
        drawCircle(
            brush = Brush.radialGradient(
                colors = listOf(
                    borderColors[3].copy(alpha = (layerAlpha * 0.9f).safeAlpha()),
                    borderColors[2].copy(alpha = (layerAlpha * 0.7f).safeAlpha()),
                    borderColors[1].copy(alpha = (layerAlpha * 0.5f).safeAlpha()),
                    borderColors[0].copy(alpha = (layerAlpha * 0.3f).safeAlpha()),
                    Color.Transparent
                ),
                center = center,
                radius = layerRadius.safeShaderRadius()
            ),
            radius = layerRadius,
            center = center
        )
    }
    
    // 2. Draw Liquid Surface with Water-like Movement
    val surfaceRadius = sphereRadius
    val surfaceWaves = liquidWaves + liquidTurbulence * 0.5f
    
    // Create liquid surface with multiple wave layers - FREE FLOATING
    val waveLayers = 3
    for (layer in 0 until waveLayers) {
        val layerOffset = layer * 0.3f
        val layerPhase = surfaceWaves + layerOffset
        val layerIntensity = coreIntensity * (0.4f - layer * 0.1f)
        
        // Wave pattern for liquid surface - TRUE 3D FLOATING
        val waveCount = 8 + (voiceIntensity * 4).toInt()
        for (i in 0 until waveCount) {
            // TRUE 3D waves: combine all rotation axes for realistic 3D sphere
            val waveAngle = i * (2f * PI.toFloat() / waveCount) + combinedRotation + liquidTurbulence * 0.4f
            val wavePhase = layerPhase + waveAngle * 0.5f
            
            // Ultra-soft liquid wave deformation
            val waveDeformation = sin(wavePhase) * 0.08f + sin(wavePhase * 2f) * 0.04f + sin(wavePhase * 3f) * 0.02f
            val waveRadius = surfaceRadius * (0.85f + waveDeformation)
            
            // Ultra-soft wave opacity based on phase and voice
            val waveAlpha = layerIntensity * when (phase) {
                OrbPhase.IDLE -> 0.2f
                OrbPhase.LISTENING -> 0.3f + phaseProgress * 0.15f
                OrbPhase.THINKING -> 0.4f + sin(phaseProgress * PI.toFloat() * 2) * 0.08f
                OrbPhase.RESPONDING -> 0.5f + (1f - phaseProgress) * 0.15f
            }
            
            // Draw ultra-soft liquid wave (no stroke, just fill)
            drawCircle(
                brush = Brush.radialGradient(
                    colors = listOf(
                        waveColors[i % waveColors.size].copy(alpha = (waveAlpha * 0.8f).safeAlpha()),
                        waveColors[(i + 1) % waveColors.size].copy(alpha = (waveAlpha * 0.6f).safeAlpha()),
                        waveColors[(i + 2) % waveColors.size].copy(alpha = (waveAlpha * 0.4f).safeAlpha()),
                        waveColors[(i + 3) % waveColors.size].copy(alpha = (waveAlpha * 0.2f).safeAlpha()),
                        Color.Transparent
                    ),
                    center = center,
                    radius = waveRadius.safeShaderRadius()
                ),
                radius = waveRadius,
                center = center
            )
        }
    }
    
    // 2.5. Draw Surface Particles - DARK COLORED DOTS COVERING ENTIRE SURFACE - CẢI THIỆN VOICE RESPONSIVENESS
    val particleCount = 200 + (voiceIntensity * 80).toInt() // Tăng particles và voice response
    for (i in 0 until particleCount) {
        // 3D particle position on sphere surface using Fibonacci spiral
        val particleAngleX = (i * 137.5f) % 360f * PI.toFloat() / 180f // Golden angle distribution
        val particleAngleY = (i * 89.3f) % 360f * PI.toFloat() / 180f
        val particleAngleZ = (i * 233.7f) % 360f * PI.toFloat() / 180f
        
        // TRUE 3D SPHERE PROJECTION - Always maintain spherical shape
        val sphereX = cos(particleAngleX) * sin(particleAngleY)
        val sphereY = sin(particleAngleX)
        val sphereZ = cos(particleAngleX) * cos(particleAngleY)
        
        // Apply 3D rotation to maintain sphere shape
        val rotatedX = sphereX * cos(rotationX) - sphereY * sin(rotationX)
        val rotatedY = sphereX * sin(rotationX) + sphereY * cos(rotationX)
        val rotatedZ = sphereZ
        
        // Apply Y rotation
        val finalX = rotatedX * cos(rotationY) - rotatedZ * sin(rotationY)
        val finalY = rotatedY
        val finalZ = rotatedX * sin(rotationY) + rotatedZ * cos(rotationY)
        
        // Apply Z rotation
        val finalRotatedX = finalX * cos(rotationZ) - finalY * sin(rotationZ)
        val finalRotatedY = finalX * sin(rotationZ) + finalY * cos(rotationZ)
        val finalRotatedZ = finalZ
        
        // Project to 2D screen position - ALWAYS SPHERICAL
        val particleX = center.x + finalRotatedX * surfaceRadius * 0.9f
        val particleY = center.y + finalRotatedY * surfaceRadius * 0.9f
        
        // Particle size and opacity based on 3D position and voice - DEPTH-BASED SIZING
        val depthFactor = (finalRotatedZ + 1f) / 2f // Convert from [-1,1] to [0,1]
        val particleSize = (2f + voiceIntensity * 1.5f + depthFactor * 1.5f).coerceIn(1f, 6f)
        val particleAlpha = (0.9f + voiceIntensity * 0.3f + depthFactor * 0.2f).coerceIn(0.4f, 1f)
        
        // Particle color based on position and phase - DARKER COLORS
        val particleColor = when (phase) {
            OrbPhase.IDLE -> borderColors[0].copy(alpha = particleAlpha.safeAlpha())
            OrbPhase.LISTENING -> borderColors[1].copy(alpha = (particleAlpha * 1.3f).safeAlpha())
            OrbPhase.THINKING -> borderColors[2].copy(alpha = (particleAlpha * 1.2f).safeAlpha())
            OrbPhase.RESPONDING -> borderColors[3].copy(alpha = (particleAlpha * 1.4f).safeAlpha())
        }
        
        // Draw particle with slight glow effect
        drawCircle(
            color = particleColor,
            radius = particleSize,
            center = Offset(particleX, particleY)
        )
        
        // Add small glow around particle for better visibility
        drawCircle(
            color = particleColor.copy(alpha = (particleAlpha * 0.3f).safeAlpha()),
            radius = particleSize * 1.5f,
            center = Offset(particleX, particleY)
        )
    }
    
    // 2.6. Draw Additional Small Particles for Density - CẢI THIỆN VOICE RESPONSIVENESS
    val smallParticleCount = 300 + (voiceIntensity * 100).toInt() // Tăng small particles và voice response
    for (i in 0 until smallParticleCount) {
        // Different distribution pattern for small particles
        val particleAngleX = (i * 222.5f) % 360f * PI.toFloat() / 180f
        val particleAngleY = (i * 144.7f) % 360f * PI.toFloat() / 180f
        val particleAngleZ = (i * 311.3f) % 360f * PI.toFloat() / 180f
        
        // TRUE 3D SPHERE PROJECTION for small particles - Always maintain spherical shape
        val sphereX = cos(particleAngleX) * sin(particleAngleY)
        val sphereY = sin(particleAngleX)
        val sphereZ = cos(particleAngleX) * cos(particleAngleY)
        
        // Apply 3D rotation to maintain sphere shape
        val rotatedX = sphereX * cos(rotationX) - sphereY * sin(rotationX)
        val rotatedY = sphereX * sin(rotationX) + sphereY * cos(rotationX)
        val rotatedZ = sphereZ
        
        // Apply Y rotation
        val finalX = rotatedX * cos(rotationY) - rotatedZ * sin(rotationY)
        val finalY = rotatedY
        val finalZ = rotatedX * sin(rotationY) + rotatedZ * cos(rotationY)
        
        // Apply Z rotation
        val finalRotatedX = finalX * cos(rotationZ) - finalY * sin(rotationZ)
        val finalRotatedY = finalX * sin(rotationZ) + finalY * cos(rotationZ)
        val finalRotatedZ = finalZ
        
        // Project to 2D screen position - ALWAYS SPHERICAL
        val particleX = center.x + finalRotatedX * surfaceRadius * 0.95f
        val particleY = center.y + finalRotatedY * surfaceRadius * 0.95f
        
        // Smaller particle size - DEPTH-BASED SIZING
        val depthFactor = (finalRotatedZ + 1f) / 2f // Convert from [-1,1] to [0,1]
        val particleSize = (1f + voiceIntensity * 0.8f + depthFactor * 1f).coerceIn(0.5f, 3f)
        val particleAlpha = (0.7f + voiceIntensity * 0.2f + depthFactor * 0.15f).coerceIn(0.3f, 0.9f)
        
        // Small particle color - darker
        val particleColor = when (phase) {
            OrbPhase.IDLE -> borderColors[1].copy(alpha = particleAlpha.safeAlpha())
            OrbPhase.LISTENING -> borderColors[2].copy(alpha = (particleAlpha * 1.2f).safeAlpha())
            OrbPhase.THINKING -> borderColors[3].copy(alpha = (particleAlpha * 1.1f).safeAlpha())
            OrbPhase.RESPONDING -> borderColors[0].copy(alpha = (particleAlpha * 1.3f).safeAlpha())
        }
        
        // Draw small particle
        drawCircle(
            color = particleColor,
            radius = particleSize,
            center = Offset(particleX, particleY)
        )
    }
    
    // 2.7. Draw Extra Dense Particles for Complete Coverage - CẢI THIỆN VOICE RESPONSIVENESS
    val extraParticleCount = 250 + (voiceIntensity * 70).toInt() // Tăng extra particles và voice response
    for (i in 0 until extraParticleCount) {
        // Third distribution pattern for maximum coverage
        val particleAngleX = (i * 99.7f) % 360f * PI.toFloat() / 180f
        val particleAngleY = (i * 177.3f) % 360f * PI.toFloat() / 180f
        val particleAngleZ = (i * 267.1f) % 360f * PI.toFloat() / 180f
        
        // TRUE 3D SPHERE PROJECTION for extra particles - Always maintain spherical shape
        val sphereX = cos(particleAngleX) * sin(particleAngleY)
        val sphereY = sin(particleAngleX)
        val sphereZ = cos(particleAngleX) * cos(particleAngleY)
        
        // Apply 3D rotation to maintain sphere shape
        val rotatedX = sphereX * cos(rotationX) - sphereY * sin(rotationX)
        val rotatedY = sphereX * sin(rotationX) + sphereY * cos(rotationX)
        val rotatedZ = sphereZ
        
        // Apply Y rotation
        val finalX = rotatedX * cos(rotationY) - rotatedZ * sin(rotationY)
        val finalY = rotatedY
        val finalZ = rotatedX * sin(rotationY) + rotatedZ * cos(rotationY)
        
        // Apply Z rotation
        val finalRotatedX = finalX * cos(rotationZ) - finalY * sin(rotationZ)
        val finalRotatedY = finalX * sin(rotationZ) + finalY * cos(rotationZ)
        val finalRotatedZ = finalZ
        
        // Project to 2D screen position - ALWAYS SPHERICAL
        val particleX = center.x + finalRotatedX * surfaceRadius * 0.92f
        val particleY = center.y + finalRotatedY * surfaceRadius * 0.92f
        
        // Extra particle size and opacity - DEPTH-BASED SIZING
        val depthFactor = (finalRotatedZ + 1f) / 2f // Convert from [-1,1] to [0,1]
        val particleSize = (1.5f + voiceIntensity * 1f + depthFactor * 1.2f).coerceIn(0.8f, 4f)
        val particleAlpha = (0.6f + voiceIntensity * 0.25f + depthFactor * 0.2f).coerceIn(0.2f, 0.8f)
        
        // Extra particle color - darker and more varied
        val particleColor = when (phase) {
            OrbPhase.IDLE -> borderColors[2].copy(alpha = particleAlpha.safeAlpha())
            OrbPhase.LISTENING -> borderColors[3].copy(alpha = (particleAlpha * 1.1f).safeAlpha())
            OrbPhase.THINKING -> borderColors[0].copy(alpha = (particleAlpha * 1.05f).safeAlpha())
            OrbPhase.RESPONDING -> borderColors[1].copy(alpha = (particleAlpha * 1.15f).safeAlpha())
        }
        
        // Draw extra particle
        drawCircle(
            color = particleColor,
            radius = particleSize,
            center = Offset(particleX, particleY)
        )
        
        // Add tiny glow for extra particles
        drawCircle(
            color = particleColor.copy(alpha = (particleAlpha * 0.2f).safeAlpha()),
            radius = particleSize * 1.3f,
            center = Offset(particleX, particleY)
        )
    }
    
    // 3. Draw Soft Liquid Border - ULTRA SOFT & BEAUTIFUL - CẢI THIỆN VOICE RESPONSIVENESS
    val borderIntensity = intensity * (1.6f + voiceIntensity * 0.8f) // Tăng voice response
    val clickEffect = if (isListening) (1.5f + voiceIntensity * 0.3f) else (1.0f + voiceIntensity * 0.1f) // Thêm voice response
    
    // Ultra-soft outer glow (no hard border)
    val outerGlowRadius = sphereRadius * (1.15f + clickEffect * 0.1f)
    drawCircle(
        brush = Brush.radialGradient(
            colors = listOf(
                borderColors[0].copy(alpha = (borderIntensity * 0.3f).safeAlpha()),
                borderColors[1].copy(alpha = (borderIntensity * 0.2f).safeAlpha()),
                borderColors[2].copy(alpha = (borderIntensity * 0.1f).safeAlpha()),
                Color.Transparent
            ),
            center = center,
            radius = outerGlowRadius.safeShaderRadius()
        ),
        radius = outerGlowRadius,
        center = center
    )
    
    // Soft liquid edge (very subtle)
    val edgeRadius = sphereRadius * (1.02f + clickEffect * 0.05f)
    drawCircle(
        brush = Brush.radialGradient(
            colors = listOf(
                borderColors[0].copy(alpha = (borderIntensity * 0.6f).safeAlpha()),
                borderColors[1].copy(alpha = (borderIntensity * 0.4f).safeAlpha()),
                borderColors[2].copy(alpha = (borderIntensity * 0.2f).safeAlpha()),
                Color.Transparent
            ),
            center = center,
            radius = edgeRadius.safeShaderRadius()
        ),
        radius = edgeRadius,
        center = center
    )
    
    // 4. Draw Liquid Highlights for 3D Effect - CẢI THIỆN VOICE RESPONSIVENESS - TRUE 3D FLOATING
    val highlightIntensity = intensity * (1.2f + voiceIntensity * 0.4f) // Tăng voice response
    
    // TRUE 3D highlights: combine all 3 rotation axes
    val highlightAngle = highlightAngleX + highlightAngleY * 0.6f + highlightAngleZ * 0.4f
    val highlightDistance = sphereRadius * 0.6f
    
    // 3D position calculation with depth
    val highlightX = center.x + cos(highlightAngle) * highlightDistance * cos(highlightAngleY)
    val highlightY = center.y + sin(highlightAngle) * highlightDistance * sin(highlightAngleX)
    
    // Ultra-soft main highlight
    drawCircle(
        brush = Brush.radialGradient(
            colors = listOf(
                Color.White.copy(alpha = (highlightIntensity * 0.8f).safeAlpha()),
                Color.White.copy(alpha = (highlightIntensity * 0.4f).safeAlpha()),
                Color.White.copy(alpha = (highlightIntensity * 0.2f).safeAlpha()),
                Color.Transparent
            ),
            center = Offset(highlightX, highlightY),
            radius = sphereRadius * 0.4f
        ),
        radius = sphereRadius * 0.4f,
        center = Offset(highlightX, highlightY)
    )
    
    // Secondary highlights - TRUE 3D FLOATING
    val secondaryHighlights = 3
    for (i in 0 until secondaryHighlights) {
        // TRUE 3D secondary highlights: combine all rotation axes
        val highlightAngle2 = highlightAngle + liquidTurbulence * 0.3f + i * (2f * PI.toFloat() / secondaryHighlights)
        val highlightDistance2 = sphereRadius * (0.4f + i * 0.1f)
        
        // 3D position calculation with different depth for each highlight
        val depthFactor = 0.8f + i * 0.1f
        val highlightX2 = center.x + cos(highlightAngle2) * highlightDistance2 * cos(highlightAngleY * depthFactor)
        val highlightY2 = center.y + sin(highlightAngle2) * highlightDistance2 * sin(highlightAngleX * depthFactor)
        
        drawCircle(
            brush = Brush.radialGradient(
                colors = listOf(
                    Color.White.copy(alpha = (highlightIntensity * 0.2f).safeAlpha()),
                    Color.White.copy(alpha = (highlightIntensity * 0.1f).safeAlpha()),
                    Color.Transparent
                ),
                center = Offset(highlightX2, highlightY2),
                radius = sphereRadius * (0.15f + i * 0.08f)
            ),
            radius = sphereRadius * (0.15f + i * 0.08f),
            center = Offset(highlightX2, highlightY2)
        )
    }
    
    // 5. Draw Liquid Ripples - CẢI THIỆN VOICE RESPONSIVENESS - TRUE 3D FLOATING
    if (phase != OrbPhase.IDLE) {
        val rippleCount = 6 + (voiceIntensity * 3).toInt() // Tăng ripple count và voice response
        for (i in 0 until rippleCount) {
            // TRUE 3D ripples: combine all rotation axes for realistic 3D sphere
            val rippleAngle = i * (2f * PI.toFloat() / rippleCount) + combinedRotation + liquidTurbulence * 0.2f
            val ripplePhase = liquidWaves + i * 0.5f
            val rippleRadius = sphereRadius * (0.3f + sin(ripplePhase) * 0.2f)
            val rippleAlpha = intensity * (0.36f + voiceIntensity * 0.54f) * phaseProgress // +80% STRONGER
            
            // Ultra-soft ripples (no stroke, just fill)
            drawCircle(
                brush = Brush.radialGradient(
                    colors = listOf(
                        waveColors[i % waveColors.size].copy(alpha = (rippleAlpha * 0.6f).safeAlpha()),
                        waveColors[(i + 1) % waveColors.size].copy(alpha = (rippleAlpha * 0.4f).safeAlpha()),
                        waveColors[(i + 2) % waveColors.size].copy(alpha = (rippleAlpha * 0.2f).safeAlpha()),
                        Color.Transparent
                    ),
                    center = center,
                    radius = rippleRadius.safeShaderRadius()
                ),
                radius = rippleRadius,
                center = center
            )
        }
    }
    
    // 6. Draw Ultra-Soft Liquid Breathing Pulse - CẢI THIỆN VOICE RESPONSIVENESS
    val breathingIntensity = (1.2f + 0.8f * (sin(breathPhase) + 1f) / 2f + voiceIntensity * 0.3f).safeAlpha() // Tăng voice response
    val pulseRadius = sphereRadius * (1f + breathingIntensity * (0.08f + voiceIntensity * 0.04f)) // Thêm voice response vào pulse radius
    
    // Ultra-soft breathing pulse (no stroke, just fill)
    drawCircle(
        brush = Brush.radialGradient(
            colors = listOf(
                borderColors[0].copy(alpha = (breathingIntensity * intensity * 0.4f).safeAlpha()),
                borderColors[1].copy(alpha = (breathingIntensity * intensity * 0.3f).safeAlpha()),
                borderColors[2].copy(alpha = (breathingIntensity * intensity * 0.2f).safeAlpha()),
                Color.Transparent
            ),
            center = center,
            radius = pulseRadius.safeShaderRadius()
        ),
        radius = pulseRadius,
        center = center
    )
}

// ==== SIMPLIFIED TRANSPARENT SPHERE (CRASH-SAFE) ====

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawSimpleTransparentSphere(
    center: Offset,
    radius: Float,
    borderColors: List<Color>,
    waveColors: List<Color>,
    intensity: Float,
    breathPhase: Float,
    phase: OrbPhase,
    phaseProgress: Float,
    voiceIntensity: Float,
    wavePhase: Float
) {
    val sphereRadius = radius * 0.9f
    val borderWidth = 6f + voiceIntensity * 2f // Reduced width
    
    // 1. Draw transparent sphere interior
    drawCircle(
        brush = Brush.radialGradient(
            colors = listOf(
                Color.Transparent,
                Color.Transparent,
                Color.Transparent,
                Color.Transparent
            ),
            center = center,
            radius = sphereRadius
        ),
        radius = sphereRadius,
        center = center
    )
    
    // 2. Draw simple colored border
    val borderIntensity = intensity * (0.7f + voiceIntensity * 0.2f)
    
    drawCircle(
        style = androidx.compose.ui.graphics.drawscope.Stroke(width = borderWidth),
        brush = Brush.radialGradient(
            colors = listOf(
                borderColors[0].copy(alpha = borderIntensity * 0.8f),
                borderColors[1].copy(alpha = borderIntensity * 0.6f),
                borderColors[2].copy(alpha = borderIntensity * 0.4f),
                Color.Transparent
            ),
            center = center,
            radius = sphereRadius
        ),
        radius = sphereRadius,
        center = center
    )
    
    // 3. Draw simple wave patterns (reduced complexity)
    val waveCount = 4 // Very reduced
    for (i in 0 until waveCount) {
        val waveAngle = i * (2f * PI.toFloat() / waveCount) + wavePhase
        val waveRadius = sphereRadius * (0.5f + sin(waveAngle + wavePhase) * 0.2f)
        val waveAlpha = intensity * (0.15f + voiceIntensity * 0.2f) * 0.5f
        
        drawCircle(
            style = androidx.compose.ui.graphics.drawscope.Stroke(width = 1.5f + voiceIntensity * 0.5f),
            color = waveColors[i % waveColors.size].copy(alpha = waveAlpha),
            radius = waveRadius,
            center = center
        )
    }
    
    // 4. Draw simple breathing pulse
    val breathingIntensity = 0.6f + 0.2f * (sin(breathPhase) + 1f) / 2f
    val pulseRadius = sphereRadius * (1f + breathingIntensity * 0.03f)
    
    drawCircle(
        style = androidx.compose.ui.graphics.drawscope.Stroke(width = borderWidth * 0.2f),
        color = borderColors[0].copy(alpha = breathingIntensity * intensity * 0.4f),
        radius = pulseRadius,
        center = center
    )
}
