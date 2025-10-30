package com.auto_fe.auto_fe.ui.screens.layers

import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.blur
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.auto_fe.auto_fe.ui.theme.*
import kotlin.math.*

/**
 * InteractionLayer - Smooth & Responsive Controls
 * 
 * Refactored Features:
 * - Smooth breathing animations cho mic button
 * - Responsive feedback với haptics
 * - Organic visual transitions
 * - Glass morphism design language
 */
@Composable
fun InteractionLayer(
    isRecording: Boolean,
    voiceLevel: Int,
    isDarkMode: Boolean = true,
    onRecordingToggle: () -> Unit,
    onModeToggle: () -> Unit,
    onTranscriptOpen: () -> Unit,
    modifier: Modifier = Modifier
) {
    val haptic = LocalHapticFeedback.current
    val normalizedLevel = remember(voiceLevel) { 
        (voiceLevel.coerceIn(0, 3) / 3f) 
    }

    Box(modifier = modifier.fillMaxSize()) {
        // Sphere is now clickable - no separate mic button needed

        // Secondary controls - bottom
        Row(
            Modifier
                .fillMaxWidth()
                .padding(horizontal = 32.dp, vertical = 32.dp)
                .align(Alignment.BottomCenter),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            SmoothSecondaryButton(
                label = if (isDarkMode) "Sáng" else "Tối",
                icon = if (isDarkMode) "☀️" else "🌙",
                onClick = {
                    haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                    onModeToggle()
                }
            )
            
            // Spacer(Modifier.width(20.dp))
            
            // SmoothSecondaryButton(
            //     label = "",
            //     icon = "📝",
            //     onClick = {
            //         haptic.performHapticFeedback(HapticFeedbackType.LongPress)
            //         onTranscriptOpen()
            //     }
            // )
        }
    }
}

@Composable
private fun SmoothMicButton(
    isRecording: Boolean,
    voiceLevel: Float,
    onClick: () -> Unit
) {
    // Smooth breathing animation
    val breathTransition = rememberInfiniteTransition(label = "mic_breath")
    
    val breathPhase by breathTransition.animateFloat(
        initialValue = 0f,
        targetValue = 2f * PI.toFloat(),
        animationSpec = infiniteRepeatable(
            animation = tween(
                durationMillis = if (isRecording) 2000 else 3500,
                easing = LinearEasing
            )
        ),
        label = "breathPhase"
    )
    
    val breathScale = 1f + 0.08f * (sin(breathPhase) + 1f) / 2f
    val breathIntensity = 0.7f + 0.3f * (sin(breathPhase * 0.5f) + 1f) / 2f

    // Voice-responsive scaling
    val voiceScale by animateFloatAsState(
        targetValue = 1f + voiceLevel * 0.15f,
        animationSpec = spring(
            dampingRatio = Spring.DampingRatioMediumBouncy,
            stiffness = Spring.StiffnessLow
        ),
        label = "voiceScale"
    )

    val combinedScale = breathScale * voiceScale

    Box(
        modifier = Modifier
            .size(180.dp)
            .scale(combinedScale),
        contentAlignment = Alignment.Center
    ) {
        // Outer glow aura
        Box(
            modifier = Modifier
                .size(180.dp)
                .alpha(breathIntensity * (0.4f + voiceLevel * 0.3f))
                .blur((20 + voiceLevel * 15).dp)
                .background(
                    brush = Brush.radialGradient(
                        colors = listOf(
                            SphereGlow.copy(alpha = 0.8f),
                            SphereAura.copy(alpha = 0.6f),
                            Color.Transparent
                        )
                    ),
                    shape = CircleShape
                )
        )

        // Glass morphism body
        Surface(
            modifier = Modifier
                .size(140.dp)
                .clip(CircleShape)
                .clickable(
                    interactionSource = remember { MutableInteractionSource() },
                    indication = null,
                    onClick = onClick
                ),
            color = GlassBackground.copy(alpha = 0.95f),
            tonalElevation = 0.dp,
            shadowElevation = 0.dp
        ) {
            Box(contentAlignment = Alignment.Center) {
                // Energy ring indicator
                Canvas(Modifier.fillMaxSize()) {
                    val strokeWidth = size.minDimension * 0.06f
                    val ringRadius = (size.minDimension - strokeWidth) / 2f
                    
                    // Background ring
                    drawCircle(
                        color = AITextPrimary.copy(alpha = 0.15f),
                        radius = ringRadius,
                        style = Stroke(width = strokeWidth)
                    )
                    
                    // Active energy ring
                    val sweepAngle = if (isRecording) {
                        60f + 280f * voiceLevel
                    } else {
                        90f * breathIntensity
                    }
                    
                    drawArc(
                        brush = Brush.sweepGradient(
                            colors = listOf(
                                GoldenCore,
                                GoldenInner,
                                GoldenMiddle,
                                GoldenOuter,
                                GoldenCore
                            )
                        ),
                        startAngle = -90f,
                        sweepAngle = sweepAngle,
                        useCenter = false,
                        style = Stroke(width = strokeWidth)
                    )
                }
            }
        }

        // Center icon
        Text(
            text = if (isRecording) "🎤" else "🎙️",
            style = MaterialTheme.typography.displaySmall,
            modifier = Modifier.alpha(breathIntensity)
        )
    }
}

@Composable
private fun SmoothSecondaryButton(
    label: String,
    icon: String,
    onClick: () -> Unit
) {
    var isPressed by remember { mutableStateOf(false) }
    
    val pressScale by animateFloatAsState(
        targetValue = if (isPressed) 0.95f else 1f,
        animationSpec = spring(
            dampingRatio = Spring.DampingRatioMediumBouncy,
            stiffness = Spring.StiffnessMedium
        ),
        label = "pressScale"
    )

    Surface(
        modifier = Modifier
            .height(64.dp)
            .scale(pressScale)
            .clip(CircleShape)
            .clickable(
                interactionSource = remember { MutableInteractionSource() },
                indication = null,
                onClick = onClick
            ),
        color = GlassBackground.copy(alpha = 0.9f),
        tonalElevation = 0.dp,
        shadowElevation = 0.dp
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 24.dp, vertical = 16.dp),
            horizontalArrangement = Arrangement.spacedBy(10.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = icon,
                style = MaterialTheme.typography.titleLarge
            )
            Text(
                text = label,
                color = AITextPrimary,
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Medium
            )
        }
    }
}