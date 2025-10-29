@file:Suppress("UnusedImport")

package com.auto_fe.auto_fe.ui.components

import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.blur
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.auto_fe.auto_fe.ui.theme.*
import kotlin.math.min
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

/**
 * AI Breathing Control Buttons - "hơi thở" giọng nói
 * 
 * Tone chủ đạo: Minimalist, Deep-Soft, cảm giác "AI sống động"
 * Features:
 * - Voice-responsive breathing animations
 * - Glass morphism effects
 * - Soft blue-purple gradient
 * - Haptic feedback với voice rhythm
 */
@Composable
fun SoftControlButtons(
    isRecording: Boolean,
    voiceLevel: Int,
    onRecordingToggle: () -> Unit,
    onAgainClick: () -> Unit,
    onTranscriptOpen: () -> Unit,
    modifier: Modifier = Modifier
) {
    val haptic = LocalHapticFeedback.current

    // AI Breathing Animation System
    val transition = rememberInfiniteTransition(label = "ai_breathing")
    
    // Breathing scale - "hơi thở" giọng nói
    val breathScale by transition.animateFloat(
        initialValue = 0.98f,
        targetValue = when {
            isRecording -> 1.05f + voiceLevel * 0.02f
            else -> 1.02f
        },
        animationSpec = infiniteRepeatable(
            animation = tween(
                durationMillis = when {
                    isRecording -> 1200 + voiceLevel * 200
                    else -> 3000
                },
                easing = CubicBezierEasing(0.25f, 0.1f, 0.25f, 1.0f)
            ),
            repeatMode = RepeatMode.Reverse
        ),
        label = "breath"
    )

    // AI Glass morphism background
    Card(
        colors = CardDefaults.cardColors(
            containerColor = GlassBackground
        ),
        shape = RoundedCornerShape(28.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp),
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 24.dp, vertical = 16.dp)
            .scale(breathScale)
    ) {
        Row(
            Modifier
                .fillMaxWidth()
                .padding(horizontal = 24.dp, vertical = 20.dp),
            horizontalArrangement = Arrangement.SpaceEvenly,
            verticalAlignment = Alignment.CenterVertically
        ) {
            // AI Minimalist Secondary Button
            AIBreathingButton(
                iconText = "🔄",
                onClick = {
                    haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                    onAgainClick()
                },
                isActive = false
            )

            // AI Main Record Button - "hơi thở" giọng nói
            AIRecordButton(
                isRecording = isRecording,
                voiceLevel = voiceLevel,
                onClick = {
                    haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                    onRecordingToggle()
                }
            )

            // AI Minimalist Secondary Button
            AIBreathingButton(
                iconText = "📋",
                onClick = {
                    haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                    onTranscriptOpen()
                },
                isActive = false
            )
        }
    }
}

// ==== AI VISUAL IDENTITY BUTTON COMPONENTS ====

@Composable
private fun AIRecordButton(
    isRecording: Boolean,
    voiceLevel: Int,
    onClick: () -> Unit
) {
    val transition = rememberInfiniteTransition(label = "record_breathing")
    
    // Breathing animation theo voice level
    val breathScale by transition.animateFloat(
        initialValue = 1.0f,
        targetValue = when {
            isRecording -> 1.1f + voiceLevel * 0.05f
            else -> 1.05f
        },
        animationSpec = infiniteRepeatable(
            animation = tween(
                durationMillis = when {
                    isRecording -> 800 + voiceLevel * 200
                    else -> 2000
                },
                easing = CubicBezierEasing(0.25f, 0.1f, 0.25f, 1.0f)
            ),
            repeatMode = RepeatMode.Reverse
        ),
        label = "recordBreath"
    )
    
    // Glow intensity theo voice level
    val glowIntensity by transition.animateFloat(
        initialValue = 0.3f,
        targetValue = when {
            isRecording -> 0.7f + voiceLevel * 0.1f
            else -> 0.5f
        },
        animationSpec = infiniteRepeatable(
            animation = tween(
                durationMillis = when {
                    isRecording -> 1000 + voiceLevel * 300
                    else -> 2500
                },
                easing = EaseInOutCubic
            ),
            repeatMode = RepeatMode.Reverse
        ),
        label = "recordGlow"
    )

    Box(
        contentAlignment = Alignment.Center,
        modifier = Modifier
            .size(80.dp)
            .scale(breathScale)
            .clickable { onClick() }
            .semantics { contentDescription = if (isRecording) "Stop recording" else "Start recording" }
    ) {
        // AI Glow layer
        Canvas(modifier = Modifier.fillMaxSize()) {
            val center = Offset(size.width / 2f, size.height / 2f)
            val radius = size.minDimension / 2f
            
            // Voice-responsive glow
            drawCircle(
                brush = Brush.radialGradient(
                    colors = listOf(
                        if (isRecording) VoiceActive.copy(alpha = glowIntensity) else VoiceIdle.copy(alpha = glowIntensity),
                        Color.Transparent
                    ),
                    center = center,
                    radius = radius * 1.5f
                ),
                radius = radius * 1.5f,
                center = center
            )
        }
        
        // AI Glass morphism button
        Card(
            colors = CardDefaults.cardColors(
                containerColor = GlassBackground
            ),
            shape = CircleShape,
            elevation = CardDefaults.cardElevation(defaultElevation = 0.dp),
            modifier = Modifier.size(64.dp)
        ) {
            Box(
                contentAlignment = Alignment.Center,
                modifier = Modifier.fillMaxSize()
            ) {
                Text(
                    text = if (isRecording) "⏹" else "🎤",
                    style = MaterialTheme.typography.headlineMedium,
                    color = if (isRecording) VoiceError else AITextPrimary
                )
            }
        }
    }
}

@Composable
private fun AIBreathingButton(
    iconText: String,
    onClick: () -> Unit,
    isActive: Boolean = false
) {
    val transition = rememberInfiniteTransition(label = "secondary_breathing")
    
    // Subtle breathing animation
    val breathScale by transition.animateFloat(
        initialValue = 1.0f,
        targetValue = if (isActive) 1.05f else 1.02f,
        animationSpec = infiniteRepeatable(
            animation = tween(
                durationMillis = 3000,
                easing = CubicBezierEasing(0.25f, 0.1f, 0.25f, 1.0f)
            ),
            repeatMode = RepeatMode.Reverse
        ),
        label = "secondaryBreath"
    )

    Box(
        contentAlignment = Alignment.Center,
        modifier = Modifier
            .size(56.dp)
            .scale(breathScale)
            .clickable { onClick() }
    ) {
        // AI Glass morphism secondary button
        Card(
            colors = CardDefaults.cardColors(
                containerColor = GlassBackground
            ),
            shape = CircleShape,
            elevation = CardDefaults.cardElevation(defaultElevation = 0.dp),
            modifier = Modifier.size(48.dp)
        ) {
            Box(
                contentAlignment = Alignment.Center,
                modifier = Modifier.fillMaxSize()
            ) {
                Text(
                    text = iconText,
                    style = MaterialTheme.typography.titleMedium,
                    color = AITextSecondary
                )
            }
        }
    }
}
