package com.auto_fe.auto_fe.ui.screens.layers

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.*
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.scale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.auto_fe.auto_fe.ui.theme.*
import kotlin.math.*

/**
 * OverlayLayer - Smooth Text & Status Display
 * 
 * Refactored Features:
 * - Smooth entrance/exit animations
 * - Breathing status indicators
 * - Glass morphism cards
 * - Minimalist & elegant design
 */
@Composable
fun OverlayLayer(
    isRecording: Boolean,
    transcriptText: String,
    errorMessage: String,
    modifier: Modifier = Modifier
) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = modifier.fillMaxSize()
    ) {
        // Smooth status indicator
        SmoothStatusCard(
            isRecording = isRecording,
            modifier = Modifier.padding(top = 60.dp, bottom = 20.dp)
        )

        // Smooth transcript display
        AnimatedVisibility(
            visible = transcriptText.isNotEmpty(),
            enter = fadeIn(animationSpec = tween(400, easing = EaseOutCubic)) +
                    slideInVertically(
                        animationSpec = tween(400, easing = EaseOutCubic),
                        initialOffsetY = { -it / 2 }
                    ),
            exit = fadeOut(animationSpec = tween(300, easing = EaseInCubic)) +
                   slideOutVertically(
                       animationSpec = tween(300, easing = EaseInCubic),
                       targetOffsetY = { -it / 2 }
                   )
        ) {
            SmoothTranscriptCard(text = transcriptText)
        }

        // Smooth error display
        AnimatedVisibility(
            visible = errorMessage.isNotEmpty(),
            enter = fadeIn(animationSpec = tween(400, easing = EaseOutCubic)) +
                    slideInVertically(
                        animationSpec = tween(400, easing = EaseOutCubic),
                        initialOffsetY = { it / 2 }
                    ),
            exit = fadeOut(animationSpec = tween(300, easing = EaseInCubic)) +
                   slideOutVertically(
                       animationSpec = tween(300, easing = EaseInCubic),
                       targetOffsetY = { it / 2 }
                   )
        ) {
            SmoothErrorCard(message = errorMessage)
        }
    }
}

@Composable
private fun SmoothStatusCard(
    isRecording: Boolean,
    modifier: Modifier = Modifier
) {
    // Smooth breathing animation
    val breathTransition = rememberInfiniteTransition(label = "status_breath")
    
    val breathPhase by breathTransition.animateFloat(
        initialValue = 0f,
        targetValue = 2f * PI.toFloat(),
        animationSpec = infiniteRepeatable(
            animation = tween(
                durationMillis = if (isRecording) 2000 else 4000,
                easing = LinearEasing
            )
        ),
        label = "breathPhase"
    )
    
    val breathScale = 1f + 0.04f * (sin(breathPhase) + 1f) / 2f
    val breathAlpha = 0.85f + 0.15f * (sin(breathPhase * 0.5f) + 1f) / 2f

    Card(
        colors = CardDefaults.cardColors(
            containerColor = GlassBackground.copy(alpha = 0.95f)
        ),
        shape = RoundedCornerShape(24.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp),
        modifier = modifier
            .scale(breathScale)
            .alpha(breathAlpha)
            .padding(horizontal = 24.dp)
    ) {
        Row(
            horizontalArrangement = Arrangement.spacedBy(14.dp),
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.padding(horizontal = 24.dp, vertical = 14.dp)
        ) {
            // Animated icon
            Text(
                text = if (isRecording) "🎤" else "💫",
                style = MaterialTheme.typography.headlineSmall,
                modifier = Modifier.alpha(breathAlpha)
            )
            
            // Status text
            Text(
                text = if (isRecording) "Đang nghe..." else "Sẵn sàng",
                style = MaterialTheme.typography.bodyLarge,
                color = AITextPrimary,
                fontWeight = FontWeight.Medium
            )
        }
    }
}

@Composable
private fun SmoothTranscriptCard(
    text: String
) {
    Card(
        colors = CardDefaults.cardColors(
            containerColor = GlassBackground.copy(alpha = 0.92f)
        ),
        shape = RoundedCornerShape(20.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp),
        modifier = Modifier
            .padding(horizontal = 32.dp, vertical = 12.dp)
            .fillMaxWidth(0.9f)
    ) {
        Text(
            text = text,
            style = MaterialTheme.typography.bodyLarge,
            color = AITextPrimary,
            textAlign = TextAlign.Center,
            lineHeight = MaterialTheme.typography.bodyLarge.lineHeight * 1.4f,
            modifier = Modifier.padding(horizontal = 28.dp, vertical = 20.dp)
        )
    }
}

@Composable
private fun SmoothErrorCard(
    message: String
) {
    // Gentle shake animation for errors
    val shakeTransition = rememberInfiniteTransition(label = "error_shake")
    
    val shakeOffset by shakeTransition.animateFloat(
        initialValue = -2f,
        targetValue = 2f,
        animationSpec = infiniteRepeatable(
            animation = tween(
                durationMillis = 100,
                easing = LinearEasing
            ),
            repeatMode = RepeatMode.Reverse
        ),
        label = "shakeOffset"
    )

    Card(
        colors = CardDefaults.cardColors(
            containerColor = AIError.copy(alpha = 0.12f)
        ),
        shape = RoundedCornerShape(20.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp),
        modifier = Modifier
            .padding(horizontal = 32.dp, vertical = 12.dp)
            .fillMaxWidth(0.9f)
            .offset(x = shakeOffset.dp)
    ) {
        Row(
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.padding(horizontal = 28.dp, vertical = 20.dp)
        ) {
            Text(
                text = "⚠️",
                style = MaterialTheme.typography.titleLarge
            )
            Text(
                text = message,
                style = MaterialTheme.typography.bodyLarge,
                color = AIError,
                lineHeight = MaterialTheme.typography.bodyLarge.lineHeight * 1.4f
            )
        }
    }
}