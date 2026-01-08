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
import androidx.compose.ui.unit.dp
import com.auto_fe.auto_fe.ui.theme.*
import kotlin.math.*
import androidx.compose.ui.unit.sp

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
    confirmationQuestion: String,
    successMessage: String,
    errorMessage: String,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier.fillMaxSize()
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier.fillMaxSize()
        ) {
            // Smooth status indicator
            SmoothStatusCard(
                isRecording = isRecording,
                modifier = Modifier.padding(top = 60.dp, bottom = 20.dp)
            )
        }
        
        // Success/Error/Confirmation messages ở gần bottom (phía trên BottomNav)
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Bottom,
            modifier = Modifier
                .fillMaxSize()
                .padding(bottom = 10.dp) // Khoảng cách từ bottom để tránh BottomNav
        ) {
            // Smooth confirmation display (ưu tiên hiển thị trước)
            AnimatedVisibility(
                visible = confirmationQuestion.isNotEmpty(),
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
                SmoothConfirmationCard(question = confirmationQuestion)
            }

            // Smooth success display
            AnimatedVisibility(
                visible = successMessage.isNotEmpty() && confirmationQuestion.isEmpty(),
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
                SmoothSuccessCard(message = successMessage)
            }

            // Smooth error display
            AnimatedVisibility(
                visible = errorMessage.isNotEmpty() && confirmationQuestion.isEmpty(),
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
                style = MaterialTheme.typography.bodyLarge.copy(fontSize = 22.sp, lineHeight = 28.sp),
                color = AITextPrimary,
                fontWeight = FontWeight.Medium
            )
        }
    }
}

@Composable
private fun SmoothConfirmationCard(
    question: String
) {
    Card(
        colors = CardDefaults.cardColors(
            containerColor = AIWarning.copy(alpha = 0.15f)
        ),
        shape = RoundedCornerShape(20.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp),
        modifier = Modifier
            .padding(horizontal = 32.dp, vertical = 12.dp)
            .fillMaxWidth(0.9f)
    ) {
        Row(
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.padding(horizontal = 28.dp, vertical = 20.dp)
        ) {
            Text(
                text = "❓",
                style = MaterialTheme.typography.titleLarge
            )
            Text(
                text = question,
                style = MaterialTheme.typography.bodyLarge.copy(fontSize = 22.sp, lineHeight = 28.sp),
                color = AIWarning
            )
        }
    }
}

@Composable
private fun SmoothSuccessCard(
    message: String
) {
    Card(
        colors = CardDefaults.cardColors(
            containerColor = AISuccess.copy(alpha = 0.15f)
        ),
        shape = RoundedCornerShape(20.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp),
        modifier = Modifier
            .padding(horizontal = 32.dp, vertical = 12.dp)
            .fillMaxWidth(0.9f)
    ) {
        Row(
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.padding(horizontal = 28.dp, vertical = 20.dp)
        ) {
            Text(
                text = "✓",
                style = MaterialTheme.typography.titleLarge
            )
            Text(
                text = message,
                style = MaterialTheme.typography.bodyLarge.copy(fontSize = 22.sp, lineHeight = 28.sp),
                color = AISuccess
            )
        }
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
                style = MaterialTheme.typography.bodyLarge.copy(fontSize = 22.sp, lineHeight = 28.sp),
                color = AIError
            )
        }
    }
}