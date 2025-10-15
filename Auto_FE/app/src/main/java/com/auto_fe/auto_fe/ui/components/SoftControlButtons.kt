@file:Suppress("UnusedImport")

package com.auto_fe.auto_fe

import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
// Removed Material Icons imports - using Text/Emoji instead
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
import kotlin.math.min
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

@Composable
fun SoftControlButtons(
    isRecording: Boolean,
    voiceLevel: Int,
    onRecordingToggle: () -> Unit,
    onAgainClick: () -> Unit,
    onTranscriptOpen: () -> Unit,
    modifier: Modifier = Modifier
) {
    val cc = MaterialTheme.colorScheme
    val haptic = LocalHapticFeedback.current

    // Nền cụm nút: mỏng hơn, ít "nhựa"
    Surface(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 24.dp),
        color = cc.surface.copy(alpha = 0.45f),
        shape = RoundedCornerShape(20.dp),
        shadowElevation = 6.dp,
        tonalElevation = 0.dp
    ) {
        Row(
            Modifier
                .fillMaxWidth()
                .padding(horizontal = 20.dp, vertical = 16.dp),
            horizontalArrangement = Arrangement.SpaceEvenly,
            verticalAlignment = Alignment.CenterVertically
        ) {
            SecondaryButton(
                label = "Lặp lại",
                iconText = "🔄",
                onClick = {
                    haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                    onAgainClick()
                }
            )

            RecordButton(
                isRecording = isRecording,
                voiceLevel = voiceLevel,
                onClick = {
                    haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                    onRecordingToggle()
                }
            )

            SecondaryButton(
                label = "Bản ghi",
                iconText = "📋",
                onClick = {
                    haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                    onTranscriptOpen()
                }
            )
        }
    }
}

/* ---------- Secondary (trái/phải): phẳng, sạch, không glow ---------- */
@Composable
private fun SecondaryButton(
    label: String,
    iconText: String,
    onClick: () -> Unit
) {
    val cc = MaterialTheme.colorScheme
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Box(
            Modifier
                .size(72.dp)
                .clip(CircleShape)
                .border(
                    width = 1.dp,
                    color = cc.onSurface.copy(alpha = 0.15f),
                    shape = CircleShape
                )
                .background(cc.surface.copy(alpha = 0.06f))
                .clickable { onClick() }
                .semantics { contentDescription = label },
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = iconText,
                style = MaterialTheme.typography.headlineMedium,
                color = cc.onSurface.copy(alpha = 0.92f)
            )
        }
        Spacer(Modifier.height(8.dp))
        Text(
            text = label,
            style = MaterialTheme.typography.labelLarge,
            color = cc.onSurface.copy(alpha = 0.9f),
            textAlign = TextAlign.Center
        )
    }
}

/* ---------- Primary (giữa): mic lớn, chỉ 1 hiệu ứng tinh tế ---------- */
@Composable
private fun RecordButton(
    isRecording: Boolean,
    voiceLevel: Int,
    onClick: () -> Unit
) {
    val cc = MaterialTheme.colorScheme
    val accent = when (voiceLevel) {
        0 -> com.auto_fe.auto_fe.ui.theme.DarkPrimary
        1 -> com.auto_fe.auto_fe.ui.theme.VoiceLowColor
        2 -> com.auto_fe.auto_fe.ui.theme.VoiceMediumColor
        else -> com.auto_fe.auto_fe.ui.theme.VoiceHighColor
    }

    val pulse = rememberInfiniteTransition(label = "mic-ring")
    // nhẫn (ring) mảnh 360° chạy nhẹ theo level – thay cho glow/blur
    val ringPhase by pulse.animateFloat(
        0f, 1f,
        animationSpec = infiniteRepeatable(
            tween(durationMillis = if (isRecording) 1200 - voiceLevel * 120 else 2000, easing = LinearEasing)
        ),
        label = "phase"
    )

    Box(
        modifier = Modifier
            .size(112.dp)
            .clip(CircleShape)
            .background(
                brush = Brush.radialGradient(
                    colors = listOf(
                        androidx.compose.ui.graphics.lerp(accent, Color.White, 0.08f),
                        accent
                    )
                )
            )
            .clickable { onClick() }
            .semantics { contentDescription = if (isRecording) "Dừng ghi âm" else "Bắt đầu ghi âm" },
        contentAlignment = Alignment.Center
    ) {
        // Progress ring khi đang ghi
        if (isRecording) {
            Canvas(Modifier.matchParentSize()) {
                val s = min(size.width, size.height)
                val stroke = s * 0.035f
                // vẽ vòng nền mờ
                drawArc(
                    color = cc.onPrimary.copy(alpha = 0.15f),
                    startAngle = -90f, sweepAngle = 360f,
                    useCenter = false,
                    topLeft = androidx.compose.ui.geometry.Offset(stroke, stroke),
                    size = androidx.compose.ui.geometry.Size(s - 2 * stroke, s - 2 * stroke),
                    style = androidx.compose.ui.graphics.drawscope.Stroke(stroke)
                )
                // sweep theo phase + level
                val sweep = 60f + 240f * (0.25f * voiceLevel + ringPhase % 1f)
                drawArc(
                    color = cc.onPrimary.copy(alpha = 0.9f),
                    startAngle = -90f, sweepAngle = sweep,
                    useCenter = false,
                    topLeft = androidx.compose.ui.geometry.Offset(stroke, stroke),
                    size = androidx.compose.ui.geometry.Size(s - 2 * stroke, s - 2 * stroke),
                    style = androidx.compose.ui.graphics.drawscope.Stroke(stroke)
                )
            }
        } else {
            // Dot pulse khi không ghi
            Canvas(Modifier.matchParentSize()) {
                val r = size.minDimension * 0.08f
                val cx = size.width * 0.72f
                val cy = size.height * 0.28f
                val pulse = 0.85f + 0.15f * kotlin.math.sin(2f * Math.PI.toFloat() * ringPhase)
                drawCircle(
                    color = cc.onPrimary.copy(alpha = 0.35f),
                    radius = r * pulse,
                    center = androidx.compose.ui.geometry.Offset(cx, cy)
                )
            }
        }

        Text(
            text = if (isRecording) "⏹️" else "🎙️",
            style = MaterialTheme.typography.headlineLarge,
            color = cc.onPrimary
        )
    }
}