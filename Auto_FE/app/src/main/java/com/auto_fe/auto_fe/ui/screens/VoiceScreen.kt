package com.auto_fe.auto_fe.ui.screens

import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.scale
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalHapticFeedback
import com.auto_fe.auto_fe.base.callback.CommandProcessorCallback
import com.auto_fe.auto_fe.core.CommandProcessor
import com.auto_fe.auto_fe.ui.screens.layers.BackgroundLayer
import com.auto_fe.auto_fe.ui.screens.layers.InteractionLayer
import com.auto_fe.auto_fe.ui.screens.layers.OverlayLayer
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlin.math.roundToInt

@Composable
fun VoiceScreen() {
    // SETUP RESOURCES & STATES
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val haptic = LocalHapticFeedback.current
    
    // UI States
    var isDarkMode by remember { mutableStateOf(true) }
    var isRecording by remember { mutableStateOf(false) }
    var transcriptText by remember { mutableStateOf("") }
    var errorMessage by remember { mutableStateOf("") }
    
    // Animation States (Voice Level)
    var rawLevel by remember { mutableStateOf(0f) }      // Level thô từ Mic
    var smoothLevel by remember { mutableStateOf(0f) }   // Level đã làm mượt cho Orb

    // Khởi tạo CommandProcessor với scope của Composable để quản lý vòng đời
    val commandProcessor = remember { CommandProcessor(context, scope) }

    // UI EFFECTS (Click, Ripple, Scale) 
    var clickScale by remember { mutableStateOf(1f) }
    var clickAlpha by remember { mutableStateOf(1f) }
    var showClickRipple by remember { mutableStateOf(false) }
    var clickRippleScale by remember { mutableStateOf(0f) }

    // Animations setup
    val clickScaleAnimation by animateFloatAsState(
        targetValue = clickScale,
        animationSpec = tween(150, easing = FastOutSlowInEasing), label = "scale"
    )
    val clickAlphaAnimation by animateFloatAsState(
        targetValue = clickAlpha,
        animationSpec = tween(200, easing = FastOutSlowInEasing), label = "alpha"
    )
    val rippleScaleAnimation by animateFloatAsState(
        targetValue = clickRippleScale,
        animationSpec = tween(600, easing = FastOutSlowInEasing), label = "ripple"
    )

    // Hàm trigger hiệu ứng khi nhấn
    fun triggerClickFeedback() {
        haptic.performHapticFeedback(HapticFeedbackType.LongPress)
        
        // Scale effect
        clickScale = 0.95f
        scope.launch { delay(100); clickScale = 1f }
        
        // Alpha effect
        clickAlpha = 0.7f
        scope.launch { delay(150); clickAlpha = 1f }
        
        // Ripple effect
        showClickRipple = true
        clickRippleScale = 0f
        scope.launch {
            delay(50); clickRippleScale = 1f
            delay(600); showClickRipple = false
        }
    }

    // MAIN LOGIC HANDLER 
    fun handleMicAction() {
        triggerClickFeedback() // Chạy hiệu ứng hình ảnh/rung

        if (!isRecording) {
            // --- BẮT ĐẦU ---
            isRecording = true
            errorMessage = ""
            transcriptText = "Đang lắng nghe..."
            
            // Gọi vào Core Logic
            commandProcessor.startVoiceControl(object : CommandProcessorCallback {
                override fun onCommandExecuted(success: Boolean, message: String) {
                    // Update UI khi thành công
                    isRecording = false
                    transcriptText = message
                    rawLevel = 0f
                    
                    // Tự động clear text sau 3s cho gọn
                    scope.launch {
                        delay(3000)
                        transcriptText = ""
                    }
                }

                override fun onError(error: String) {
                    // Update UI khi lỗi
                    isRecording = false
                    errorMessage = error
                    rawLevel = 0f
                    
                    // Tự động clear lỗi sau 3s
                    scope.launch {
                        delay(3000)
                        errorMessage = ""
                    }
                }

                // Để Orb Sphere chuyển động theo giọng nói
                override fun onVoiceLevelChanged(level: Int) {
                    rawLevel = level.toFloat()
                }
            })
        } else {
            // Nếu đang ghi âm mà bấm nút -> Hủy
            commandProcessor.cancel()
        
            isRecording = false
            rawLevel = 0f
            transcriptText = "Đã hủy"
            
            // Clear text sau 1s
            scope.launch {
                delay(1000)
                transcriptText = ""
            }
        }
    }

    // SMOOTH LEVEL ANIMATION 
    // EMA Filter: Giúp Orb chuyển động mượt mà, không bị giật cục theo mic
    LaunchedEffect(rawLevel, isRecording) {
        val target = if (isRecording) rawLevel else 0f
        smoothLevel = 0.6f * smoothLevel + 0.4f * target
    }

    // RENDER UI
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(
                brush = Brush.verticalGradient(
                    colors = listOf(Color.White, Color(0xFFFAFAFA), Color(0xFFF5F5F5))
                )
            )
            .scale(clickScaleAnimation)
            .alpha(clickAlphaAnimation)
            .clickable { 
                // Xử lý click ở background cũng trigger mic
                handleMicAction() 
            }
    ) {
        // Layer 1: Background - Orb Sphere + Dynamic Glow
        BackgroundLayer(
            voiceLevel = smoothLevel.roundToInt(), // Dùng level đã làm mượt
            isListening = isRecording,
            isDarkMode = isDarkMode,
            performancePreset = "lite",
            onSphereClick = { handleMicAction() },
            modifier = Modifier.fillMaxSize()
        )
        
        // Layer 2: Interaction - Buttons
        InteractionLayer(
            isRecording = isRecording,
            voiceLevel = smoothLevel.roundToInt(),
            isDarkMode = isDarkMode,
            onRecordingToggle = { handleMicAction() },
            onModeToggle = { isDarkMode = !isDarkMode },
            onTranscriptOpen = { /* TODO: Mở lịch sử chat */ },
            modifier = Modifier.fillMaxSize()
        )
        
        // Layer 3: Overlay - Text info
        OverlayLayer(
            isRecording = isRecording,
            transcriptText = transcriptText,
            errorMessage = errorMessage,
            modifier = Modifier.fillMaxSize()
        )
        
        // Layer 4: Ripple Effect (Visual candy)
        if (showClickRipple) {
            RippleEffect(rippleScaleAnimation, isDarkMode)
        }
    }
}

// Ripple Composable
@Composable
private fun RippleEffect(scale: Float, isDarkMode: Boolean) {
    Box(modifier = Modifier.fillMaxSize().alpha(0.3f)) {
        Canvas(modifier = Modifier.fillMaxSize().scale(scale)) {
            val center = Offset(size.width / 2f, size.height / 2f)
            val rippleRadius = size.minDimension * 0.4f * scale
            
            for (i in 0..3) {
                val alpha = (1f - scale) * (1f - i * 0.2f)
                val radius = rippleRadius * (1f + i * 0.3f)
                drawCircle(
                    color = if (isDarkMode) Color.Yellow.copy(alpha = alpha * 0.6f) 
                            else Color.Red.copy(alpha = alpha * 0.6f),
                    radius = radius,
                    center = center,
                    style = Stroke(width = 4f)
                )
            }
        }
    }
}