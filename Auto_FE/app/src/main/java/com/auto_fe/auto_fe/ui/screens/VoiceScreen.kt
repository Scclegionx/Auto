package com.auto_fe.auto_fe.ui.screens

import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
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
    var isRecording by remember { mutableStateOf(false) }
    var confirmationQuestion by remember { mutableStateOf("") }
    var successMessage by remember { mutableStateOf("") }
    var errorMessage by remember { mutableStateOf("") }
    
    // Animation States (Voice Level)
    var rawLevel by remember { mutableStateOf(0f) }      // Level thô từ Mic
    var smoothLevel by remember { mutableStateOf(0f) }   // Level đã làm mượt cho Orb

    // Khởi tạo CommandProcessor với scope của Composable để quản lý vòng đời
    val commandProcessor = remember { CommandProcessor(context, scope) }

    // UI EFFECTS (Click, Opacity) 
    var clickOpacity by remember { mutableStateOf(1f) }

    // Animations setup
    val clickOpacityAnimation by animateFloatAsState(
        targetValue = clickOpacity,
        animationSpec = tween(100, easing = FastOutSlowInEasing), label = "opacity"
    )

    // Hàm trigger hiệu ứng khi nhấn
    fun triggerClickFeedback() {
        haptic.performHapticFeedback(HapticFeedbackType.LongPress)
        
        clickOpacity = 0.92f
        scope.launch { delay(80); clickOpacity = 1f }
    }

    // MAIN LOGIC HANDLER 
    fun handleMicAction() {
        triggerClickFeedback()

        if (!isRecording) {
            // --- BẮT ĐẦU ---
            isRecording = true
            confirmationQuestion = ""
            successMessage = ""
            errorMessage = ""
            
            // Gọi vào Core Logic
            commandProcessor.startVoiceControl(object : CommandProcessorCallback {
                override fun onCommandExecuted(success: Boolean, message: String) {
                    // Update UI khi thành công
                    isRecording = false
                    rawLevel = 0f
                    confirmationQuestion = ""
                    successMessage = message
                    errorMessage = ""
                    
                    // Tự động clear thông báo sau 3s
                    scope.launch {
                        delay(3000)
                        successMessage = ""
                    }
                }

                override fun onError(error: String) {
                    // Update UI khi lỗi
                    isRecording = false
                    errorMessage = error
                    confirmationQuestion = ""
                    successMessage = ""
                    rawLevel = 0f
                    
                    // Tự động clear lỗi sau 3s
                    scope.launch {
                        delay(3000)
                        errorMessage = ""
                    }
                }

                override fun onConfirmationRequired(question: String) {
                    // Update UI khi cần xác nhận
                    confirmationQuestion = question
                    // Vẫn giữ isRecording = true vì đang chờ phản hồi từ người dùng
                }

                override fun onVoiceLevelChanged(level: Int) {
                    rawLevel = level.toFloat()
                }
            })
        } else {
            // Nếu đang ghi âm mà bấm nút -> Hủy
            commandProcessor.cancel()
        
            isRecording = false
            rawLevel = 0f
        }
    }

    // SMOOTH LEVEL ANIMATION 
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
            .alpha(clickOpacityAnimation)
            .clickable { 
                // Xử lý click ở background cũng trigger mic
                handleMicAction() 
            }
    ) {
        // Layer 1: Background - Orb Sphere + Dynamic Glow
        BackgroundLayer(
            voiceLevel = smoothLevel.roundToInt(), // Dùng level đã làm mượt
            isListening = isRecording,
            performancePreset = "lite",
            onSphereClick = { handleMicAction() },
            modifier = Modifier.fillMaxSize()
        )
        
        // Layer 2: Interaction - Buttons
        InteractionLayer(
            isRecording = isRecording,
            voiceLevel = smoothLevel.roundToInt(),
            onRecordingToggle = { handleMicAction() },
            onTranscriptOpen = { /* TODO: Mở lịch sử chat */ },
            modifier = Modifier.fillMaxSize()
        )
        
        // Layer 3: Overlay - Text info
        OverlayLayer(
            isRecording = isRecording,
            confirmationQuestion = confirmationQuestion,
            successMessage = successMessage,
            errorMessage = errorMessage,
            modifier = Modifier.fillMaxSize()
        )
    }
}