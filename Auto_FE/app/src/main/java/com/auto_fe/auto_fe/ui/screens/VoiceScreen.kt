package com.auto_fe.auto_fe.ui.screens

import androidx.compose.animation.core.*
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.tween
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.scale
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.foundation.Canvas
import com.auto_fe.auto_fe.audio.VoiceManager
import com.auto_fe.auto_fe.ui.screens.layers.BackgroundLayer
import com.auto_fe.auto_fe.ui.screens.layers.InteractionLayer
import com.auto_fe.auto_fe.ui.screens.layers.OverlayLayer
import com.auto_fe.auto_fe.ui.theme.*
import com.auto_fe.auto_fe.core.CommandProcessor
import com.auto_fe.auto_fe.domain.VoiceState
import com.auto_fe.auto_fe.domain.VoiceEvent
import kotlin.math.roundToInt
import kotlinx.coroutines.delay
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import android.util.Log

/**
 * VoiceScreen - Layered Architecture Coordinator
 * 
 * Chức năng:
 * - Điều phối các layer: Background, Interaction, Overlay
 * - Quản lý state và logic âm thanh
 * - Tách biệt hoàn toàn UI rendering khỏi business logic
 * 
 * Cấu trúc layers:
 * - BackgroundLayer: Orb Sphere + dynamic glow
 * - InteractionLayer: Mic button + control ring + gesture states  
 * - OverlayLayer: Text, transcript, hints, toast
 */
@Composable
fun VoiceScreen() {
    // Dark/Light mode state
    var isDarkMode by remember { mutableStateOf(true) }
    
    // State management
    var isRecording by remember { mutableStateOf(false) }
    var voiceLevel by remember { mutableStateOf(0) }
    var transcriptText by remember { mutableStateOf("") }
    var errorMessage by remember { mutableStateOf("") }
    var isProcessing by remember { mutableStateOf(false) } // Flag để tránh vòng lặp
    
    // Smooth level system để tránh "giật cục"
    var rawLevel by remember { mutableStateOf(0f) }     // 0f..3f thô
    var smoothLevel by remember { mutableStateOf(0f) }  // 0f..3f đã lọc
    
    // ==== CLICK FEEDBACK EFFECTS ====
    var clickScale by remember { mutableStateOf(1f) }
    var clickAlpha by remember { mutableStateOf(1f) }
    var showClickRipple by remember { mutableStateOf(false) }
    var clickRippleScale by remember { mutableStateOf(0f) }
    
    val haptic = LocalHapticFeedback.current
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    
    // ==== CLICK ANIMATIONS ====
    val clickScaleAnimation by animateFloatAsState(
        targetValue = clickScale,
        animationSpec = tween(150, easing = FastOutSlowInEasing),
        label = "clickScale"
    )
    
    val clickAlphaAnimation by animateFloatAsState(
        targetValue = clickAlpha,
        animationSpec = tween(200, easing = FastOutSlowInEasing),
        label = "clickAlpha"
    )
    
    val rippleScaleAnimation by animateFloatAsState(
        targetValue = clickRippleScale,
        animationSpec = tween(600, easing = FastOutSlowInEasing),
        label = "rippleScale"
    )
    
    // ==== CLICK FEEDBACK FUNCTION ====
    fun triggerClickFeedback() {
        // Haptic feedback
        haptic.performHapticFeedback(HapticFeedbackType.LongPress)
        
        // Scale animation
        clickScale = 0.95f
        scope.launch {
            delay(100)
            clickScale = 1f
        }
        
        // Alpha animation
        clickAlpha = 0.7f
        scope.launch {
            delay(150)
            clickAlpha = 1f
        }
        
        // Ripple effect
        showClickRipple = true
        clickRippleScale = 0f
        scope.launch {
            delay(50)
            clickRippleScale = 1f
            delay(600)
            showClickRipple = false
        }
    }
    
    // Initialize VoiceManager và State Machine
    val voiceManager = remember { VoiceManager.getInstance(context) }
    val commandProcessor = remember { CommandProcessor(context) }
    
    // Setup StateMachine callbacks để lắng nghe state changes
    LaunchedEffect(commandProcessor) {
        // Lắng nghe state changes từ tất cả StateMachines
        commandProcessor.setupStateCallbacks(
            onSuccess = { message ->
                Log.d("VoiceScreen", "StateMachine success: $message")
                transcriptText = message
                isRecording = false
                isProcessing = false
                voiceLevel = 0
                rawLevel = 0f

                // Clear transcript sau delay
                CoroutineScope(Dispatchers.Main).launch {
                    delay(3000)
                    transcriptText = ""
                }
            },
            onError = { error ->
                Log.e("VoiceScreen", "StateMachine error: $error")
                errorMessage = error
                transcriptText = ""
                isRecording = false
                isProcessing = false
                voiceLevel = 0
                rawLevel = 0f

                // Clear error sau delay
                CoroutineScope(Dispatchers.Main).launch {
                    delay(5000)
                    errorMessage = ""
                }
            }
        )
    }
    
    // Smooth level animation với EMA + tween
    val level01 by animateFloatAsState(
        targetValue = (smoothLevel / 3f).coerceIn(0f, 1f),
        animationSpec = tween(180, easing = FastOutSlowInEasing),
        label = "level01"
    )
    
    // EMA filter cho smooth level
    LaunchedEffect(rawLevel, isRecording) {
        val target = if (isRecording) rawLevel else 0f
        // EMA — alpha nhỏ = mượt hơn (0.25 = mượt vừa phải)
        smoothLevel = 0.75f * smoothLevel + 0.25f * target
    }
    
    // Cleanup resources when component is disposed
    DisposableEffect(Unit) {
        onDispose {
            commandProcessor.release()
            voiceManager.release()
        }
    }

    // VoiceManager integration - sử dụng CommandProcessor
    LaunchedEffect(isRecording) {
        if (isRecording && !isProcessing) {
            isProcessing = true
            // Reset voice level khi bắt đầu recording
            voiceLevel = 0
            rawLevel = 0f
            
            // Chào hỏi và bắt đầu lắng nghe
            voiceManager.textToSpeech("Bạn cần tôi trợ giúp điều gì?", 0, object : VoiceManager.VoiceControllerCallback {
                override fun onSpeechResult(spokenText: String) {
                    if (spokenText.isNotEmpty()) {
                        Log.d("VoiceScreen", "Speech result: $spokenText")
                        transcriptText = spokenText

                        // Gửi lệnh đến CommandProcessor
                        // StateMachine callbacks sẽ handle UI updates
                        commandProcessor.processCommand(spokenText, object : CommandProcessor.CommandProcessorCallback {
                            override fun onCommandExecuted(success: Boolean, message: String) {
                                // StateMachine callbacks sẽ handle UI updates
                                Log.d("VoiceScreen", "Command executed: $success, $message")
                            }

                            override fun onError(error: String) {
                                // StateMachine callbacks sẽ handle UI updates
                                Log.e("VoiceScreen", "Command error: $error")
                            }

            override fun onNeedConfirmation(command: String, receiver: String, message: String) {
                // Không cần xử lý confirmation ở đây nữa
                // StateMachine sẽ tự xử lý confirmation
                Log.d("VoiceScreen", "Confirmation handled by StateMachine: $command -> $receiver: $message")
            }
                        })
                    } else {
                        Log.w("VoiceScreen", "Empty speech result")
                        errorMessage = "Không nhận được lệnh"
                        isRecording = false
                        isProcessing = false
                        voiceLevel = 0
                        rawLevel = 0f
                        
                        // Clear error sau delay
                        CoroutineScope(Dispatchers.Main).launch {
                            delay(3000)
                            errorMessage = ""
                        }
                    }
                }
                override fun onConfirmationResult(confirmed: Boolean) {
                    // Not used in this context
                }

                override fun onError(error: String) {
                    Log.e("VoiceScreen", "Speech recognition error: $error")
                    errorMessage = "Lỗi nhận dạng giọng nói: $error"
                    isRecording = false
                    isProcessing = false
                    voiceLevel = 0
                    rawLevel = 0f

                    // Clear error sau delay
                    CoroutineScope(Dispatchers.Main).launch {
                        delay(3000)
                        errorMessage = ""
                    }
                }
                override fun onAudioLevelChanged(level: Int) {
                    // Cập nhật voice level từ VoiceManager
                    if (isRecording) {
                        voiceLevel = level.coerceIn(0, 3)
                        rawLevel = level.coerceIn(0, 3).toFloat()
                        Log.d("VoiceScreen", "Voice level: $level")
                    }
                }
            })
        
        } else {
            // Reset voice level khi dừng recording
            voiceLevel = 0
            rawLevel = 0f
        }
    }

    // Layered Architecture - AI Visual Identity Background
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(
                brush = Brush.verticalGradient(
                    colors = listOf(
                        Color.White,
                        Color(0xFFFAFAFA),
                        Color(0xFFF5F5F5)
                    )
                )
            )
            .scale(clickScaleAnimation)
            .alpha(clickAlphaAnimation)
            .clickable { triggerClickFeedback() }
    ) {
        // Layer 1: Background - Orb Sphere + Dynamic Glow
        BackgroundLayer(
            voiceLevel = (level01 * 3f).roundToInt(),
            isListening = isRecording,
            isDarkMode = isDarkMode,
            performancePreset = "lite", // Ép lite để xác nhận hết crash
            onSphereClick = {
                if (!isRecording) {
                    // Bắt đầu ghi âm
                    isRecording = true
                } else {
                    // Dừng ghi âm - Cancel State Machine
                    isRecording = false
                    isProcessing = false
                    voiceLevel = 0
                    rawLevel = 0f
                    transcriptText = ""
                    errorMessage = ""
                }
            },
            modifier = Modifier.fillMaxSize()
        )
        
        // Layer 2: Interaction - Sphere Click + Control Buttons
        InteractionLayer(
            isRecording = isRecording,
            voiceLevel = voiceLevel,
            isDarkMode = isDarkMode,
            onRecordingToggle = {
                if (!isRecording) {
                    // Bắt đầu ghi âm
                    isRecording = true
                } else {
                    // Dừng ghi âm - Cancel State Machine
                    isRecording = false
                    isProcessing = false
                    voiceLevel = 0
                    rawLevel = 0f
                    transcriptText = ""
                    errorMessage = ""
                }
            },
            onModeToggle = {
                // Toggle dark/light mode
                isDarkMode = !isDarkMode
            },
            onTranscriptOpen = { /* Handle transcript view */ },
            modifier = Modifier.fillMaxSize()
        )
        
        // Layer 3: Overlay - Text, Transcript, Hints, Toast
        OverlayLayer(
            isRecording = isRecording,
            transcriptText = transcriptText,
            errorMessage = errorMessage,
            modifier = Modifier.fillMaxSize()
        )
        
        // Layer 4: Click Ripple Effect
        if (showClickRipple) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .alpha(0.3f)
            ) {
                Canvas(
                    modifier = Modifier
                        .fillMaxSize()
                        .scale(rippleScaleAnimation)
                ) {
                    val center = Offset(size.width / 2f, size.height / 2f)
                    val rippleRadius = size.minDimension * 0.4f * rippleScaleAnimation
                    
                    // Ripple circles
                    for (i in 0..3) {
                        val alpha = (1f - rippleScaleAnimation) * (1f - i * 0.2f)
                        val radius = rippleRadius * (1f + i * 0.3f)
                        
                        drawCircle(
                            color = if (isDarkMode) {
                                Color.Yellow.copy(alpha = alpha * 0.6f)
                            } else {
                                Color.Red.copy(alpha = alpha * 0.6f)
                            },
                            radius = radius,
                            center = center,
                            style = androidx.compose.ui.graphics.drawscope.Stroke(width = 4f)
                        )
                    }
                }
            }
        }
    }
}
