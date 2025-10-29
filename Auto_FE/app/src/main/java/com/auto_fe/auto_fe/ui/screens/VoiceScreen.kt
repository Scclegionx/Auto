package com.auto_fe.auto_fe.ui.screens

import androidx.compose.animation.core.*
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.tween
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.platform.LocalContext
import androidx.compose.foundation.shape.RoundedCornerShape
import com.auto_fe.auto_fe.audio.VoiceManager
import com.auto_fe.auto_fe.SoftControlButtons
import com.auto_fe.auto_fe.Rotating3DSphere
import com.auto_fe.auto_fe.ui.theme.*
import com.auto_fe.auto_fe.core.CommandProcessor
import com.auto_fe.auto_fe.automation.device.CameraAutomation
import com.auto_fe.auto_fe.domain.VoiceState
import com.auto_fe.auto_fe.domain.VoiceEvent
import kotlin.math.roundToInt
import kotlinx.coroutines.delay
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import android.util.Log

/**
 * Màn hình ghi âm chính - màn hình mặc định khi mở app
 */
@Composable
fun VoiceScreen() {
    // State management
    var isRecording by remember { mutableStateOf(false) }
    var voiceLevel by remember { mutableStateOf(0) }
    var transcriptText by remember { mutableStateOf("") }
    var errorMessage by remember { mutableStateOf("") }
    var isProcessing by remember { mutableStateOf(false) } // Flag để tránh vòng lặp
    
    // Confirmation state
    
    // Smooth level system để tránh "giật cục"
    var rawLevel by remember { mutableStateOf(0f) }     // 0f..3f thô
    var smoothLevel by remember { mutableStateOf(0f) }  // 0f..3f đã lọc
    
    val haptic = LocalHapticFeedback.current
    val context = LocalContext.current
    
    // Helper function để kiểm tra phản hồi xác nhận
    
    // Initialize VoiceManager và CommandProcessor - chỉ tạo 1 lần
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
    
    // VoiceManager không có property onAudioLevelChanged, sẽ handle trong callback
    
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

    // Set up screen background with enhanced gradient
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(
                brush = Brush.verticalGradient(
                    colors = listOf(
                        DarkGradientStart,
                        DarkGradientEnd
                    )
                )
            )
    ) {
        // Enhanced vignetting for depth
        Canvas(
            modifier = Modifier.fillMaxSize()
        ) {
            drawCircle(
                brush = Brush.radialGradient(
                    colors = listOf(
                        Color.Transparent,
                        Color.Black.copy(alpha = 0.12f)
                    ),
                    center = Offset(size.width / 2f, size.height / 2f),
                    radius = size.minDimension * 1.5f
                ),
                radius = size.minDimension * 1.5f,
                center = Offset(size.width / 2f, size.height / 2f)
            )
        }

        // Full screen sphere as background với smooth level
        Rotating3DSphere(
            voiceLevel = (level01 * 3f).roundToInt(),
            isNightMode = true,
            modifier = Modifier.fillMaxSize()
        )

        // Overlay UI elements on top of sphere
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier.fillMaxSize()
        ) {
            // Title with subtle animation
            val titleScale by animateFloatAsState(
                targetValue = if (isRecording) 0.9f else 1.0f,
                animationSpec = spring(
                    dampingRatio = Spring.DampingRatioMediumBouncy,
                    stiffness = Spring.StiffnessLow
                ),
                label = "title"
            )

            Box(
                contentAlignment = Alignment.Center,
                modifier = Modifier
                    .padding(top = 40.dp, bottom = 8.dp)
                    .scale(titleScale)
            ) {
                Card(
                    colors = CardDefaults.cardColors(
                        containerColor = DarkSurface.copy(alpha = 0.7f)
                    ),
                    shape = RoundedCornerShape(24.dp),
                    elevation = CardDefaults.cardElevation(
                        defaultElevation = 2.dp
                    )
                ) {
                    Text(
                        text = if (isRecording) "🎤 Đang lắng nghe" else "🎤 Trợ lý giọng nói",
                        style = MaterialTheme.typography.headlineMedium,
                        color = DarkOnSurface,
                        textAlign = TextAlign.Center,
                        modifier = Modifier.padding(horizontal = 24.dp, vertical = 12.dp)
                    )
                }
            }

            // Status hint with animation
            val hintAlpha by animateFloatAsState(
                targetValue = if (isRecording) 0.0f else 0.9f,
                animationSpec = tween(
                    durationMillis = 500,
                    easing = FastOutSlowInEasing
                ),
                label = "hint"
            )

            // Transcript display area
                if (transcriptText.isNotEmpty()) {
                Card(
                    colors = CardDefaults.cardColors(
                        containerColor = DarkSurface.copy(alpha = 0.8f)
                    ),
                    shape = RoundedCornerShape(16.dp),
                    elevation = CardDefaults.cardElevation(
                        defaultElevation = 2.dp
                    ),
                    modifier = Modifier
                        .padding(horizontal = 24.dp, vertical = 8.dp)
                        .alpha(0.95f)
                ) {
                    Text(
                        text = transcriptText,
                        style = MaterialTheme.typography.bodyLarge,
                        color = DarkOnSurface,
                        textAlign = TextAlign.Center,
                        modifier = Modifier.padding(horizontal = 20.dp, vertical = 16.dp)
                    )
                }
            } else if (errorMessage.isNotEmpty()) {
                Card(
                    colors = CardDefaults.cardColors(
                        containerColor = Color.Red.copy(alpha = 0.1f)
                    ),
                    shape = RoundedCornerShape(16.dp),
                    elevation = CardDefaults.cardElevation(
                        defaultElevation = 2.dp
                    ),
                    modifier = Modifier
                        .padding(horizontal = 24.dp, vertical = 8.dp)
                        .alpha(0.95f)
                ) {
                    Text(
                        text = "❌ $errorMessage",
                        style = MaterialTheme.typography.bodyLarge,
                        color = Color.Red.copy(alpha = 0.9f),
                        textAlign = TextAlign.Center,
                        modifier = Modifier.padding(horizontal = 20.dp, vertical = 16.dp)
                    )
                }
            } else if (!isRecording) {
                Card(
                    colors = CardDefaults.cardColors(
                        containerColor = DarkSurface.copy(alpha = 0.6f)
                    ),
                    shape = RoundedCornerShape(16.dp),
                    elevation = CardDefaults.cardElevation(
                        defaultElevation = 1.dp
                    ),
                    modifier = Modifier
                        .padding(horizontal = 32.dp, vertical = 8.dp)
                        .alpha(hintAlpha)
                ) {
                    Text(
                        text = "Nhấn nút ở dưới để trò chuyện",
                        style = MaterialTheme.typography.bodyLarge,
                        color = DarkOnSurface.copy(alpha = 0.9f),
                        textAlign = TextAlign.Center,
                        modifier = Modifier.padding(horizontal = 24.dp, vertical = 12.dp)
                    )
                }
            }

            // Spacer to push buttons to bottom
            Spacer(modifier = Modifier.weight(1f))

            // Extra spacing for better visual separation
            Spacer(modifier = Modifier.height(8.dp))

            // Control buttons overlay
            SoftControlButtons(
                isRecording = isRecording,
                voiceLevel = voiceLevel,
                onRecordingToggle = {
                    if (!isRecording && !isProcessing) {
                        // Bắt đầu ghi âm
                        isRecording = true
                        haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                    } else {
                        // Dừng ghi âm
                        isRecording = false
                        isProcessing = false
                        voiceLevel = 0
                        rawLevel = 0f
                        transcriptText = ""
                        errorMessage = ""
                        haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                    }
                },
                onAgainClick = { 
                    // Replay last recording
                    if (!isProcessing) {
                        isRecording = true
                        haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                    }
                },
                onTranscriptOpen = { /* Handle transcript view */ }
            )
        }
        
    }
}
