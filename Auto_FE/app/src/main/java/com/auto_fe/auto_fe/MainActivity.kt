package com.auto_fe.auto_fe

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.height
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.animation.core.*
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.expandVertically
import androidx.compose.animation.shrinkVertically
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.DisposableEffect
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.tween
import androidx.compose.animation.core.EaseInOut
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.Spring.DampingRatioMediumBouncy
import androidx.compose.animation.core.Spring.StiffnessLow
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.scale
import kotlin.math.roundToInt
import kotlinx.coroutines.delay
import kotlin.random.Random
import kotlinx.coroutines.isActive
import com.auto_fe.auto_fe.ui.FloatingWindow
import com.auto_fe.auto_fe.ui.theme.Auto_FETheme
import com.auto_fe.auto_fe.utils.PermissionManager
import com.auto_fe.auto_fe.SoftControlButtons
import com.auto_fe.auto_fe.Rotating3DSphere
import com.auto_fe.auto_fe.automation.msg.WAAutomation
import com.auto_fe.auto_fe.automation.alarm.AlarmAutomation
import com.auto_fe.auto_fe.audio.VoiceManager
import com.auto_fe.auto_fe.automation.calendar.CalendarAutomation
import com.auto_fe.auto_fe.automation.third_apps.YouTubeAutomation
import com.auto_fe.auto_fe.automation.third_apps.ChromeAutomation
import android.util.Log

class MainActivity : ComponentActivity() {
    private lateinit var permissionManager: PermissionManager
    private lateinit var floatingWindow: FloatingWindow

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val allGranted = permissions.values.all { it }
        if (allGranted) {
            checkOverlayPermission()
        } else {
            Toast.makeText(this, "Cần cấp quyền để sử dụng ứng dụng", Toast.LENGTH_LONG).show()
        }
    }

    private val overlayPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (Settings.canDrawOverlays(this)) {
                startFloatingWindow()
            } else {
                Toast.makeText(this, "Cần cấp quyền hiển thị trên các ứng dụng khác", Toast.LENGTH_LONG).show()
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        permissionManager = PermissionManager(this)
        floatingWindow = FloatingWindow(this)

        setContent {
            Auto_FETheme {
                MainScreen()
            }
        }

        checkPermissions()
    }

    private fun checkPermissions() {
        if (!permissionManager.checkAllPermissions()) {
            requestPermissionLauncher.launch(permissionManager.getMissingPermissions().toTypedArray())
        } else {
            checkOverlayPermission()
        }
    }

    private fun checkOverlayPermission() {
        if (!permissionManager.checkOverlayPermission()) {
            val intent = Intent(
                Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                Uri.parse("package:$packageName")
            )
            overlayPermissionLauncher.launch(intent)
        } else {
            startFloatingWindow()
        }
    }

    private fun startFloatingWindow() {
        floatingWindow.showFloatingWindow()
        Toast.makeText(this, "Auto FE đã sẵn sàng!", Toast.LENGTH_SHORT).show()
    }

    override fun onDestroy() {
        super.onDestroy()
        floatingWindow.hideFloatingWindow()
        // Giải phóng resources để tránh memory leak
        floatingWindow.release()
    }
}

@Composable
fun MainScreen() {
    val context = LocalContext.current
    var selectedTab by remember { mutableStateOf(0) }

    Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
        ) {
            // Tab Header
            TabHeader(
                selectedTab = selectedTab,
                onTabSelected = { selectedTab = it }
            )

            // Tab Content
            when (selectedTab) {
                0 -> CommandsTab(context = context)
                1 -> VoiceAssistantScreen()
            }
        }
    }
}

@Composable
fun TabHeader(
    selectedTab: Int,
    onTabSelected: (Int) -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(16.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        TabButton(
            text = "🎤 Lệnh",
            icon = null,
            isSelected = selectedTab == 0,
            onClick = { onTabSelected(0) },
            modifier = Modifier.weight(1f)
        )

        TabButton(
            text = "🎵 Voice",
            icon = null,
            isSelected = selectedTab == 1,
            onClick = { onTabSelected(1) },
            modifier = Modifier.weight(1f)
        )
    }
}

@Composable
fun TabButton(
    text: String,
    icon: ImageVector?,
    isSelected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier
            .clickable { onClick() }
            .clip(RoundedCornerShape(12.dp)),
        colors = CardDefaults.cardColors(
            containerColor = if (isSelected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.surface
        ),
        elevation = CardDefaults.cardElevation(
            defaultElevation = if (isSelected) 8.dp else 2.dp
        )
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            if (icon != null) {
                Icon(
                    imageVector = icon,
                    contentDescription = text,
                    tint = if (isSelected) Color.White else Color(0xFFC57B57),
                    modifier = Modifier.size(24.dp)
                )

                Spacer(modifier = Modifier.height(4.dp))
            }

            Text(
                text = text,
                color = if (isSelected) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSurface,
                style = MaterialTheme.typography.labelMedium,
                fontWeight = FontWeight.Bold
            )
        }
    }
}

@Composable
fun CommandsTab(context: android.content.Context) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(
            text = "Auto FE",
            style = MaterialTheme.typography.headlineLarge,
            fontWeight = FontWeight.Bold
        )

        Spacer(modifier = Modifier.height(16.dp))

        Text(
            text = "Ứng dụng tự động hóa thao tác điện thoại",
            style = MaterialTheme.typography.bodyLarge,
            modifier = Modifier.padding(horizontal = 16.dp)
        )

        Spacer(modifier = Modifier.height(32.dp))

        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp),
            elevation = CardDefaults.cardElevation(defaultElevation = 1.dp)  // Giảm elevation
        ) {
            Column(
                modifier = Modifier.padding(16.dp)
            ) {
                Text(
                    text = "Hướng dẫn sử dụng:",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold
                )

                Spacer(modifier = Modifier.height(8.dp))

                Text(
                    text = "1. Cấp quyền cần thiết cho ứng dụng\n" +
                            "2. Cửa sổ nổi sẽ xuất hiện\n" +
                            "3. Nhấn vào cửa sổ nổi để mở menu\n" +
                            "4. Chọn 'Ghi âm lệnh' để bắt đầu",
                    style = MaterialTheme.typography.bodyMedium
                )
            }
        }

        Spacer(modifier = Modifier.height(24.dp))

        // Button test WhatsApp
        Button(
            onClick = {
                testWhatsAppFunction(context)
            },
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp)
        ) {
            Text("Test WhatsApp")
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Button test Alarm
        Button(
            onClick = {
                testAlarmFunction(context)
            },
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp)
        ) {
            Text("Test Alarm")
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Button test Calendar
        Button(
            onClick = {
                testCalendarFunction(context)
            },
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp)
        ) {
            Text("Test Calendar")
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Button test YouTube
        Button(
            onClick = {
                testYouTubeFunction(context)
            },
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp)
        ) {
            Text("Test YouTube")
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Button test Chrome
        Button(
            onClick = {
                testChromeFunction(context)
            },
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp)
        ) {
            Text("Test Chrome")
        }
    }
}


private fun testWhatsAppFunction(context: android.content.Context) {
    Log.d("MainActivity", "Starting WhatsApp test...")

    // Test trực tiếp với WAAutomation
    val waAutomation = WAAutomation(context)
    waAutomation.sendWA("mẹ", "con sắp về", object : WAAutomation.WACallback {
        override fun onSuccess() {
            Log.d("MainActivity", "WhatsApp test successful!")
            android.widget.Toast.makeText(context, "Test WhatsApp thành công!", android.widget.Toast.LENGTH_LONG).show()
        }
        override fun onError(error: String) {
            Log.e("MainActivity", "WhatsApp test error: $error")
            android.widget.Toast.makeText(context, "Test WhatsApp lỗi: $error", android.widget.Toast.LENGTH_LONG).show()
        }
    })
}

private fun testAlarmFunction(context: android.content.Context) {
    Log.d("MainActivity", "Starting Alarm test...")

    // Test trực tiếp với AlarmAutomation
    val alarmAutomation = AlarmAutomation(context)

    // Tạo báo thức mặc định (9h sáng thứ 2 hàng tuần)
    alarmAutomation.createDefaultAlarm(object : AlarmAutomation.AlarmCallback {
        override fun onSuccess() {
            Log.d("MainActivity", "Alarm test successful!")
            android.widget.Toast.makeText(context, "Đã tạo báo thức thành công!", android.widget.Toast.LENGTH_LONG).show()
        }
        override fun onError(error: String) {
            Log.e("MainActivity", "Alarm test error: $error")
            android.widget.Toast.makeText(context, "Test Alarm lỗi: $error", android.widget.Toast.LENGTH_LONG).show()
        }
    })
}

private fun testCalendarFunction(context: android.content.Context) {
    Log.d("MainActivity", "Starting Calendar test...")

    // Kiểm tra quyền calendar
    if (ContextCompat.checkSelfPermission(context, Manifest.permission.WRITE_CALENDAR)
        != PackageManager.PERMISSION_GRANTED) {

        Log.d("MainActivity", "Calendar permission not granted")
        android.widget.Toast.makeText(context, "Cần quyền truy cập lịch để tạo sự kiện", android.widget.Toast.LENGTH_LONG).show()
        return
    }

    // Test trực tiếp với CalendarAutomation
    val calendarAutomation = CalendarAutomation(context)

    // Tạo sự kiện mặc định (Họp thứ 4 tới lúc 10h sáng)
    calendarAutomation.createDefaultEvent(object : CalendarAutomation.CalendarCallback {
        override fun onSuccess() {
            Log.d("MainActivity", "Calendar test successful!")
            android.widget.Toast.makeText(context, "Đã tạo sự kiện thành công!", android.widget.Toast.LENGTH_LONG).show()
        }
        override fun onError(error: String) {
            Log.e("MainActivity", "Calendar test error: $error")
            android.widget.Toast.makeText(context, "Test Calendar lỗi: $error", android.widget.Toast.LENGTH_LONG).show()
        }
    })
}

private fun testYouTubeFunction(context: android.content.Context) {
    Log.d("MainActivity", "Starting YouTube test...")

    // Test trực tiếp với YouTubeAutomation
    val youtubeAutomation = YouTubeAutomation(context)

    // Tìm kiếm mặc định: "nhạc sơn tùng MTP"
    youtubeAutomation.searchDefault(object : YouTubeAutomation.YouTubeCallback {
        override fun onSuccess() {
            Log.d("MainActivity", "YouTube test successful!")
            android.widget.Toast.makeText(context, "Đã mở YouTube tìm kiếm!", android.widget.Toast.LENGTH_LONG).show()
        }
        override fun onError(error: String) {
            Log.e("MainActivity", "YouTube test error: $error")
            android.widget.Toast.makeText(context, "Test YouTube lỗi: $error", android.widget.Toast.LENGTH_LONG).show()
        }
    })
}

private fun testChromeFunction(context: android.content.Context) {
    Log.d("MainActivity", "Starting Chrome test...")

    // Test trực tiếp với ChromeAutomation
    val chromeAutomation = ChromeAutomation(context)

    // Tìm kiếm mặc định: "nhạc sơn tùng MTP"
    chromeAutomation.searchDefault(object : ChromeAutomation.ChromeCallback {
        override fun onSuccess() {
            Log.d("MainActivity", "Chrome test successful!")
            android.widget.Toast.makeText(context, "Đã mở Chrome tìm kiếm!", android.widget.Toast.LENGTH_LONG).show()
        }
        override fun onError(error: String) {
            Log.e("MainActivity", "Chrome test error: $error")
            android.widget.Toast.makeText(context, "Test Chrome lỗi: $error", android.widget.Toast.LENGTH_LONG).show()
        }
    })
}


@Composable
fun StatusChip(
    text: String,
    level: Int,
    modifier: Modifier = Modifier
) {
    val chipColor = when (level) {
        0 -> com.auto_fe.auto_fe.ui.theme.DarkSurface
        1 -> com.auto_fe.auto_fe.ui.theme.VoiceLowColor    // Xanh lá nhẹ
        2 -> com.auto_fe.auto_fe.ui.theme.VoiceMediumColor  // Xanh dương
        else -> com.auto_fe.auto_fe.ui.theme.VoiceHighColor // Xanh dương đậm
    }

    val textColor = when (level) {
        0 -> com.auto_fe.auto_fe.ui.theme.DarkOnSurface
        1, 2, 3 -> Color.White
        else -> com.auto_fe.auto_fe.ui.theme.DarkOnSurface
    }

    Card(
        modifier = modifier
            .padding(horizontal = 32.dp)
            .height(48.dp),
        colors = CardDefaults.cardColors(
            containerColor = chipColor
        ),
        shape = RoundedCornerShape(24.dp),
        elevation = CardDefaults.cardElevation(
            defaultElevation = 4.dp
        )
    ) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 20.dp),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = text,
                style = MaterialTheme.typography.labelLarge,
                color = textColor,
                textAlign = androidx.compose.ui.text.style.TextAlign.Center,
                fontWeight = FontWeight.Medium
            )
        }
    }
}

// Main Voice Assistant Screen that integrates all components
@Composable
fun VoiceAssistantScreen() {
    // State management
    var isRecording by remember { mutableStateOf(false) }
    var voiceLevel by remember { mutableStateOf(0) }
    var transcriptText by remember { mutableStateOf("") }
    var errorMessage by remember { mutableStateOf("") }
    
    // Smooth level system để tránh "giật cục"
    var rawLevel by remember { mutableStateOf(0f) }     // 0f..3f thô
    var smoothLevel by remember { mutableStateOf(0f) }  // 0f..3f đã lọc
    
    // Removed isNightMode - always use dark theme
    val haptic = LocalHapticFeedback.current
    val context = LocalContext.current
    
    // Initialize VoiceManager
    val voiceManager = remember { VoiceManager.getInstance(context) }
    
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
    
    // Helper function để update voice level từ external source (SpeechRecognizer)
    fun updateVoiceLevel(level: Int) {
        voiceLevel = level.coerceIn(0, 3)
    }
    
    // Cleanup VoiceManager when component is disposed
    DisposableEffect(Unit) {
        onDispose {
            voiceManager.release()
        }
    }

    // VoiceManager integration - tương thích với audio system hiện có
    LaunchedEffect(isRecording) {
        if (isRecording) {
            // Reset voice level khi bắt đầu recording
            voiceLevel = 0
            
            // Sử dụng startVoiceInteractionSilent để không phát TTS
            voiceManager.startVoiceInteractionSilent(
                object : VoiceManager.VoiceControllerCallback {
                    override fun onSpeechResult(spokenText: String) {
                        transcriptText = spokenText
                        isRecording = false
                        voiceLevel = 0
                        Log.d("VoiceAssistant", "Speech result: $spokenText")
                    }
                    
                    override fun onConfirmationResult(confirmed: Boolean) {
                        // Handle confirmation if needed
                    }
                    
                    override fun onError(error: String) {
                        errorMessage = error
                        isRecording = false
                        voiceLevel = 0
                        transcriptText = ""
                        Log.e("VoiceAssistant", "Speech error: $error")
                    }
                    
                    override fun onAudioLevelChanged(level: Int) {
                        // Cập nhật rawLevel cho smooth system
                        if (isRecording) {
                            rawLevel = level.coerceIn(0, 3).toFloat()
                            Log.d("VoiceAssistant", "Raw audio level: $level")
                        }
                    }
                }
            )
        } else {
            // Reset voice level khi dừng recording
            voiceLevel = 0
            rawLevel = 0f
            // Clear transcript và error sau delay
            delay(5000)
            transcriptText = ""
            errorMessage = ""
        }
    }

    // Use our custom theme - always dark
    Auto_FETheme(
        darkTheme = true
    ) {
        // Set up screen background with enhanced gradient
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(
                    brush = androidx.compose.ui.graphics.Brush.verticalGradient(
                        colors = listOf(
                            com.auto_fe.auto_fe.ui.theme.DarkGradientStart,
                            com.auto_fe.auto_fe.ui.theme.DarkGradientEnd
                        )
                    )
                )
        ) {
            // Enhanced vignetting for depth
            androidx.compose.foundation.Canvas(
                modifier = Modifier.fillMaxSize()
            ) {
                drawCircle(
                    brush = androidx.compose.ui.graphics.Brush.radialGradient(
                        colors = listOf(
                            Color.Transparent,
                            Color.Black.copy(alpha = 0.12f)
                        ),
                        center = androidx.compose.ui.geometry.Offset(size.width / 2f, size.height / 2f),
                        radius = size.minDimension * 1.5f
                    ),
                    radius = size.minDimension * 1.5f,
                    center = androidx.compose.ui.geometry.Offset(size.width / 2f, size.height / 2f)
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
                            containerColor = com.auto_fe.auto_fe.ui.theme.DarkSurface.copy(alpha = 0.7f)
                        ),
                        shape = RoundedCornerShape(24.dp),
                        elevation = CardDefaults.cardElevation(
                            defaultElevation = 2.dp
                        )
                    ) {
                        Text(
                            text = if (isRecording) "🎤 Đang lắng nghe" else "🎤 Trợ lý giọng nói",
                            style = MaterialTheme.typography.headlineMedium,
                            color = com.auto_fe.auto_fe.ui.theme.DarkOnSurface,
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
                            containerColor = com.auto_fe.auto_fe.ui.theme.DarkSurface.copy(alpha = 0.8f)
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
                            color = com.auto_fe.auto_fe.ui.theme.DarkOnSurface,
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
                            containerColor = com.auto_fe.auto_fe.ui.theme.DarkSurface.copy(alpha = 0.6f)
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
                            color = com.auto_fe.auto_fe.ui.theme.DarkOnSurface.copy(alpha = 0.9f),
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
                        isRecording = !isRecording
                        haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                        if (!isRecording) voiceLevel = 0
                    },
                    onAgainClick = { 
                        // Replay last recording
                        isRecording = true
                        haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                    },
                    onTranscriptOpen = { /* Handle transcript view */ }
                )
            }
        }
    }
}
    