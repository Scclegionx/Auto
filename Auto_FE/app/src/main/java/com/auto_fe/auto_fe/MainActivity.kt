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
import androidx.activity.compose.BackHandler
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
import com.auto_fe.auto_fe.ui.screens.VoiceScreen
import com.auto_fe.auto_fe.ui.screens.AuthScreen
import com.auto_fe.auto_fe.ui.screens.SettingsScreen
import com.auto_fe.auto_fe.ui.screens.PrescriptionListScreen
import com.auto_fe.auto_fe.ui.screens.PrescriptionDetailScreen
import com.auto_fe.auto_fe.ui.screens.CreatePrescriptionScreen
import com.auto_fe.auto_fe.ui.components.CustomBottomNavigation
import com.auto_fe.auto_fe.utils.PermissionManager
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

    private val notificationPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            Log.d("MainActivity", "Notification permission granted")
            Toast.makeText(this, "✅ Đã cấp quyền thông báo", Toast.LENGTH_SHORT).show()
        } else {
            Log.w("MainActivity", "Notification permission denied")
            Toast.makeText(this, "⚠️ Cần cấp quyền thông báo để nhận nhắc nhở uống thuốc", Toast.LENGTH_LONG).show()
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

        // Request notification permission cho Android 13+
        requestNotificationPermission()

        setContent {
            Auto_FETheme(darkTheme = true) {
                MainScreen()
            }
        }

        checkPermissions()
    }

    private fun requestNotificationPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.POST_NOTIFICATIONS
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                Log.d("MainActivity", "Requesting POST_NOTIFICATIONS permission")
                notificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
            } else {
                Log.d("MainActivity", "POST_NOTIFICATIONS permission already granted")
            }
        }
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
    var selectedTab by remember { mutableStateOf(1) } // Default là tab ghi âm (index 1)
    
    // State để quản lý navigation
    var isLoggedIn by remember { mutableStateOf(false) }
    var accessToken by remember { mutableStateOf<String?>(null) }
    var selectedPrescriptionId by remember { mutableStateOf<Long?>(null) }
    var showCreatePrescription by remember { mutableStateOf(false) }

    // BackHandler để xử lý nút back
    BackHandler(enabled = selectedPrescriptionId != null || showCreatePrescription || (isLoggedIn && accessToken != null)) {
        when {
            // Nếu đang ở màn tạo đơn thuốc → quay về danh sách
            showCreatePrescription -> {
                showCreatePrescription = false
            }
            // Nếu đang ở chi tiết đơn thuốc → quay về danh sách
            selectedPrescriptionId != null -> {
                selectedPrescriptionId = null
            }
            // Nếu đang ở danh sách đơn thuốc → logout và quay về auth
            isLoggedIn && accessToken != null -> {
                isLoggedIn = false
                accessToken = null
                selectedTab = 0
            }
        }
    }

    when {
        // Màn hình tạo đơn thuốc mới
        showCreatePrescription && accessToken != null -> {
            CreatePrescriptionScreen(
                accessToken = accessToken!!,
                onBackClick = { showCreatePrescription = false },
                onSuccess = {
                    showCreatePrescription = false
                    // Optionally refresh prescription list
                }
            )
        }
        // Màn hình chi tiết đơn thuốc
        selectedPrescriptionId != null -> {
            PrescriptionDetailScreen(
                prescriptionId = selectedPrescriptionId!!,
                accessToken = accessToken ?: "",
                onBackClick = { selectedPrescriptionId = null }
            )
        }
        // Màn hình danh sách đơn thuốc (sau khi login)
        isLoggedIn && accessToken != null -> {
            PrescriptionListScreen(
                accessToken = accessToken!!,
                onPrescriptionClick = { prescriptionId ->
                    selectedPrescriptionId = prescriptionId
                },
                onCreateClick = {
                    showCreatePrescription = true
                },
                onLogout = {
                    isLoggedIn = false
                    accessToken = null
                    selectedTab = 0
                }
            )
        }
        // Màn hình chính với bottom navigation
        else -> {
            Scaffold(
                modifier = Modifier.fillMaxSize(),
                bottomBar = {
                    CustomBottomNavigation(
                        selectedTab = selectedTab,
                        onTabSelected = { selectedTab = it }
                    )
                }
            ) { innerPadding ->
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(innerPadding)
                ) {
                    // Tab Content
                    when (selectedTab) {
                        0 -> AuthScreen(
                            onLoginSuccess = { token ->
                                accessToken = token
                                isLoggedIn = true
                            }
                        )
                        1 -> VoiceScreen() // Màn hình ghi âm là mặc định
                        2 -> SettingsScreen()
                    }
                }
            }
        }
    }
}
    