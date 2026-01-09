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
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.core.content.ContextCompat
import androidx.compose.foundation.layout.Box
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import kotlinx.coroutines.launch
import com.auto_fe.auto_fe.ui.components.popup.FloatingWindow
import com.auto_fe.auto_fe.ui.theme.Auto_FETheme
import com.auto_fe.auto_fe.ui.screens.VoiceScreen
import com.auto_fe.auto_fe.ui.screens.AuthScreen
import com.auto_fe.auto_fe.ui.screens.SettingsScreen
import com.auto_fe.auto_fe.ui.screens.GuideScreen
import com.auto_fe.auto_fe.ui.screens.MedicationTabScreen
import com.auto_fe.auto_fe.ui.screens.ManageConnectionsScreen
import com.auto_fe.auto_fe.ui.screens.ElderListScreen
import com.auto_fe.auto_fe.ui.screens.CreateStandaloneMedicationScreen
import com.auto_fe.auto_fe.ui.screens.SearchUserForConnectionScreen
import com.auto_fe.auto_fe.ui.screens.PrescriptionDetailScreen
import com.auto_fe.auto_fe.ui.screens.CreatePrescriptionScreen
import com.auto_fe.auto_fe.ui.screens.VerificationScreen
import com.auto_fe.auto_fe.ui.screens.ProfileScreen
import com.auto_fe.auto_fe.ui.screens.ForgotPasswordScreen
import com.auto_fe.auto_fe.ui.screens.ChangePasswordScreen
import com.auto_fe.auto_fe.ui.screens.NotificationScreen
import com.auto_fe.auto_fe.ui.screens.EmergencyContactScreen
import com.auto_fe.auto_fe.ui.screens.ChatListScreen
import com.auto_fe.auto_fe.ui.screens.SearchUserScreen
import com.auto_fe.auto_fe.ui.screens.MedicalDocumentsScreen
import com.auto_fe.auto_fe.utils.be.SessionManager
import com.auto_fe.auto_fe.utils.common.PermissionManager
import com.auto_fe.auto_fe.network.ApiClient
import android.util.Log
import com.auto_fe.auto_fe.audio.TTSManager
import com.auto_fe.auto_fe.ui.components.CustomBottomNavigation
import com.auto_fe.auto_fe.automation.msg.SMSReceiver

class MainActivity : ComponentActivity() {
    private lateinit var permissionManager: PermissionManager
    private lateinit var floatingWindow: FloatingWindow
    private lateinit var sessionManager: SessionManager
    private var smsReceiver: SMSReceiver? = null

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
            Toast.makeText(this, "Đã cấp quyền thông báo", Toast.LENGTH_SHORT).show()
        } else {
            Log.w("MainActivity", "Notification permission denied")
            Toast.makeText(this, "Cần cấp quyền thông báo để nhận nhắc nhở uống thuốc", Toast.LENGTH_LONG).show()
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
        permissionManager.requestOverlayPermission(this)
        floatingWindow = FloatingWindow(this)
        sessionManager = SessionManager(this)

        // Request notification permission cho Android 13+
        requestNotificationPermission()

        setContent {
            Auto_FETheme(darkTheme = true) {
                MainScreen(sessionManager = sessionManager)
            }
        }

        checkPermissions()
        
        // Khởi tạo và bắt đầu SMSReceiver (ContentObserver)
        initializeSMSReceiver()
    }
    
    private fun initializeSMSReceiver() {
        try {
            smsReceiver = SMSReceiver(this)
            smsReceiver?.startObserving()
            Log.d("MainActivity", "SMSReceiver initialized and started observing")
        } catch (e: Exception) {
            Log.e("MainActivity", "Error initializing SMSReceiver: ${e.message}", e)
        }
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
        // Dừng SMSReceiver khi activity bị destroy
        smsReceiver?.stopObserving()
        smsReceiver = null
        Log.d("MainActivity", "SMSReceiver stopped")
        TTSManager.getInstance(this).release()
        floatingWindow.hideFloatingWindow()
        // Giải phóng resources để tránh memory leak
        floatingWindow.release()
    }
}

@Composable
fun MainScreen(sessionManager: SessionManager) {
    val context = LocalContext.current
    var selectedTab by remember { mutableStateOf(1) } // Default là tab ghi âm (index 1)
    
    // State để quản lý navigation
    var isLoggedIn by remember { mutableStateOf(sessionManager.isLoggedIn()) }
    var accessToken by remember { mutableStateOf(sessionManager.getAccessToken()) }
    var selectedPrescriptionId by remember { mutableStateOf<Long?>(null) }
    var showCreatePrescription by remember { mutableStateOf(false) }
    var editPrescriptionId by remember { mutableStateOf<Long?>(null) }  //  Thêm state cho edit
    var showVerification by remember { mutableStateOf(false) }
    var showProfile by remember { mutableStateOf(false) }
    var showForgotPassword by remember { mutableStateOf(false) }  //  Thêm state cho forgot password
    var showChangePassword by remember { mutableStateOf(false) }  //  Thêm state cho change password
    var showNotificationHistory by remember { mutableStateOf(false) }  //  Thêm state cho notification history
    var showEmergencyContact by remember { mutableStateOf(false) }  //  Thêm state cho emergency contact
    var showManageConnections by remember { mutableStateOf(false) }  //  Thêm state cho quản lý kết nối
    var showCreateStandaloneMedication by remember { mutableStateOf(false) }  // Thêm state cho create standalone medication
    var showChatList by remember { mutableStateOf(false) }  //  Thêm state cho chat list
    var showSearchUser by remember { mutableStateOf(false) }  // Thêm state cho search user (cho chat)
    var showSearchConnection by remember { mutableStateOf(false) }  //  Thêm state cho search connection
    var showChatDetail by remember { mutableStateOf(false) }  // Thêm state cho chat detail
    var showMedicalDocuments by remember { mutableStateOf(false) }  // Thêm state cho medical documents
    var selectedChatId by remember { mutableStateOf<Long?>(null) }  //  Chat ID đã tồn tại
    var selectedReceiverId by remember { mutableStateOf<Long?>(null) }  //  Receiver ID cho chat mới
    var selectedChatName by remember { mutableStateOf<String?>(null) }  //  Tên người nhận
    var selectedUserId by remember { mutableStateOf<Long?>(null) }  //  State để trigger create chat
    var selectedUserName by remember { mutableStateOf<String?>(null) }  //  State để lưu tên user
    var selectedElderUserId by remember { mutableStateOf<Long?>(null) }  //  State cho Supervisor xem Elder
    var selectedElderUserName by remember { mutableStateOf<String?>(null) }  // Tên Elder
    var supervisorCanView by remember { mutableStateOf(true) }  // Quyền xem của Supervisor
    var supervisorCanUpdate by remember { mutableStateOf(true) }  // Quyền sửa của Supervisor
    var verificationEmail by remember { mutableStateOf("") }
    var verificationPassword by remember { mutableStateOf("") }
    var verifiedEmail by remember { mutableStateOf<String?>(null) }
    var verifiedPassword by remember { mutableStateOf<String?>(null) }
    
    // Callback để logout
    val onLogout: () -> Unit = {
        sessionManager.clearSession()
        
        val settingsManager = com.auto_fe.auto_fe.utils.common.SettingsManager(context)
        settingsManager.resetAllSettingsToDefault()
        
        isLoggedIn = false
        accessToken = null
        selectedPrescriptionId = null
        showCreatePrescription = false
        showVerification = false
        showProfile = false
        showForgotPassword = false
        showChangePassword = false
        verificationEmail = ""
        verificationPassword = ""
        verifiedEmail = null
        verifiedPassword = null
        selectedTab = 0
    }
    
    // Initialize ApiClient với callback khi token hết hạn
    LaunchedEffect(Unit) {
        ApiClient.initialize(sessionManager) {
            // Token hết hạn - hiển thị thông báo và logout
            android.os.Handler(android.os.Looper.getMainLooper()).post {
                Toast.makeText(
                    context,
                    "Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.",
                    Toast.LENGTH_LONG
                ).show()
                onLogout()
            }
        }
    }
    
    // Auto-login nếu đã có session
    LaunchedEffect(Unit) {
        if (sessionManager.isLoggedIn() && sessionManager.getAccessToken() != null) {
            isLoggedIn = true
            accessToken = sessionManager.getAccessToken()
            // Không cần chuyển tab, để user ở tab hiện tại
            Log.d("MainActivity", "Auto-login successful: ${sessionManager.getUserEmail()}")
        }
    }

    // Handle create chat with user
    LaunchedEffect(selectedUserId) {
        selectedUserId?.let { userId ->
            selectedUserName?.let { userName ->
                try {
                    val chatService = com.auto_fe.auto_fe.service.be.ChatService()
                    val result = chatService.getOrCreateChatWithUser(userId, accessToken ?: "")
                    result.onSuccess { chatId ->
                        // Navigate đến ChatDetailScreen với chatId
                        selectedChatId = chatId
                        selectedChatName = userName
                        showChatDetail = true
                        Log.d("MainActivity", "Open chat with user: $userId, chatId: $chatId")
                    }.onFailure { error ->
                        Toast.makeText(
                            context,
                            "Lỗi: ${error.message}",
                            Toast.LENGTH_LONG
                        ).show()
                    }
                } catch (e: Exception) {
                    Toast.makeText(
                        context,
                        "Lỗi: ${e.message}",
                        Toast.LENGTH_LONG
                    ).show()
                } finally {
                    // Reset state
                    selectedUserId = null
                    selectedUserName = null
                }
            }
        }
    }

    // BackHandler để xử lý nút back
    BackHandler(enabled = selectedElderUserId != null || selectedPrescriptionId != null || showCreatePrescription || showCreateStandaloneMedication || showVerification || showProfile || showForgotPassword || showChangePassword || showNotificationHistory || showEmergencyContact || showMedicalDocuments || showChatList || showSearchUser || showSearchConnection || showChatDetail) {
        when {
            //  PRIORITY 1: Handle detail/create screens first (highest priority overlays)
            showCreatePrescription -> {
                showCreatePrescription = false
                editPrescriptionId = null
            }
            selectedPrescriptionId != null -> {
                selectedPrescriptionId = null
            }
            showCreateStandaloneMedication -> {
                showCreateStandaloneMedication = false
            }
            //  PRIORITY 2: Main screens
            selectedElderUserId != null -> {
                selectedElderUserId = null
                selectedElderUserName = null
            }
            // Nếu đang ở màn chat detail → quay về chat list
            showChatDetail -> {
                showChatDetail = false
                selectedChatId = null
                selectedReceiverId = null
                selectedChatName = null
                showChatList = true
            }
            // Nếu đang ở màn search connection → quay về danh sách
            showSearchConnection -> {
                showSearchConnection = false
            }
            // Nếu đang ở màn search user → quay về chat list
            showSearchUser -> {
                showSearchUser = false
            }
            // Nếu đang ở màn chat list → quay về danh sách
            showChatList -> {
                showChatList = false
            }
            // Nếu đang ở màn emergency contact → quay về danh sách
            showEmergencyContact -> {
                showEmergencyContact = false
            }
            // Nếu đang ở màn medical documents → quay về profile
            showMedicalDocuments -> {
                showMedicalDocuments = false
                showProfile = true
            }
            // Nếu đang ở màn notification history → quay về danh sách
            showNotificationHistory -> {
                showNotificationHistory = false
            }
            // Nếu đang ở màn change password → quay về profile
            showChangePassword -> {
                showChangePassword = false
                showProfile = true
            }
            // Nếu đang ở màn forgot password → quay về auth
            showForgotPassword -> {
                showForgotPassword = false
            }
            // Nếu đang ở màn profile → quay về danh sách
            showProfile -> {
                showProfile = false
            }
            // Nếu đang ở màn verification → quay về auth
            showVerification -> {
                showVerification = false
                verificationEmail = ""
                verificationPassword = ""
            }
        }
    }

    when {
        // ===== PRIORITY 1: Fullscreen overlays (highest priority) =====
        
        // Màn hình tạo đơn thuốc mới (fullscreen, không có bottom nav)
        showCreatePrescription && accessToken != null -> {
            CreatePrescriptionScreen(
                accessToken = accessToken!!,
                onBackClick = { 
                    showCreatePrescription = false
                    editPrescriptionId = null  // Reset edit state
                },
                onSuccess = {
                    showCreatePrescription = false
                    editPrescriptionId = null  //  Reset edit state
                    // Optionally refresh prescription list
                },
                editPrescriptionId = editPrescriptionId,  //  Truyền prescription ID để edit
                elderUserId = selectedElderUserId,  //  Pass if Supervisor is creating for Elder
                elderUserName = selectedElderUserName
            )
        }
        
        // Màn hình chi tiết đơn thuốc (fullscreen, không có bottom nav)
        selectedPrescriptionId != null && accessToken != null -> {
            PrescriptionDetailScreen(
                prescriptionId = selectedPrescriptionId!!,
                accessToken = accessToken ?: "",
                elderUserId = selectedElderUserId,  //  Pass elderUserId for Supervisor mode
                onBackClick = { selectedPrescriptionId = null },
                onEditClick = { prescriptionId ->
                    editPrescriptionId = prescriptionId
                    showCreatePrescription = true
                    selectedPrescriptionId = null
                }
            )
        }
        
        // Màn hình tạo thuốc ngoài đơn (fullscreen)
        showCreateStandaloneMedication && accessToken != null -> {
            CreateStandaloneMedicationScreen(
                accessToken = accessToken!!,
                elderUserId = selectedElderUserId,  // Pass elderUserId for Supervisor mode
                elderUserName = selectedElderUserName,
                onDismiss = { showCreateStandaloneMedication = false },
                onSuccess = { showCreateStandaloneMedication = false }
            )
        }
        
        // ===== PRIORITY 2: Main screens =====
        
        //  Màn hình Supervisor xem thuốc của Elder (fullscreen) - Dùng MedicationTabScreen
        selectedElderUserId != null && accessToken != null -> {
            MedicationTabScreen(
                accessToken = accessToken!!,
                currentUserId = sessionManager.getUserId(),  // Add current user ID
                userName = sessionManager.getUserName() ?: "Supervisor",
                userEmail = sessionManager.getUserEmail() ?: "",
                userAvatar = sessionManager.getUserAvatar(),
                elderUserId = selectedElderUserId,  //  Pass elderUserId để load thuốc của Elder
                elderUserName = selectedElderUserName,
                canViewMedications = supervisorCanView,  // Pass quyền xem
                canUpdateMedications = supervisorCanUpdate,  // Pass quyền sửa
                onPrescriptionClick = { prescriptionId ->
                    selectedPrescriptionId = prescriptionId
                },
                onBackClick = {
                    selectedElderUserId = null
                    selectedElderUserName = null
                },
                onCreatePrescriptionClick = {
                    // Supervisor tạo đơn thuốc cho Elder
                    showCreatePrescription = true
                },
                onCreateStandaloneMedicationClick = {
                    // Supervisor tạo thuốc ngoài đơn cho Elder
                    showCreateStandaloneMedication = true
                },
                onLogout = { onLogout() },
                onProfileClick = { showProfile = true },
                onNotificationHistoryClick = { showNotificationHistory = true },
                onEmergencyContactClick = { showEmergencyContact = true },
                onManageConnectionsClick = { /* TODO */ },
                onChatClick = { showChatList = true }
            )
        }
        // Màn hình Chat Detail (fullscreen)
        showChatDetail && accessToken != null -> {
            com.auto_fe.auto_fe.ui.screens.ChatDetailScreen(
                accessToken = accessToken ?: "",
                currentUserId = sessionManager.getUserId() ?: 0L,
                userEmail = sessionManager.getUserEmail() ?: "",
                chatId = selectedChatId,
                receiverId = selectedReceiverId,
                chatName = selectedChatName,
                onBackClick = {
                    showChatDetail = false
                    selectedChatId = null
                    selectedReceiverId = null
                    selectedChatName = null
                    showChatList = true
                }
            )
        }
        // Màn hình Search User (fullscreen - cho chat)
        showSearchUser && accessToken != null -> {
            SearchUserScreen(
                accessToken = accessToken ?: "",
                onUserClick = { userId, userName ->
                    showSearchUser = false
                    selectedUserId = userId
                    selectedUserName = userName
                },
                onBackClick = {
                    showSearchUser = false
                }
            )
        }
        // Màn hình Search User For Connection (fullscreen - cho yêu cầu kết nối)
        showSearchConnection && accessToken != null -> {
            SearchUserForConnectionScreen(
                accessToken = accessToken ?: "",
                onBack = {
                    showSearchConnection = false
                }
            )
        }
        // Màn hình Chat List (fullscreen)
        showChatList && accessToken != null -> {
            ChatListScreen(
                accessToken = accessToken ?: "",
                currentUserId = sessionManager.getUserId() ?: 0L,
                onChatClick = { chatId, chatName ->
                    // Navigate to chat detail screen
                    selectedChatId = chatId
                    selectedChatName = chatName
                    showChatDetail = true
                    Log.d("MainActivity", "Navigate to chat: $chatId with name: $chatName")
                },
                onSearchUserClick = {
                    showSearchUser = true
                },
                onBackClick = {
                    showChatList = false
                }
            )
        }
        // Màn hình change password (fullscreen)
        showChangePassword && accessToken != null -> {
            ChangePasswordScreen(
                accessToken = accessToken!!,
                onBackClick = { 
                    showChangePassword = false
                    showProfile = true
                },
                onSuccess = {
                    showChangePassword = false
                    showProfile = true
                }
            )
        }
        // Màn hình forgot password (fullscreen)
        showForgotPassword -> {
            ForgotPasswordScreen(
                onBackClick = { showForgotPassword = false },
                onSuccessNavigateToLogin = { showForgotPassword = false }
            )
        }
        // Màn hình profile (fullscreen)
        showProfile && accessToken != null -> {
            ProfileScreen(
                accessToken = accessToken!!,
                onBackClick = { showProfile = false },
                onChangePasswordClick = {
                    showProfile = false
                    showChangePassword = true
                },
                onMedicalDocumentsClick = {
                    showProfile = false
                    showMedicalDocuments = true
                }
            )
        }
        // Màn hình thông báo (fullscreen)
        showNotificationHistory && accessToken != null -> {
            val token = accessToken // Smart cast fix
            if (token != null) {
                NotificationScreen(
                    accessToken = token,
                    onBackClick = { showNotificationHistory = false },
                    onNotificationClick = { notification ->
                        // Handle notification click - Navigate based on type
                        when (notification.notificationType) {
                            "MEDICATION_REMINDER" -> {
                                // TODO: Navigate to medication log detail
                                notification.relatedMedicationLogId?.let { logId ->
                                    android.util.Log.d("MainActivity", "Navigate to medication log: $logId")
                                    // navController.navigate("medication-log/$logId")
                                }
                            }
                            "ELDER_MISSED_MEDICATION", "ELDER_LATE_MEDICATION" -> {
                                // TODO: Navigate to medication log detail
                                notification.relatedMedicationLogId?.let { logId ->
                                    android.util.Log.d("MainActivity", "Navigate to medication log: $logId")
                                }
                            }
                            "ELDER_HEALTH_ALERT" -> {
                                // TODO: Navigate to elder health detail
                                notification.relatedElderId?.let { elderId ->
                                    android.util.Log.d("MainActivity", "Navigate to elder: $elderId")
                                }
                            }
                            "RELATIONSHIP_REQUEST" -> {
                                // TODO: Navigate to relationship requests
                                android.util.Log.d("MainActivity", "Navigate to relationship requests")
                            }
                            else -> {
                                android.util.Log.d("MainActivity", "Notification clicked: ${notification.title}")
                            }
                        }
                    }
                )
            }
        }
        // Màn hình liên hệ khẩn cấp (fullscreen)
        showEmergencyContact && accessToken != null -> {
            val token = accessToken // Smart cast fix
            if (token != null) {
                EmergencyContactScreen(
                    accessToken = token,
                    onBackClick = { showEmergencyContact = false }
                )
            }
        }
        // Màn hình Medical Documents (fullscreen)
        showMedicalDocuments && accessToken != null -> {
            val token = accessToken // Smart cast fix
            if (token != null) {
                MedicalDocumentsScreen(
                    accessToken = token,
                    onBackClick = { showMedicalDocuments = false }
                )
            }
        }
        // Màn hình Manage Connections (fullscreen)
        showManageConnections && accessToken != null -> {
            val token = accessToken // Smart cast fix
            val role = sessionManager.getUserRole() ?: "ELDER"
            if (token != null) {
                ManageConnectionsScreen(
                    accessToken = token,
                    userRole = role,
                    onBackClick = { showManageConnections = false },
                    onSearchUserClick = { showSearchConnection = true }
                )
            }
        }
        
        // Màn hình verification (fullscreen)
        showVerification -> {
            VerificationScreen(
                email = verificationEmail,
                password = verificationPassword,
                onVerificationSuccess = { email, password ->
                    // Verification thành công → quay về login với autofill
                    showVerification = false
                    verificationEmail = ""
                    verificationPassword = ""
                    verifiedEmail = email
                    verifiedPassword = password
                    Toast.makeText(context, "Xác thực thành công! Vui lòng đăng nhập.", Toast.LENGTH_LONG).show()
                },
                onBackClick = {
                    showVerification = false
                    verificationEmail = ""
                    verificationPassword = ""
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
                        onTabSelected = { selectedTab = it },
                        isLoggedIn = isLoggedIn
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
                        0 -> {
                            // Tab Đơn thuốc/Danh sách Elder - chỉ hiển thị khi đã login
                            if (isLoggedIn && accessToken != null) {
                                val displayName = sessionManager.getUserName()
                                    ?.takeIf { it.isNotBlank() && it != "null" } 
                                    ?: "Người dùng"
                                
                                val context = LocalContext.current
                                val userRole = sessionManager.getUserRole() // Get role từ session
                                
                                // Phân biệt Elder vs Supervisor
                                if (userRole == "SUPERVISOR") {
                                    // SUPERVISOR → Hiển thị danh sách Elder
                                    ElderListScreen(
                                        accessToken = accessToken!!,
                                        userAvatar = sessionManager.getUserAvatar(),
                                        onElderClick = { elderUserId, elderUserName ->
                                            // Set selected elder
                                            selectedElderUserId = elderUserId
                                            selectedElderUserName = elderUserName
                                            android.util.Log.d("MainActivity", "Selected Elder: $elderUserId - $elderUserName")
                                            
                                            // Gọi API getRole để lấy permissions
                                            kotlinx.coroutines.CoroutineScope(kotlinx.coroutines.Dispatchers.Main).launch {
                                                try {
                                                    val relationshipService = com.auto_fe.auto_fe.service.be.RelationshipService()
                                                    val result = relationshipService.getRole(accessToken!!, elderUserId)
                                                    
                                                    result.fold(
                                                        onSuccess = { permission ->
                                                            supervisorCanView = permission.canViewMedications
                                                            supervisorCanUpdate = permission.canUpdateMedications
                                                            android.util.Log.d("MainActivity", "Permissions loaded - canView: $supervisorCanView, canUpdate: $supervisorCanUpdate")
                                                            
                                                            if (!permission.canViewMedications) {
                                                                android.widget.Toast.makeText(
                                                                    context,
                                                                    "Bạn không có quyền xem thông tin thuốc của ${permission.elderName}",
                                                                    android.widget.Toast.LENGTH_LONG
                                                                ).show()
                                                            }
                                                        },
                                                        onFailure = { error ->
                                                            android.util.Log.e("MainActivity", "Failed to get permissions: ${error.message}")
                                                            android.widget.Toast.makeText(
                                                                context,
                                                                "Không thể lấy thông tin quyền: ${error.message}",
                                                                android.widget.Toast.LENGTH_LONG
                                                            ).show()
                                                            // Reset permissions về default
                                                            supervisorCanView = true
                                                            supervisorCanUpdate = true
                                                        }
                                                    )
                                                } catch (e: Exception) {
                                                    android.util.Log.e("MainActivity", "Error calling getRole: ${e.message}")
                                                    supervisorCanView = true
                                                    supervisorCanUpdate = true
                                                }
                                            }
                                        },
                                        onSearchUserClick = {
                                            showSearchConnection = true
                                        },
                                        onChatClick = {
                                            showChatList = true
                                        },
                                        onProfileClick = {
                                            showProfile = true
                                        },
                                        onNotificationHistoryClick = {
                                            showNotificationHistory = true
                                        },
                                        onManageConnectionsClick = {
                                            showManageConnections = true
                                        },
                                        onLogout = {
                                            onLogout()
                                        }
                                    )
                                } else {
                                    // ELDER → Hiển thị danh sách đơn thuốc
                                    MedicationTabScreen(
                                        accessToken = accessToken!!,
                                        currentUserId = sessionManager.getUserId(),  // Add current user ID
                                        userName = displayName,
                                        userEmail = sessionManager.getUserEmail() ?: "",
                                        userAvatar = sessionManager.getUserAvatar(),
                                        onPrescriptionClick = { prescriptionId ->
                                            selectedPrescriptionId = prescriptionId
                                        },
                                        onCreatePrescriptionClick = {
                                            showCreatePrescription = true
                                        },
                                        onCreateStandaloneMedicationClick = {
                                            showCreateStandaloneMedication = true
                                        },
                                        onChatClick = {
                                            showChatList = true
                                        },
                                        onProfileClick = {
                                            showProfile = true
                                        },
                                        onNotificationHistoryClick = {
                                            showNotificationHistory = true
                                        },
                                        onEmergencyContactClick = {
                                            showEmergencyContact = true
                                        },
                                        onManageConnectionsClick = {
                                            showManageConnections = true
                                        },
                                        onLogout = {
                                            onLogout()
                                        }
                                    )
                                }
                            } else {
                                // Chưa login → Hiển thị màn đăng nhập
                                AuthScreen(
                                    onLoginSuccess = { token, userEmail, userName, userId, userAvatar, userRole ->
                                        // Lưu session với role
                                        sessionManager.saveLoginSession(
                                            accessToken = token,
                                            userEmail = userEmail,
                                            userName = userName,
                                            userId = userId,
                                            userAvatar = userAvatar,
                                            userRole = userRole // Lưu role
                                        )
                                        accessToken = token
                                        isLoggedIn = true
                                        // Clear verified credentials sau khi login thành công
                                        verifiedEmail = null
                                        verifiedPassword = null
                                        // Giữ nguyên tab 0 để hiển thị PrescriptionList
                                        

                                        // if (userRole == "SUPERVISOR") {
                                        //     // Chuyển sang màn hình giám sát
                                        // } else {
                                        //     // ELDER - giữ nguyên
                                        // }
                                    },
                                    onVerificationClick = { email, password ->
                                        // Chuyển sang màn hình verification với email và password
                                        verificationEmail = email
                                        verificationPassword = password
                                        showVerification = true
                                    },
                                    onForgotPasswordClick = {
                                        // Chuyển sang màn hình quên mật khẩu
                                        showForgotPassword = true
                                    },
                                    verifiedEmail = verifiedEmail,
                                    verifiedPassword = verifiedPassword
                                )
                            }
                        }
                        1 -> VoiceScreen() // Màn hình ghi âm là mặc định
                        2 -> GuideScreen() // Màn hình hướng dẫn
                        3 -> SettingsScreen()
                    }
                }
            }
        }
    }
}
    
