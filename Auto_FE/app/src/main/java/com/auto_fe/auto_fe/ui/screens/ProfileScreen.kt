package com.auto_fe.auto_fe.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.platform.LocalContext
import android.widget.Toast
import android.content.Context
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import coil.compose.AsyncImage
import com.auto_fe.auto_fe.service.be.UserService
import com.auto_fe.auto_fe.ui.theme.*
import com.auto_fe.auto_fe.ui.theme.AppTextSize
import com.auto_fe.auto_fe.utils.be.SessionManager
import kotlinx.coroutines.launch
import java.io.File
import androidx.compose.material3.rememberDatePickerState
import androidx.compose.material3.DatePicker
import androidx.compose.material3.DatePickerDefaults

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ProfileScreen(
    accessToken: String,
    onBackClick: () -> Unit = {},
    onChangePasswordClick: () -> Unit = {},
    onMedicalDocumentsClick: () -> Unit = {}
) {
    val context = LocalContext.current
    val userService = remember { UserService() }
    val sessionManager = remember { SessionManager(context) }
    val coroutineScope = rememberCoroutineScope()
    
    var profileData by remember { mutableStateOf<UserService.ProfileData?>(null) }
    var isLoading by remember { mutableStateOf(true) }
    var isEditMode by remember { mutableStateOf(false) }
    var showSuccessMessage by remember { mutableStateOf(false) }
    var isUploadingAvatar by remember { mutableStateOf(false) }

    // Function để load profile
    fun loadProfile() {
        coroutineScope.launch {
            isLoading = true
            val result = userService.getUserProfile(accessToken)
            isLoading = false
            
            result.onSuccess { response ->
                profileData = response.data
                // Lưu avatar vào SessionManager nếu có
                response.data?.avatar?.let { avatarUrl ->
                    if (avatarUrl.isNotBlank()) {
                        sessionManager.updateUserAvatar(avatarUrl)
                    }
                }
            }.onFailure { error ->
                Toast.makeText(context, "${error.message}", Toast.LENGTH_LONG).show()
            }
        }
    }

    // Image picker launcher
    val imagePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            coroutineScope.launch {
                try {
                    isUploadingAvatar = true
                    
                    // Convert URI to File
                    val file = uriToFile(context, uri)
                    
                    // Upload avatar
                    val result = userService.uploadAvatar(accessToken, file)
                    
                    result.onSuccess { avatarUrl ->
                        // Lưu avatar URL vào SessionManager
                        sessionManager.updateUserAvatar(avatarUrl)
                        
                        Toast.makeText(context, "Cập nhật ảnh đại diện thành công", Toast.LENGTH_SHORT).show()
                        // Reload profile để lấy avatar mới
                        loadProfile()
                    }.onFailure { error ->
                        Toast.makeText(context, "${error.message}", Toast.LENGTH_LONG).show()
                    }
                    
                    isUploadingAvatar = false
                    
                    // Clean up temp file
                    file.delete()
                } catch (e: Exception) {
                    Toast.makeText(context, "Lỗi: ${e.message}", Toast.LENGTH_LONG).show()
                    isUploadingAvatar = false
                }
            }
        }
    }

    // Load profile khi màn hình mở
    LaunchedEffect(Unit) {
        loadProfile()
    }

    // Show success message
    LaunchedEffect(showSuccessMessage) {
        if (showSuccessMessage) {
            kotlinx.coroutines.delay(2000)
            showSuccessMessage = false
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(if (isEditMode) "Chỉnh sửa hồ sơ" else "Thông tin cá nhân") },
                navigationIcon = {
                    IconButton(onClick = {
                        if (isEditMode) {
                            isEditMode = false
                        } else {
                            onBackClick()
                        }
                    }) {
                        Icon(Icons.Default.ArrowBack, "Quay lại")
                    }
                },
                actions = {
                    if (!isEditMode && profileData != null) {
                        IconButton(onClick = { isEditMode = true }) {
                            Icon(
                                Icons.Default.Edit,
                                "Chỉnh sửa",
                                tint = DarkPrimary
                            )
                        }
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = DarkSurface,
                    titleContentColor = DarkOnSurface,
                    navigationIconContentColor = DarkOnSurface
                )
            )
        },
        containerColor = DarkBackground,
        snackbarHost = {
            if (showSuccessMessage) {
                Snackbar(
                    modifier = Modifier.padding(16.dp),
                    containerColor = DarkPrimary
                ) {
                    Text("✓ Cập nhật thành công!")
                }
            }
        }
    ) { paddingValues ->
        when {
            isLoading -> {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(paddingValues),
                    contentAlignment = Alignment.Center
                ) {
                    CircularProgressIndicator(color = DarkPrimary)
                }
            }
            
            profileData != null -> {
                if (isEditMode) {
                    EditProfileContent(
                        profileData = profileData!!,
                        accessToken = accessToken,
                        userService = userService,
                        modifier = Modifier.padding(paddingValues),
                        onUpdateSuccess = { updatedData ->
                            profileData = updatedData
                            isEditMode = false
                            showSuccessMessage = true
                        },
                        onUpdateError = { error ->
                            Toast.makeText(context, "$error", Toast.LENGTH_LONG).show()
                        }
                    )
                } else {
                    ProfileContent(
                        profileData = profileData!!,
                        onChangePasswordClick = onChangePasswordClick,
                        onMedicalDocumentsClick = onMedicalDocumentsClick,
                        onAvatarClick = { imagePickerLauncher.launch("image/*") },
                        isUploadingAvatar = isUploadingAvatar,
                        modifier = Modifier.padding(paddingValues)
                    )
                }
            }
        }
    }
}

@Composable
fun ProfileContent(
    profileData: UserService.ProfileData,
    onChangePasswordClick: () -> Unit = {},
    onMedicalDocumentsClick: () -> Unit = {},
    onAvatarClick: () -> Unit = {},
    isUploadingAvatar: Boolean = false,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp)
    ) {
        // Avatar và tên
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(
                containerColor = DarkSurface
            )
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                // Avatar - Clickable để upload
                Box(
                    modifier = Modifier
                        .size(100.dp)
                        .clip(CircleShape)
                        .background(DarkPrimary.copy(alpha = 0.2f))
                        .clickable(enabled = !isUploadingAvatar) { onAvatarClick() },
                    contentAlignment = Alignment.Center
                ) {
                    if (isUploadingAvatar) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(40.dp),
                            color = DarkPrimary,
                            strokeWidth = 3.dp
                        )
                    } else if (!profileData.avatar.isNullOrBlank()) {
                        AsyncImage(
                            model = profileData.avatar,
                            contentDescription = "Avatar",
                            modifier = Modifier
                                .fillMaxSize()
                                .clip(CircleShape)
                        )
                    } else {
                        Icon(
                            imageVector = Icons.Default.Person,
                            contentDescription = "Avatar placeholder",
                            tint = DarkPrimary,
                            modifier = Modifier.size(60.dp)
                        )
                    }
                    
                    // Camera icon overlay
                    if (!isUploadingAvatar) {
                        Box(
                            modifier = Modifier
                                .align(Alignment.BottomEnd)
                                .size(32.dp)
                                .clip(CircleShape)
                                .background(DarkPrimary),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(
                                imageVector = Icons.Default.Edit,
                                contentDescription = "Change avatar",
                                tint = DarkOnPrimary,
                                modifier = Modifier.size(18.dp)
                            )
                        }
                    }
                }
                
                Spacer(modifier = Modifier.height(16.dp))
                
                // Tên
                Text(
                    text = profileData.fullName?.takeIf { it.isNotBlank() && it != "null" } ?: "Người dùng",
                    fontSize = AppTextSize.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = DarkOnSurface
                )
                
                // Email
                profileData.email?.takeIf { it.isNotBlank() && it != "null" }?.let { email ->
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = email,
                        fontSize = AppTextSize.bodyMedium,
                        color = DarkOnSurface.copy(alpha = 0.7f)
                    )
                }
                
                // Status badge
                Spacer(modifier = Modifier.height(12.dp))
                Surface(
                    shape = RoundedCornerShape(12.dp),
                    color = if (profileData.isActive == true) 
                        DarkPrimary.copy(alpha = 0.2f) 
                    else 
                        AIError.copy(alpha = 0.2f)
                ) {
                    Text(
                        text = if (profileData.isActive == true) "✓ Đã xác thực" else "⚠ Chưa xác thực",
                        color = if (profileData.isActive == true) AIPrimarySoft else AIError,
                        fontSize = AppTextSize.bodySmall,
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp)
                    )
                }
            }
        }
        
        Spacer(modifier = Modifier.height(16.dp))
        
        // Thông tin cá nhân
        Text(
            text = "Thông tin cá nhân",
            fontSize = AppTextSize.titleSmall,
            fontWeight = FontWeight.Bold,
            color = DarkOnSurface,
            modifier = Modifier.padding(start = 4.dp, bottom = 8.dp)
        )
        
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(
                containerColor = DarkSurface
            )
        ) {
            Column(
                modifier = Modifier.padding(16.dp)
            ) {
                ProfileInfoRow(
                    icon = Icons.Default.Phone,
                    label = "Số điện thoại",
                    value = profileData.phoneNumber?.takeIf { it.isNotBlank() && it != "null" } ?: "Chưa cập nhật"
                )
                
                Divider(
                    color = DarkOnSurface.copy(alpha = 0.1f),
                    modifier = Modifier.padding(vertical = 12.dp)
                )
                
                ProfileInfoRow(
                    icon = Icons.Default.DateRange,
                    label = "Ngày sinh",
                    value = profileData.dateOfBirth?.takeIf { it.isNotBlank() && it != "null" } ?: "Chưa cập nhật"
                )
                
                Divider(
                    color = DarkOnSurface.copy(alpha = 0.1f),
                    modifier = Modifier.padding(vertical = 12.dp)
                )
                
                ProfileInfoRow(
                    icon = Icons.Default.Person,
                    label = "Giới tính",
                    value = when(profileData.gender) {
                        "MALE" -> "Nam"
                        "FEMALE" -> "Nữ"
                        "OTHER" -> "Khác"
                        else -> "Chưa cập nhật"
                    }
                )
                
                Divider(
                    color = DarkOnSurface.copy(alpha = 0.1f),
                    modifier = Modifier.padding(vertical = 12.dp)
                )
                
                ProfileInfoRow(
                    icon = Icons.Default.Home,
                    label = "Địa chỉ",
                    value = profileData.address?.takeIf { it.isNotBlank() && it != "null" } ?: "Chưa cập nhật"
                )
            }
        }
        
        Spacer(modifier = Modifier.height(16.dp))
        
        // Thông tin sức khỏe - CHỈ hiển thị cho ELDER
        if (profileData.role == "ELDER") {
            Text(
                text = "Thông tin sức khỏe",
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold,
                color = DarkOnSurface,
                modifier = Modifier.padding(start = 4.dp, bottom = 8.dp)
            )
            
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(
                    containerColor = DarkSurface
                )
            ) {
                Column(
                    modifier = Modifier.padding(16.dp)
                ) {
                    ProfileInfoRow(
                        icon = Icons.Default.Favorite,
                        label = "Nhóm máu",
                        value = profileData.bloodType?.takeIf { it.isNotBlank() && it != "null" }?.let { 
                            it.replace("_", " ")
                        } ?: "Chưa cập nhật"
                    )
                    
                    Divider(
                        color = DarkOnSurface.copy(alpha = 0.1f),
                        modifier = Modifier.padding(vertical = 12.dp)
                    )
                    
                    ProfileInfoRow(
                        icon = Icons.Default.Info,
                        label = "Chiều cao",
                        value = profileData.height?.let { "$it cm" } ?: "Chưa cập nhật"
                    )
                    
                    Divider(
                        color = DarkOnSurface.copy(alpha = 0.1f),
                        modifier = Modifier.padding(vertical = 12.dp)
                    )
                    
                    ProfileInfoRow(
                        icon = Icons.Default.Star,
                        label = "Cân nặng",
                        value = profileData.weight?.let { "$it kg" } ?: "Chưa cập nhật"
                    )
                }
            }
            
            Spacer(modifier = Modifier.height(16.dp))
        }
        
        // Thông tin công việc - CHỈ hiển thị cho SUPERVISOR
        if (profileData.role == "SUPERVISOR") {
            Text(
                text = "Thông tin công việc",
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold,
                color = DarkOnSurface,
                modifier = Modifier.padding(start = 4.dp, bottom = 8.dp)
            )
            
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(
                    containerColor = DarkSurface
                )
            ) {
                Column(
                    modifier = Modifier.padding(16.dp)
                ) {
                    ProfileInfoRow(
                        icon = Icons.Default.AccountBox,
                        label = "Nghề nghiệp",
                        value = profileData.occupation?.takeIf { it.isNotBlank() && it != "null" } ?: "Chưa cập nhật"
                    )
                    
                    Divider(
                        color = DarkOnSurface.copy(alpha = 0.1f),
                        modifier = Modifier.padding(vertical = 12.dp)
                    )
                    
                    ProfileInfoRow(
                        icon = Icons.Default.LocationOn,
                        label = "Nơi làm việc",
                        value = profileData.workplace?.takeIf { it.isNotBlank() && it != "null" } ?: "Chưa cập nhật"
                    )
                }
            }
            
            Spacer(modifier = Modifier.height(16.dp))
        }
        
        // Hồ sơ bệnh án - CHỈ hiển thị cho ELDER
        if (profileData.role == "ELDER") {
            Text(
                text = "Tài liệu y tế",
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold,
                color = DarkOnSurface,
                modifier = Modifier.padding(start = 4.dp, bottom = 8.dp)
            )
            
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(
                    containerColor = DarkSurface
                )
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable { onMedicalDocumentsClick() }
                        .padding(16.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        imageVector = Icons.Default.Description,
                        contentDescription = "Hồ sơ bệnh án",
                        tint = DarkPrimary,
                        modifier = Modifier.size(24.dp)
                    )
                    
                    Spacer(modifier = Modifier.width(16.dp))
                    
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = "Hồ sơ bệnh án",
                            fontSize = AppTextSize.bodyMedium,
                            color = DarkOnSurface,
                            fontWeight = FontWeight.Medium
                        )
                        Text(
                            text = "Quản lý tài liệu y tế, kết quả xét nghiệm",
                            fontSize = AppTextSize.bodySmall,
                            color = DarkOnSurface.copy(alpha = 0.6f)
                        )
                    }
                    
                    Icon(
                        imageVector = Icons.Default.KeyboardArrowRight,
                        contentDescription = "Go",
                        tint = DarkOnSurface.copy(alpha = 0.4f)
                    )
                }
            }
            
            Spacer(modifier = Modifier.height(16.dp))
        }
        
        // Bảo mật
        Text(
            text = "Bảo mật",
            fontSize = 18.sp,
            fontWeight = FontWeight.Bold,
            color = DarkOnSurface,
            modifier = Modifier.padding(start = 4.dp, bottom = 8.dp)
        )
        
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(
                containerColor = DarkSurface
            )
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable { onChangePasswordClick() }
                    .padding(16.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    imageVector = Icons.Default.Lock,
                    contentDescription = "Đổi mật khẩu",
                    tint = DarkPrimary,
                    modifier = Modifier.size(24.dp)
                )
                
                Spacer(modifier = Modifier.width(16.dp))
                
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "Đổi mật khẩu",
                        fontSize = AppTextSize.bodyMedium,
                        color = DarkOnSurface,
                        fontWeight = FontWeight.Medium
                    )
                    Text(
                        text = "Thay đổi mật khẩu của bạn",
                        fontSize = AppTextSize.bodySmall,
                        color = DarkOnSurface.copy(alpha = 0.6f)
                    )
                }
                
                Icon(
                    imageVector = Icons.Default.KeyboardArrowRight,
                    contentDescription = "Go",
                    tint = DarkOnSurface.copy(alpha = 0.4f)
                )
            }
        }
        
        Spacer(modifier = Modifier.height(24.dp))
    }
}

@Composable
fun ProfileInfoRow(
    icon: ImageVector,
    label: String,
    value: String
) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier.fillMaxWidth()
    ) {
        Icon(
            imageVector = icon,
            contentDescription = label,
            tint = DarkPrimary,
            modifier = Modifier.size(24.dp)
        )
        
        Spacer(modifier = Modifier.width(16.dp))
        
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = label,
                fontSize = AppTextSize.bodySmall,
                color = DarkOnSurface.copy(alpha = 0.6f)
            )
            Spacer(modifier = Modifier.height(2.dp))
            Text(
                text = value,
                fontSize = AppTextSize.bodyMedium,
                color = DarkOnSurface,
                fontWeight = FontWeight.Medium
            )
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun EditProfileContent(
    profileData: UserService.ProfileData,
    accessToken: String,
    userService: UserService,
    modifier: Modifier = Modifier,
    onUpdateSuccess: (UserService.ProfileData) -> Unit,
    onUpdateError: (String) -> Unit
) {
    val coroutineScope = rememberCoroutineScope()
    
    var fullName by remember { mutableStateOf(profileData.fullName ?: "") }
    var phoneNumber by remember { mutableStateOf(profileData.phoneNumber ?: "") }
    var dateOfBirth by remember { mutableStateOf(profileData.dateOfBirth ?: "") }
    var gender by remember { mutableStateOf(profileData.gender ?: "MALE") }
    var address by remember { mutableStateOf(profileData.address ?: "") }
    
    // Elder fields
    var bloodType by remember { mutableStateOf(profileData.bloodType ?: "A_POSITIVE") }
    var height by remember { mutableStateOf(profileData.height?.toString() ?: "") }
    var weight by remember { mutableStateOf(profileData.weight?.toString() ?: "") }
    
    //  Supervisor fields
    var occupation by remember { mutableStateOf(profileData.occupation ?: "") }
    var workplace by remember { mutableStateOf(profileData.workplace ?: "") }
    
    var isUpdating by remember { mutableStateOf(false) }
    var expandedGender by remember { mutableStateOf(false) }
    var expandedBloodType by remember { mutableStateOf(false) }
    var showDatePicker by remember { mutableStateOf(false) } // DatePicker dialog state

    val genderOptions = listOf(
        "MALE" to "Nam",
        "FEMALE" to "Nữ",
        "OTHER" to "Khác"
    )

    val bloodTypeOptions = listOf(
        "A_POSITIVE" to "A+",
        "A_NEGATIVE" to "A-",
        "B_POSITIVE" to "B+",
        "B_NEGATIVE" to "B-",
        "AB_POSITIVE" to "AB+",
        "AB_NEGATIVE" to "AB-",
        "O_POSITIVE" to "O+",
        "O_NEGATIVE" to "O-"
    )

    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp)
    ) {
        // Thông tin cá nhân
        Text(
            text = "Thông tin cá nhân",
            fontSize = AppTextSize.titleSmall,
            fontWeight = FontWeight.Bold,
            color = DarkOnSurface,
            modifier = Modifier.padding(start = 4.dp, bottom = 8.dp)
        )
        
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(
                containerColor = DarkSurface
            )
        ) {
            Column(
                modifier = Modifier.padding(16.dp)
            ) {
                // Họ tên
                OutlinedTextField(
                    value = fullName,
                    onValueChange = { fullName = it },
                    label = { Text("Họ và tên") },
                    leadingIcon = {
                        Icon(Icons.Default.Person, "Họ và tên", tint = DarkPrimary)
                    },
                    modifier = Modifier.fillMaxWidth(),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = DarkPrimary,
                        unfocusedBorderColor = DarkOnSurface.copy(alpha = 0.3f),
                        focusedLabelColor = DarkPrimary,
                        unfocusedLabelColor = DarkOnSurface.copy(alpha = 0.6f),
                        cursorColor = DarkPrimary,
                        focusedTextColor = DarkOnSurface,
                        unfocusedTextColor = DarkOnSurface
                    )
                )
                
                Spacer(modifier = Modifier.height(12.dp))
                
                // Số điện thoại
                OutlinedTextField(
                    value = phoneNumber,
                    onValueChange = { phoneNumber = it },
                    label = { Text("Số điện thoại") },
                    leadingIcon = {
                        Icon(Icons.Default.Phone, "Số điện thoại", tint = DarkPrimary)
                    },
                    modifier = Modifier.fillMaxWidth(),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = DarkPrimary,
                        unfocusedBorderColor = DarkOnSurface.copy(alpha = 0.3f),
                        focusedLabelColor = DarkPrimary,
                        unfocusedLabelColor = DarkOnSurface.copy(alpha = 0.6f),
                        cursorColor = DarkPrimary,
                        focusedTextColor = DarkOnSurface,
                        unfocusedTextColor = DarkOnSurface
                    )
                )
                
                Spacer(modifier = Modifier.height(12.dp))
                
                // Ngày sinh (DatePicker)
                OutlinedTextField(
                    value = dateOfBirth,
                    onValueChange = {}, // Read-only, chỉ cho phép chọn từ DatePicker
                    readOnly = true,
                    label = { Text("Ngày sinh") },
                    leadingIcon = {
                        Icon(Icons.Default.DateRange, "Ngày sinh", tint = DarkPrimary)
                    },
                    trailingIcon = {
                        Icon(Icons.Default.KeyboardArrowDown, "Chọn ngày", tint = DarkOnSurface)
                    },
                    placeholder = { Text("Chọn ngày sinh") },
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable { showDatePicker = true },
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = DarkPrimary,
                        unfocusedBorderColor = DarkOnSurface.copy(alpha = 0.3f),
                        focusedLabelColor = DarkPrimary,
                        unfocusedLabelColor = DarkOnSurface.copy(alpha = 0.6f),
                        cursorColor = DarkPrimary,
                        focusedTextColor = DarkOnSurface,
                        unfocusedTextColor = DarkOnSurface,
                        disabledTextColor = DarkOnSurface,
                        disabledBorderColor = DarkOnSurface.copy(alpha = 0.3f),
                        disabledLabelColor = DarkOnSurface.copy(alpha = 0.6f)
                    ),
                    enabled = false // Disable editing nhưng vẫn clickable
                )
                
                Spacer(modifier = Modifier.height(12.dp))
                
                // Giới tính
                ExposedDropdownMenuBox(
                    expanded = expandedGender,
                    onExpandedChange = { expandedGender = it }
                ) {
                    OutlinedTextField(
                        value = genderOptions.find { it.first == gender }?.second ?: "Nam",
                        onValueChange = {},
                        readOnly = true,
                        label = { Text("Giới tính") },
                        leadingIcon = {
                            Icon(Icons.Default.Person, "Giới tính", tint = DarkPrimary)
                        },
                        trailingIcon = {
                            Icon(
                                if (expandedGender) Icons.Default.KeyboardArrowUp else Icons.Default.KeyboardArrowDown,
                                "Dropdown",
                                tint = DarkOnSurface
                            )
                        },
                        modifier = Modifier
                            .fillMaxWidth()
                            .menuAnchor(),
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedBorderColor = DarkPrimary,
                            unfocusedBorderColor = DarkOnSurface.copy(alpha = 0.3f),
                            focusedLabelColor = DarkPrimary,
                            unfocusedLabelColor = DarkOnSurface.copy(alpha = 0.6f),
                            cursorColor = DarkPrimary,
                            focusedTextColor = DarkOnSurface,
                            unfocusedTextColor = DarkOnSurface
                        )
                    )
                    
                    ExposedDropdownMenu(
                        expanded = expandedGender,
                        onDismissRequest = { expandedGender = false },
                        modifier = Modifier.background(DarkSurface)
                    ) {
                        genderOptions.forEach { option ->
                            DropdownMenuItem(
                                text = { Text(option.second, color = DarkOnSurface) },
                                onClick = {
                                    gender = option.first
                                    expandedGender = false
                                },
                                colors = MenuDefaults.itemColors(
                                    textColor = DarkOnSurface
                                )
                            )
                        }
                    }
                }
                
                Spacer(modifier = Modifier.height(12.dp))
                
                // Địa chỉ
                OutlinedTextField(
                    value = address,
                    onValueChange = { address = it },
                    label = { Text("Địa chỉ") },
                    leadingIcon = {
                        Icon(Icons.Default.Home, "Địa chỉ", tint = DarkPrimary)
                    },
                    modifier = Modifier.fillMaxWidth(),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = DarkPrimary,
                        unfocusedBorderColor = DarkOnSurface.copy(alpha = 0.3f),
                        focusedLabelColor = DarkPrimary,
                        unfocusedLabelColor = DarkOnSurface.copy(alpha = 0.6f),
                        cursorColor = DarkPrimary,
                        focusedTextColor = DarkOnSurface,
                        unfocusedTextColor = DarkOnSurface
                    )
                )
            }
        }
        
        Spacer(modifier = Modifier.height(16.dp))
        
        // Thông tin sức khỏe (chỉ hiển thị cho ELDER)
        if (profileData.role == "ELDER") {
            Text(
                text = "Thông tin sức khỏe",
                fontSize = AppTextSize.titleSmall,
                fontWeight = FontWeight.Bold,
                color = DarkOnSurface,
                modifier = Modifier.padding(start = 4.dp, bottom = 8.dp)
            )
            
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(
                    containerColor = DarkSurface
                )
            ) {
                Column(
                    modifier = Modifier.padding(16.dp)
                ) {
                    // Nhóm máu
                    ExposedDropdownMenuBox(
                        expanded = expandedBloodType,
                        onExpandedChange = { expandedBloodType = it }
                    ) {
                        OutlinedTextField(
                            value = bloodTypeOptions.find { it.first == bloodType }?.second ?: "A+",
                            onValueChange = {},
                            readOnly = true,
                            label = { Text("Nhóm máu") },
                            leadingIcon = {
                                Icon(Icons.Default.Favorite, "Nhóm máu", tint = DarkPrimary)
                            },
                            trailingIcon = {
                                Icon(
                                    if (expandedBloodType) Icons.Default.KeyboardArrowUp else Icons.Default.KeyboardArrowDown,
                                    "Dropdown",
                                    tint = DarkOnSurface
                                )
                            },
                            modifier = Modifier
                                .fillMaxWidth()
                                .menuAnchor(),
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = DarkPrimary,
                                unfocusedBorderColor = DarkOnSurface.copy(alpha = 0.3f),
                                focusedLabelColor = DarkPrimary,
                                unfocusedLabelColor = DarkOnSurface.copy(alpha = 0.6f),
                                cursorColor = DarkPrimary,
                                focusedTextColor = DarkOnSurface,
                                unfocusedTextColor = DarkOnSurface
                            )
                        )
                        
                        ExposedDropdownMenu(
                            expanded = expandedBloodType,
                            onDismissRequest = { expandedBloodType = false },
                            modifier = Modifier.background(DarkSurface)
                        ) {
                            bloodTypeOptions.forEach { option ->
                                DropdownMenuItem(
                                    text = { Text(option.second, color = DarkOnSurface) },
                                    onClick = {
                                        bloodType = option.first
                                        expandedBloodType = false
                                    },
                                    colors = MenuDefaults.itemColors(
                                        textColor = DarkOnSurface
                                    )
                                )
                            }
                        }
                    }
                    
                    Spacer(modifier = Modifier.height(12.dp))
                    
                    // Chiều cao
                    OutlinedTextField(
                        value = height,
                        onValueChange = { height = it },
                        label = { Text("Chiều cao (cm)") },
                        leadingIcon = {
                            Icon(Icons.Default.Info, "Chiều cao", tint = DarkPrimary)
                        },
                        placeholder = { Text("170") },
                        modifier = Modifier.fillMaxWidth(),
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedBorderColor = DarkPrimary,
                            unfocusedBorderColor = DarkOnSurface.copy(alpha = 0.3f),
                            focusedLabelColor = DarkPrimary,
                            unfocusedLabelColor = DarkOnSurface.copy(alpha = 0.6f),
                            cursorColor = DarkPrimary,
                            focusedTextColor = DarkOnSurface,
                            unfocusedTextColor = DarkOnSurface
                        )
                    )
                    
                    Spacer(modifier = Modifier.height(12.dp))
                    
                    // Cân nặng
                    OutlinedTextField(
                        value = weight,
                        onValueChange = { weight = it },
                        label = { Text("Cân nặng (kg)") },
                        leadingIcon = {
                            Icon(Icons.Default.Star, "Cân nặng", tint = DarkPrimary)
                        },
                        placeholder = { Text("65") },
                        modifier = Modifier.fillMaxWidth(),
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedBorderColor = DarkPrimary,
                            unfocusedBorderColor = DarkOnSurface.copy(alpha = 0.3f),
                            focusedLabelColor = DarkPrimary,
                            unfocusedLabelColor = DarkOnSurface.copy(alpha = 0.6f),
                            cursorColor = DarkPrimary,
                            focusedTextColor = DarkOnSurface,
                            unfocusedTextColor = DarkOnSurface
                        )
                    )
                }
            }
            
            Spacer(modifier = Modifier.height(24.dp))
        }
        
        // Thông tin công việc (chỉ hiển thị cho SUPERVISOR)
        if (profileData.role == "SUPERVISOR") {
            Text(
                text = "Thông tin công việc",
                fontSize = AppTextSize.titleSmall,
                fontWeight = FontWeight.Bold,
                color = DarkOnSurface,
                modifier = Modifier.padding(start = 4.dp, bottom = 8.dp)
            )
            
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(
                    containerColor = DarkSurface
                )
            ) {
                Column(
                    modifier = Modifier.padding(16.dp)
                ) {
                    // Nghề nghiệp
                    OutlinedTextField(
                        value = occupation,
                        onValueChange = { occupation = it },
                        label = { Text("Nghề nghiệp") },
                        leadingIcon = {
                            Icon(Icons.Default.Build, "Nghề nghiệp", tint = DarkPrimary)
                        },
                        placeholder = { Text("Bác sĩ, Điều dưỡng, ...") },
                        modifier = Modifier.fillMaxWidth(),
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedBorderColor = DarkPrimary,
                            unfocusedBorderColor = DarkOnSurface.copy(alpha = 0.3f),
                            focusedLabelColor = DarkPrimary,
                            unfocusedLabelColor = DarkOnSurface.copy(alpha = 0.6f),
                            cursorColor = DarkPrimary,
                            focusedTextColor = DarkOnSurface,
                            unfocusedTextColor = DarkOnSurface
                        )
                    )
                    
                    Spacer(modifier = Modifier.height(12.dp))
                    
                    // Nơi làm việc
                    OutlinedTextField(
                        value = workplace,
                        onValueChange = { workplace = it },
                        label = { Text("Nơi làm việc") },
                        leadingIcon = {
                            Icon(Icons.Default.LocationOn, "Nơi làm việc", tint = DarkPrimary)
                        },
                        placeholder = { Text("Bệnh viện, Phòng khám, ...") },
                        modifier = Modifier.fillMaxWidth(),
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedBorderColor = DarkPrimary,
                            unfocusedBorderColor = DarkOnSurface.copy(alpha = 0.3f),
                            focusedLabelColor = DarkPrimary,
                            unfocusedLabelColor = DarkOnSurface.copy(alpha = 0.6f),
                            cursorColor = DarkPrimary,
                            focusedTextColor = DarkOnSurface,
                            unfocusedTextColor = DarkOnSurface
                        )
                    )
                }
            }
            
            Spacer(modifier = Modifier.height(24.dp))
        }
        
        // DatePicker Dialog
        if (showDatePicker) {
            val datePickerState = rememberDatePickerState(
                initialSelectedDateMillis = try {
                    if (dateOfBirth.isNotBlank() && dateOfBirth != "null") {
                        val formatter = java.time.format.DateTimeFormatter.ofPattern("yyyy-MM-dd")
                        java.time.LocalDate.parse(dateOfBirth, formatter)
                            .atStartOfDay(java.time.ZoneId.systemDefault())
                            .toInstant()
                            .toEpochMilli()
                    } else {
                        System.currentTimeMillis()
                    }
                } catch (e: Exception) {
                    System.currentTimeMillis()
                }
            )
            
            androidx.compose.material3.DatePickerDialog(
                onDismissRequest = { showDatePicker = false },
                confirmButton = {
                    TextButton(
                        onClick = {
                            datePickerState.selectedDateMillis?.let { millis ->
                                val instant = java.time.Instant.ofEpochMilli(millis)
                                val localDate = instant.atZone(java.time.ZoneId.systemDefault()).toLocalDate()
                                dateOfBirth = localDate.format(java.time.format.DateTimeFormatter.ofPattern("yyyy-MM-dd"))
                            }
                            showDatePicker = false
                        }
                    ) {
                        Text("Xác nhận", color = DarkPrimary)
                    }
                },
                dismissButton = {
                    TextButton(onClick = { showDatePicker = false }) {
                        Text("Hủy", color = DarkOnSurface.copy(alpha = 0.6f))
                    }
                },
                colors = DatePickerDefaults.colors(
                    containerColor = DarkSurface
                )
            ) {
                DatePicker(
                    state = datePickerState,
                    colors = DatePickerDefaults.colors(
                        containerColor = DarkSurface,
                        titleContentColor = DarkOnSurface,
                        headlineContentColor = DarkOnSurface,
                        weekdayContentColor = DarkOnSurface,
                        subheadContentColor = DarkOnSurface,
                        yearContentColor = DarkOnSurface,
                        currentYearContentColor = DarkPrimary,
                        selectedYearContentColor = DarkOnPrimary,
                        selectedYearContainerColor = DarkPrimary,
                        dayContentColor = DarkOnSurface,
                        selectedDayContentColor = DarkOnPrimary,
                        selectedDayContainerColor = DarkPrimary,
                        todayContentColor = DarkPrimary,
                        todayDateBorderColor = DarkPrimary
                    )
                )
            }
        }
        
        // Nút cập nhật
        Button(
            onClick = {
                coroutineScope.launch {
                    isUpdating = true
                    
                    // Convert date format từ dd/MM/yyyy sang yyyy-MM-dd nếu cần
                    val formattedDate = if (dateOfBirth.isNotBlank() && dateOfBirth != "null") {
                        try {
                            // Kiểm tra nếu là format dd/MM/yyyy thì convert
                            if (dateOfBirth.contains("/")) {
                                val parts = dateOfBirth.split("/")
                                if (parts.size == 3) {
                                    val day = parts[0].padStart(2, '0')
                                    val month = parts[1].padStart(2, '0')
                                    val year = parts[2]
                                    "$year-$month-$day" // yyyy-MM-dd
                                } else {
                                    dateOfBirth
                                }
                            } else {
                                dateOfBirth
                            }
                        } catch (e: Exception) {
                            dateOfBirth
                        }
                    } else {
                        null
                    }
                    
                    val result = userService.updateUserProfile(
                        accessToken = accessToken,
                        fullName = fullName.takeIf { it.isNotBlank() },
                        dateOfBirth = formattedDate,
                        gender = gender.takeIf { it.isNotBlank() && it != "null" },
                        phoneNumber = phoneNumber.takeIf { it.isNotBlank() },
                        address = address.takeIf { it.isNotBlank() },
                        // Elder fields (chỉ gửi khi role là ELDER)
                        bloodType = if (profileData.role == "ELDER") bloodType.takeIf { it.isNotBlank() && it != "null" } else null,
                        height = if (profileData.role == "ELDER") height.toDoubleOrNull() else null,
                        weight = if (profileData.role == "ELDER") weight.toDoubleOrNull() else null,
                        // Supervisor fields (chỉ gửi khi role là SUPERVISOR)
                        occupation = if (profileData.role == "SUPERVISOR") occupation.takeIf { it.isNotBlank() && it != "null" } else null,
                        workplace = if (profileData.role == "SUPERVISOR") workplace.takeIf { it.isNotBlank() && it != "null" } else null
                    )
                    
                    isUpdating = false
                    
                    result.onSuccess { response ->
                        android.util.Log.d("ProfileScreen", "Update success: ${response.message}")
                        response.data?.let { onUpdateSuccess(it) }
                    }.onFailure { error ->
                        android.util.Log.e("ProfileScreen", "Update failed: ${error.message}")
                        onUpdateError(error.message ?: "Không thể cập nhật")
                    }
                }
            },
            modifier = Modifier
                .fillMaxWidth()
                .height(56.dp),
            enabled = !isUpdating && fullName.isNotBlank(),
            shape = RoundedCornerShape(16.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = DarkPrimary,
                disabledContainerColor = DarkPrimary.copy(alpha = 0.5f)
            )
        ) {
            if (isUpdating) {
                CircularProgressIndicator(
                    color = DarkOnPrimary,
                    modifier = Modifier.size(24.dp)
                )
            } else {
                Text(
                    "Cập nhật thông tin",
                    fontSize = AppTextSize.labelMedium,
                    fontWeight = FontWeight.Bold
                )
            }
        }
        
        Spacer(modifier = Modifier.height(16.dp))
    }
}

/**
 * Helper function để convert URI thành File
 */
private fun uriToFile(context: Context, uri: Uri): File {
    val inputStream = context.contentResolver.openInputStream(uri)
        ?: throw Exception("Không thể đọc file")
    
    val extension = when (context.contentResolver.getType(uri)) {
        "image/jpeg", "image/jpg" -> "jpg"
        "image/png" -> "png"
        "image/webp" -> "webp"
        "image/gif" -> "gif"
        else -> "jpg"
    }
    
    val file = File(context.cacheDir, "avatar_${System.currentTimeMillis()}.$extension")
    inputStream.use { input ->
        file.outputStream().use { output ->
            input.copyTo(output)
        }
    }
    return file
}
