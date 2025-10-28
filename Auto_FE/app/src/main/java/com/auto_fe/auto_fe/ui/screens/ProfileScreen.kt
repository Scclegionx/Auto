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
import com.auto_fe.auto_fe.ui.service.UserService
import com.auto_fe.auto_fe.ui.theme.*
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ProfileScreen(
    accessToken: String,
    onBackClick: () -> Unit = {},
    onChangePasswordClick: () -> Unit = {}
) {
    val userService = remember { UserService() }
    val coroutineScope = rememberCoroutineScope()
    
    var profileData by remember { mutableStateOf<UserService.ProfileData?>(null) }
    var isLoading by remember { mutableStateOf(true) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var isEditMode by remember { mutableStateOf(false) }
    var showSuccessMessage by remember { mutableStateOf(false) }

    // Load profile khi màn hình mở
    LaunchedEffect(Unit) {
        coroutineScope.launch {
            isLoading = true
            val result = userService.getUserProfile(accessToken)
            isLoading = false
            
            result.onSuccess { response ->
                profileData = response.data
            }.onFailure { error ->
                errorMessage = error.message
            }
        }
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
            
            errorMessage != null -> {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(paddingValues),
                    contentAlignment = Alignment.Center
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.Center
                    ) {
                        Text(
                            text = "❌",
                            fontSize = 48.sp
                        )
                        Spacer(modifier = Modifier.height(16.dp))
                        Text(
                            text = errorMessage ?: "Lỗi không xác định",
                            color = DarkError,
                            fontSize = 14.sp
                        )
                        Spacer(modifier = Modifier.height(16.dp))
                        Button(
                            onClick = {
                                coroutineScope.launch {
                                    isLoading = true
                                    errorMessage = null
                                    val result = userService.getUserProfile(accessToken)
                                    isLoading = false
                                    
                                    result.onSuccess { response ->
                                        profileData = response.data
                                    }.onFailure { error ->
                                        errorMessage = error.message
                                    }
                                }
                            },
                            colors = ButtonDefaults.buttonColors(
                                containerColor = DarkPrimary
                            )
                        ) {
                            Text("Thử lại")
                        }
                    }
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
                            errorMessage = error
                        }
                    )
                } else {
                    ProfileContent(
                        profileData = profileData!!,
                        onChangePasswordClick = onChangePasswordClick,
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
                // Avatar
                Box(
                    modifier = Modifier
                        .size(100.dp)
                        .clip(CircleShape)
                        .background(DarkPrimary),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = Icons.Default.Person,
                        contentDescription = "Avatar",
                        tint = DarkOnPrimary,
                        modifier = Modifier.size(60.dp)
                    )
                }
                
                Spacer(modifier = Modifier.height(16.dp))
                
                // Tên
                Text(
                    text = profileData.fullName?.takeIf { it.isNotBlank() } ?: "Người dùng",
                    fontSize = 24.sp,
                    fontWeight = FontWeight.Bold,
                    color = DarkOnSurface
                )
                
                // Email
                if (!profileData.email.isNullOrBlank()) {
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = profileData.email,
                        fontSize = 14.sp,
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
                        DarkError.copy(alpha = 0.2f)
                ) {
                    Text(
                        text = if (profileData.isActive == true) "✓ Đã xác thực" else "⚠ Chưa xác thực",
                        color = if (profileData.isActive == true) DarkPrimary else DarkError,
                        fontSize = 12.sp,
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp)
                    )
                }
            }
        }
        
        Spacer(modifier = Modifier.height(16.dp))
        
        // Thông tin cá nhân
        Text(
            text = "Thông tin cá nhân",
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
                    icon = Icons.Default.Phone,
                    label = "Số điện thoại",
                    value = profileData.phoneNumber ?: "Chưa cập nhật"
                )
                
                Divider(
                    color = DarkOnSurface.copy(alpha = 0.1f),
                    modifier = Modifier.padding(vertical = 12.dp)
                )
                
                ProfileInfoRow(
                    icon = Icons.Default.DateRange,
                    label = "Ngày sinh",
                    value = profileData.dateOfBirth ?: "Chưa cập nhật"
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
                    value = profileData.address ?: "Chưa cập nhật"
                )
            }
        }
        
        Spacer(modifier = Modifier.height(16.dp))
        
        // Thông tin sức khỏe
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
                    value = profileData.bloodType?.let { 
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
                        fontSize = 16.sp,
                        color = DarkOnSurface,
                        fontWeight = FontWeight.Medium
                    )
                    Text(
                        text = "Thay đổi mật khẩu của bạn",
                        fontSize = 12.sp,
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
                fontSize = 12.sp,
                color = DarkOnSurface.copy(alpha = 0.6f)
            )
            Spacer(modifier = Modifier.height(2.dp))
            Text(
                text = value,
                fontSize = 16.sp,
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
    var bloodType by remember { mutableStateOf(profileData.bloodType ?: "A_POSITIVE") }
    var height by remember { mutableStateOf(profileData.height?.toString() ?: "") }
    var weight by remember { mutableStateOf(profileData.weight?.toString() ?: "") }
    
    var isUpdating by remember { mutableStateOf(false) }
    var expandedGender by remember { mutableStateOf(false) }
    var expandedBloodType by remember { mutableStateOf(false) }

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
                
                // Ngày sinh
                OutlinedTextField(
                    value = dateOfBirth,
                    onValueChange = { dateOfBirth = it },
                    label = { Text("Ngày sinh (yyyy-MM-dd)") },
                    leadingIcon = {
                        Icon(Icons.Default.DateRange, "Ngày sinh", tint = DarkPrimary)
                    },
                    placeholder = { Text("2000-01-01") },
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
        
        // Thông tin sức khỏe
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
        
        // Nút cập nhật
        Button(
            onClick = {
                coroutineScope.launch {
                    isUpdating = true
                    
                    val result = userService.updateUserProfile(
                        accessToken = accessToken,
                        fullName = fullName.takeIf { it.isNotBlank() },
                        dateOfBirth = dateOfBirth.takeIf { it.isNotBlank() },
                        gender = gender,
                        phoneNumber = phoneNumber.takeIf { it.isNotBlank() },
                        address = address.takeIf { it.isNotBlank() },
                        bloodType = bloodType,
                        height = height.toDoubleOrNull(),
                        weight = weight.toDoubleOrNull()
                    )
                    
                    isUpdating = false
                    
                    result.onSuccess { response ->
                        response.data?.let { onUpdateSuccess(it) }
                    }.onFailure { error ->
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
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold
                )
            }
        }
        
        Spacer(modifier = Modifier.height(16.dp))
    }
}
