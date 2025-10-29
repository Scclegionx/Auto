package com.auto_fe.auto_fe.ui.screens

import androidx.compose.foundation.background
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
    onBackClick: () -> Unit = {}
) {
    val userService = remember { UserService() }
    val coroutineScope = rememberCoroutineScope()
    
    var profileData by remember { mutableStateOf<UserService.ProfileData?>(null) }
    var isLoading by remember { mutableStateOf(true) }
    var errorMessage by remember { mutableStateOf<String?>(null) }

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

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Thông tin cá nhân") },
                navigationIcon = {
                    IconButton(onClick = onBackClick) {
                        Icon(Icons.Default.ArrowBack, "Quay lại")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = DarkSurface,
                    titleContentColor = DarkOnSurface,
                    navigationIconContentColor = DarkOnSurface
                )
            )
        },
        containerColor = DarkBackground
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
                            color = AIError,
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
                ProfileContent(
                    profileData = profileData!!,
                    modifier = Modifier.padding(paddingValues)
                )
            }
        }
    }
}

@Composable
fun ProfileContent(
    profileData: UserService.ProfileData,
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
                        AIError.copy(alpha = 0.2f)
                ) {
                    Text(
                        text = if (profileData.isActive == true) "✓ Đã xác thực" else "⚠ Chưa xác thực",
                        color = if (profileData.isActive == true) AIPrimarySoft else AIError,
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
