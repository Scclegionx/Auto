package com.auto_fe.auto_fe.ui.screens

import android.widget.Toast
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Person
import androidx.compose.material.icons.filled.ExitToApp
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.Notifications
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.auto_fe.auto_fe.ui.service.PrescriptionService
import com.auto_fe.auto_fe.ui.theme.*
import com.auto_fe.auto_fe.ui.theme.AppTextSize
import kotlinx.coroutines.launch

@Composable
fun PrescriptionListScreen(
    accessToken: String,
    onPrescriptionClick: (Long) -> Unit,
    onCreateClick: () -> Unit = {},
    onLogout: () -> Unit = {},
    onProfileClick: () -> Unit = {}, // Thêm callback profile
    onNotificationHistoryClick: () -> Unit = {}, // Thêm callback notification history
    userName: String = "User", // Thêm tên user
    userEmail: String = "" // Thêm email user
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val prescriptionService = remember { PrescriptionService() }
    
    var prescriptions by remember { mutableStateOf<List<PrescriptionService.Prescription>>(emptyList()) }
    var isLoading by remember { mutableStateOf(true) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var showMenu by remember { mutableStateOf(false) } // State cho dropdown menu

    // Load prescriptions when screen opens
    LaunchedEffect(Unit) {
        scope.launch {
            isLoading = true
            val result = prescriptionService.getAllPrescriptions(accessToken)
            result.fold(
                onSuccess = { response ->
                    prescriptions = response.data ?: emptyList()
                    isLoading = false
                },
                onFailure = { error ->
                    errorMessage = error.message
                    isLoading = false
                    Toast.makeText(context, "❌ ${error.message}", Toast.LENGTH_LONG).show()
                }
            )
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(
                brush = Brush.verticalGradient(
                    colors = listOf(
                        AIBackgroundDeep,
                        AIBackgroundSoft
                    )
                )
            )
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
        ) {
            // Header Card với Avatar và Dropdown Menu
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
                colors = CardDefaults.cardColors(
                    containerColor = DarkSurface.copy(alpha = 0.9f)
                ),
                shape = RoundedCornerShape(16.dp),
                elevation = CardDefaults.cardElevation(defaultElevation = 8.dp)
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    // Thông tin user và tiêu đề
                    Column(
                        modifier = Modifier.weight(1f)
                    ) {
                        Text(
                            text = "Đơn thuốc của tôi",
                            fontSize = AppTextSize.titleMedium,
                            fontWeight = FontWeight.Bold,
                            color = DarkOnSurface
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = "Xin chào, ${userName.takeIf { it.isNotBlank() } ?: "Người dùng"}",
                            fontSize = AppTextSize.bodyMedium,
                            color = DarkOnSurface.copy(alpha = 0.7f)
                        )
                    }
                    
                    // Avatar với Dropdown Menu
                    Box {
                        Box(
                            modifier = Modifier
                                .size(48.dp)
                                .clip(CircleShape)
                                .background(DarkPrimary)
                                .clickable { showMenu = !showMenu },
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(
                                imageVector = Icons.Default.Person,
                                contentDescription = "User Menu",
                                tint = DarkOnPrimary,
                                modifier = Modifier.size(28.dp)
                            )
                        }
                        
                        // Dropdown Menu
                        DropdownMenu(
                            expanded = showMenu,
                            onDismissRequest = { showMenu = false },
                            modifier = Modifier
                                .background(DarkSurface)
                                .width(200.dp)
                        ) {
                            // User Info
                            Column(
                                modifier = Modifier.padding(16.dp, 12.dp)
                            ) {
                                Text(
                                    text = userName.takeIf { it.isNotBlank() } ?: "Người dùng",
                                    fontSize = AppTextSize.bodyMedium,
                                    fontWeight = FontWeight.Bold,
                                    color = DarkOnSurface
                                )
                                if (userEmail.isNotEmpty()) {
                                    Text(
                                        text = userEmail,
                                        fontSize = AppTextSize.bodySmall,
                                        color = DarkOnSurface.copy(alpha = 0.6f)
                                    )
                                }
                            }
                            
                            Divider(color = DarkOnSurface.copy(alpha = 0.2f))
                            
                            // Menu Items - Có thể mở rộng thêm
                            DropdownMenuItem(
                                text = {
                                    Row(
                                        verticalAlignment = Alignment.CenterVertically,
                                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                                    ) {
                                        Icon(
                                            imageVector = Icons.Default.Person,
                                            contentDescription = null,
                                            tint = DarkOnSurface,
                                            modifier = Modifier.size(20.dp)
                                        )
                                        Text("Hồ sơ", color = DarkOnSurface)
                                    }
                                },
                                onClick = {
                                    showMenu = false
                                    onProfileClick()
                                }
                            )
                            
                            DropdownMenuItem(
                                text = {
                                    Row(
                                        verticalAlignment = Alignment.CenterVertically,
                                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                                    ) {
                                        Icon(
                                            imageVector = Icons.Default.Notifications,
                                            contentDescription = null,
                                            tint = DarkOnSurface,
                                            modifier = Modifier.size(20.dp)
                                        )
                                        Text("Lịch sử thông báo", color = DarkOnSurface)
                                    }
                                },
                                onClick = {
                                    showMenu = false
                                    onNotificationHistoryClick()
                                }
                            )
                            
                            Divider(color = DarkOnSurface.copy(alpha = 0.2f))
                            
                            DropdownMenuItem(
                                text = {
                                    Row(
                                        verticalAlignment = Alignment.CenterVertically,
                                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                                    ) {
                                        Icon(
                                            imageVector = Icons.Default.Settings,
                                            contentDescription = null,
                                            tint = DarkOnSurface,
                                            modifier = Modifier.size(20.dp)
                                        )
                                        Text("Cài đặt", color = DarkOnSurface)
                                    }
                                },
                                onClick = {
                                    showMenu = false
                                    // TODO: Navigate to settings
                                    Toast.makeText(context, "Chức năng sắp ra mắt", Toast.LENGTH_SHORT).show()
                                }
                            )
                            
                            Divider(color = DarkOnSurface.copy(alpha = 0.2f))
                            
                            DropdownMenuItem(
                                text = {
                                    Row(
                                        verticalAlignment = Alignment.CenterVertically,
                                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                                    ) {
                                        Icon(
                                            imageVector = Icons.Default.ExitToApp,
                                            contentDescription = null,
                                            tint = AIError,
                                            modifier = Modifier.size(20.dp)
                                        )
                                        Text("Đăng xuất", color = AIError)
                                    }
                                },
                                onClick = {
                                    showMenu = false
                                    onLogout()
                                }
                            )
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Subtitle
            Text(
                text = "Quản lý lịch nhắc uống thuốc",
                fontSize = AppTextSize.bodyMedium,
                color = DarkOnSurface.copy(alpha = 0.7f),
                modifier = Modifier.padding(horizontal = 16.dp)
            )

            Spacer(modifier = Modifier.height(16.dp))

            // Content
            when {
                isLoading -> {
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center
                    ) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            CircularProgressIndicator(color = DarkPrimary)
                            Spacer(modifier = Modifier.height(16.dp))
                            Text(
                                text = "Đang tải đơn thuốc...",
                                color = DarkOnSurface.copy(alpha = 0.7f)
                            )
                        }
                    }
                }
                errorMessage != null -> {
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center
                    ) {
                        Column(
                            horizontalAlignment = Alignment.CenterHorizontally,
                            modifier = Modifier.padding(32.dp)
                        ) {
                            Text(
                                text = "❌",
                                fontSize = 48.sp
                            )
                            Spacer(modifier = Modifier.height(16.dp))
                            Text(
                                text = errorMessage ?: "Có lỗi xảy ra",
                                color = AIError,
                                textAlign = TextAlign.Center
                            )
                            Spacer(modifier = Modifier.height(16.dp))
                            Button(
                                onClick = {
                                    scope.launch {
                                        isLoading = true
                                        errorMessage = null
                                        val result = prescriptionService.getAllPrescriptions(accessToken)
                                        result.fold(
                                            onSuccess = { response ->
                                                prescriptions = response.data ?: emptyList()
                                                isLoading = false
                                            },
                                            onFailure = { error ->
                                                errorMessage = error.message
                                                isLoading = false
                                            }
                                        )
                                    }
                                },
                                colors = ButtonDefaults.buttonColors(containerColor = DarkPrimary)
                            ) {
                                Text("🔄 Thử lại")
                            }
                        }
                    }
                }
                prescriptions.isEmpty() -> {
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center
                    ) {
                        Column(
                            horizontalAlignment = Alignment.CenterHorizontally,
                            modifier = Modifier.padding(32.dp)
                        ) {
                            Text(
                                text = "📋",
                                fontSize = 64.sp
                            )
                            Spacer(modifier = Modifier.height(16.dp))
                            Text(
                                text = "Chưa có đơn thuốc nào",
                                fontSize = AppTextSize.titleSmall,
                                fontWeight = FontWeight.Bold,
                                color = DarkOnSurface
                            )
                            Spacer(modifier = Modifier.height(8.dp))
                            Text(
                                text = "Thêm đơn thuốc đầu tiên để bắt đầu",
                                fontSize = AppTextSize.bodyMedium,
                                color = DarkOnSurface.copy(alpha = 0.7f),
                                textAlign = TextAlign.Center
                            )
                        }
                    }
                }
                else -> {
                    LazyColumn(
                        modifier = Modifier.padding(horizontal = 16.dp),
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        items(prescriptions) { prescription ->
                            PrescriptionCard(
                                prescription = prescription,
                                onClick = { onPrescriptionClick(prescription.id) }
                            )
                        }
                        
                        // Thêm spacing cuối để FAB không che content
                        item {
                            Spacer(modifier = Modifier.height(80.dp))
                        }
                    }
                }
            }
        }

        // Floating Action Button để thêm đơn thuốc mới
        FloatingActionButton(
            onClick = onCreateClick,
            containerColor = DarkPrimary,
            modifier = Modifier
                .align(Alignment.BottomEnd)
                .padding(24.dp)
        ) {
            Icon(
                imageVector = Icons.Default.Add,
                contentDescription = "Thêm đơn thuốc",
                tint = DarkOnPrimary
            )
        }
    }
}

@Composable
fun PrescriptionCard(
    prescription: PrescriptionService.Prescription,
    onClick: () -> Unit
) {
    Card(
        colors = CardDefaults.cardColors(
            containerColor = DarkSurface.copy(alpha = 0.9f)
        ),
        shape = RoundedCornerShape(16.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick)
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.Top
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = prescription.name,
                        fontSize = AppTextSize.titleSmall,
                        fontWeight = FontWeight.Bold,
                        color = DarkOnSurface
                    )
                    
                    if (!prescription.description.isNullOrBlank()) {
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = prescription.description,
                            fontSize = AppTextSize.bodyMedium,
                            color = DarkOnSurface.copy(alpha = 0.7f),
                            maxLines = 2,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                }
                
                // Status badge
                Surface(
                    color = if (prescription.isActive) AISuccess.copy(alpha = 0.2f) else DarkOnSurface.copy(alpha = 0.1f),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Text(
                        text = if (prescription.isActive) "✓ Đang dùng" else "⏸ Tạm ngưng",
                        fontSize = 11.sp,
                        color = if (prescription.isActive) AISuccess else DarkOnSurface.copy(alpha = 0.5f),
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                        fontWeight = FontWeight.Medium
                    )
                }
            }

            Spacer(modifier = Modifier.height(12.dp))

            Divider(color = DarkOnSurface.copy(alpha = 0.1f))

            Spacer(modifier = Modifier.height(12.dp))

            // Medication count và info
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = "💊",
                        fontSize = AppTextSize.bodyMedium
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "${prescription.medications?.size ?: prescription.medicationReminders?.size ?: 0} loại thuốc",
                        fontSize = AppTextSize.bodyMedium,
                        color = DarkOnSurface.copy(alpha = 0.8f)
                    )
                }

                Text(
                    text = "Xem chi tiết →",
                    fontSize = AppTextSize.bodySmall,
                    color = DarkPrimary,
                    fontWeight = FontWeight.Medium
                )
            }
            
            // Nút xem ảnh (nếu có)
            if (!prescription.imageUrl.isNullOrBlank()) {
                Spacer(modifier = Modifier.height(12.dp))
                
                var showImageDialog by remember { mutableStateOf(false) }
                
                OutlinedButton(
                    onClick = { 
                        showImageDialog = true
                    },
                    modifier = Modifier.fillMaxWidth(),
                    colors = ButtonDefaults.outlinedButtonColors(
                        contentColor = DarkPrimary
                    ),
                    border = androidx.compose.foundation.BorderStroke(1.dp, DarkPrimary.copy(alpha = 0.5f)),
                    contentPadding = PaddingValues(vertical = 8.dp)
                ) {
                    Text(text = "📷", fontSize = 16.sp)
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        text = "Xem ảnh đơn thuốc",
                        fontSize = 13.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
                
                if (showImageDialog) {
                    ZoomableImageDialog(
                        imageUrl = prescription.imageUrl,
                        onDismiss = { showImageDialog = false }
                    )
                }
            }
        }
    }
}
