package com.auto_fe.auto_fe.ui.screens

import android.widget.Toast
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.auto_fe.auto_fe.ui.service.PrescriptionService
import com.auto_fe.auto_fe.ui.theme.*
import kotlinx.coroutines.launch

@Composable
fun PrescriptionListScreen(
    accessToken: String,
    onPrescriptionClick: (Long) -> Unit,
    onCreateClick: () -> Unit = {},
    onLogout: () -> Unit = {}
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val prescriptionService = remember { PrescriptionService() }
    
    var prescriptions by remember { mutableStateOf<List<PrescriptionService.Prescription>>(emptyList()) }
    var isLoading by remember { mutableStateOf(true) }
    var errorMessage by remember { mutableStateOf<String?>(null) }

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
                        DarkGradientStart,
                        DarkGradientEnd
                    )
                )
            )
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp)
        ) {
            // Header với Logout button
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "💊 Đơn thuốc của tôi",
                    fontSize = 28.sp,
                    fontWeight = FontWeight.Bold,
                    color = DarkPrimary
                )
                
                TextButton(onClick = onLogout) {
                    Text(
                        text = "🚪 Đăng xuất",
                        color = DarkError,
                        fontSize = 14.sp
                    )
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = "Quản lý lịch nhắc uống thuốc",
                fontSize = 14.sp,
                color = DarkOnSurface.copy(alpha = 0.7f)
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
                                color = DarkError,
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
                                fontSize = 18.sp,
                                fontWeight = FontWeight.Bold,
                                color = DarkOnSurface
                            )
                            Spacer(modifier = Modifier.height(8.dp))
                            Text(
                                text = "Thêm đơn thuốc đầu tiên để bắt đầu",
                                fontSize = 14.sp,
                                color = DarkOnSurface.copy(alpha = 0.7f),
                                textAlign = TextAlign.Center
                            )
                        }
                    }
                }
                else -> {
                    LazyColumn(
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        items(prescriptions) { prescription ->
                            PrescriptionCard(
                                prescription = prescription,
                                onClick = { onPrescriptionClick(prescription.id) }
                            )
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
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold,
                        color = DarkOnSurface
                    )
                    
                    if (!prescription.description.isNullOrBlank()) {
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = prescription.description,
                            fontSize = 14.sp,
                            color = DarkOnSurface.copy(alpha = 0.7f),
                            maxLines = 2,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                }
                
                // Status badge
                Surface(
                    color = if (prescription.isActive) SuccessColor.copy(alpha = 0.2f) else DarkOnSurface.copy(alpha = 0.1f),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Text(
                        text = if (prescription.isActive) "✓ Đang dùng" else "⏸ Tạm ngưng",
                        fontSize = 12.sp,
                        color = if (prescription.isActive) SuccessColor else DarkOnSurface.copy(alpha = 0.5f),
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
                        fontSize = 16.sp
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "${prescription.medicationReminders.size} loại thuốc",
                        fontSize = 14.sp,
                        color = DarkOnSurface.copy(alpha = 0.8f)
                    )
                }

                Text(
                    text = "Xem chi tiết →",
                    fontSize = 13.sp,
                    color = DarkPrimary,
                    fontWeight = FontWeight.Medium
                )
            }
        }
    }
}
