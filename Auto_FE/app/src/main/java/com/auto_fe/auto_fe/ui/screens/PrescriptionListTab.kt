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
fun PrescriptionListTab(
    accessToken: String,
    onPrescriptionClick: (Long) -> Unit,
    onCreateClick: () -> Unit = {}
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

    Box(modifier = Modifier.fillMaxSize()) {
        Column(modifier = Modifier.fillMaxSize()) {
            // Subtitle
            Text(
                text = "Quản lý lịch nhắc uống thuốc theo đơn",
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

        // Floating Action Button
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
