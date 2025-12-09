package com.auto_fe.auto_fe.ui.screens

import android.widget.Toast
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
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
import com.auto_fe.auto_fe.ui.service.StandaloneMedicationService
import com.auto_fe.auto_fe.ui.theme.*
import com.auto_fe.auto_fe.utils.SessionManager
import com.auto_fe.auto_fe.ui.utils.formatDaysOfWeek
import kotlinx.coroutines.launch

@Composable
fun StandaloneMedicationTab(
    accessToken: String,
    onCreateClick: () -> Unit = {},
    elderUserId: Long? = null,  //  Add elderUserId parameter
    elderUserName: String? = null  // Add elderUserName parameter
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val medicationService = remember { StandaloneMedicationService() }
    val sessionManager = remember { SessionManager(context) }
    
    //  Nếu có elderUserId (Supervisor mode) thì dùng elderUserId, không thì dùng userId của chính mình
    val targetUserId = elderUserId ?: (sessionManager.getUserId() ?: 0L)
    
    var medications by remember { mutableStateOf<List<StandaloneMedicationService.Medication>>(emptyList()) }
    var isLoading by remember { mutableStateOf(true) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var showDeleteDialog by remember { mutableStateOf(false) }
    var selectedMedication by remember { mutableStateOf<StandaloneMedicationService.Medication?>(null) }

    fun loadMedications() {
        scope.launch {
            isLoading = true
            errorMessage = null
            android.util.Log.d("StandaloneMedicationTab", "Loading medications for userId: $targetUserId (elderMode: ${elderUserId != null})")
            val result = medicationService.getAll(accessToken, targetUserId)
            result.fold(
                onSuccess = { response ->
                    medications = response.data ?: emptyList()
                    isLoading = false
                },
                onFailure = { error ->
                    errorMessage = error.message
                    isLoading = false
                    Toast.makeText(context, "${error.message}", Toast.LENGTH_LONG).show()
                }
            )
        }
    }

    // Load medications when screen opens or elderUserId changes
    LaunchedEffect(elderUserId) {  //  Reload when elderUserId changes
        loadMedications()
    }

    Box(modifier = Modifier.fillMaxSize()) {
        Column(modifier = Modifier.fillMaxSize()) {
            // Subtitle
            Text(
                text = "Quản lý thuốc không có đơn bác sĩ",
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
                                text = "Đang tải danh sách thuốc...",
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
                            Text(text = "❌", fontSize = 48.sp)
                            Spacer(modifier = Modifier.height(16.dp))
                            Text(
                                text = errorMessage ?: "Có lỗi xảy ra",
                                color = AIError,
                                textAlign = TextAlign.Center
                            )
                            Spacer(modifier = Modifier.height(16.dp))
                            Button(
                                onClick = { loadMedications() },
                                colors = ButtonDefaults.buttonColors(containerColor = DarkPrimary)
                            ) {
                                Text("🔄 Thử lại")
                            }
                        }
                    }
                }
                medications.isEmpty() -> {
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center
                    ) {
                        Column(
                            horizontalAlignment = Alignment.CenterHorizontally,
                            modifier = Modifier.padding(32.dp)
                        ) {
                            Text(text = "💊", fontSize = 64.sp)
                            Spacer(modifier = Modifier.height(16.dp))
                            Text(
                                text = "Chưa có thuốc nào",
                                fontSize = AppTextSize.titleSmall,
                                fontWeight = FontWeight.Bold,
                                color = DarkOnSurface
                            )
                            Spacer(modifier = Modifier.height(8.dp))
                            Text(
                                text = "Thêm thuốc uống hàng ngày của bạn",
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
                        items(medications) { medication ->
                            StandaloneMedicationCard(
                                medication = medication,
                                onToggleActive = {
                                    scope.launch {
                                        val result = medicationService.toggleActive(accessToken, medication.id)
                                        result.onSuccess {
                                            Toast.makeText(
                                                context,
                                                if (medication.isActive) "⏸ Đã tạm ngưng" else "✓ Đã kích hoạt",
                                                Toast.LENGTH_SHORT
                                            ).show()
                                            loadMedications()
                                        }.onFailure { error ->
                                            Toast.makeText(context, "${error.message}", Toast.LENGTH_SHORT).show()
                                        }
                                    }
                                },
                                onDelete = {
                                    selectedMedication = medication
                                    showDeleteDialog = true
                                }
                            )
                        }
                        
                        // Spacing cuối
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
                contentDescription = "Thêm thuốc",
                tint = DarkOnPrimary
            )
        }
    }

    // Delete Confirmation Dialog
    if (showDeleteDialog && selectedMedication != null) {
        AlertDialog(
            onDismissRequest = {
                showDeleteDialog = false
                selectedMedication = null
            },
            shape = RoundedCornerShape(20.dp),
            containerColor = DarkSurface,
            icon = {
                Icon(
                    imageVector = Icons.Default.Delete,
                    contentDescription = null,
                    tint = AIError,
                    modifier = Modifier.size(48.dp)
                )
            },
            title = {
                Text(
                    "Xóa thuốc?",
                    fontWeight = FontWeight.Bold,
                    color = DarkOnSurface
                )
            },
            text = {
                Text(
                    "Bạn có chắc muốn xóa \"${selectedMedication!!.name}\"? Hành động này không thể hoàn tác.",
                    color = DarkOnSurface.copy(alpha = 0.8f)
                )
            },
            confirmButton = {
                Button(
                    onClick = {
                        scope.launch {
                            val result = medicationService.delete(accessToken, selectedMedication!!.id)
                            result.onSuccess {
                                Toast.makeText(context, "Đã xóa thuốc", Toast.LENGTH_SHORT).show()
                                showDeleteDialog = false
                                selectedMedication = null
                                loadMedications()
                            }.onFailure { error ->
                                Toast.makeText(context, "${error.message}", Toast.LENGTH_SHORT).show()
                            }
                        }
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = AIError)
                ) {
                    Text("Xóa", fontWeight = FontWeight.Bold)
                }
            },
            dismissButton = {
                TextButton(onClick = {
                    showDeleteDialog = false
                    selectedMedication = null
                }) {
                    Text("Hủy", color = DarkOnSurface)
                }
            }
        )
    }
}

@Composable
fun StandaloneMedicationCard(
    medication: StandaloneMedicationService.Medication,
    onToggleActive: () -> Unit,
    onDelete: () -> Unit
) {
    Card(
        colors = CardDefaults.cardColors(
            containerColor = DarkSurface.copy(alpha = 0.9f)
        ),
        shape = RoundedCornerShape(16.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            // Header
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.Top
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = medication.name,
                        fontSize = AppTextSize.titleSmall,
                        fontWeight = FontWeight.Bold,
                        color = DarkOnSurface
                    )
                    
                    if (!medication.dosage.isNullOrBlank() && !medication.unit.isNullOrBlank()) {
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = "${medication.dosage} ${medication.unit}",
                            fontSize = AppTextSize.bodyMedium,
                            color = DarkPrimary,
                            fontWeight = FontWeight.Medium
                        )
                    }
                }
                
                // Status badge
                Surface(
                    color = if (medication.isActive) AISuccess.copy(alpha = 0.2f) else DarkOnSurface.copy(alpha = 0.1f),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Text(
                        text = if (medication.isActive) "✓ Đang dùng" else "⏸ Tạm ngưng",
                        fontSize = 11.sp,
                        color = if (medication.isActive) AISuccess else DarkOnSurface.copy(alpha = 0.5f),
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                        fontWeight = FontWeight.Medium
                    )
                }
            }

            Spacer(modifier = Modifier.height(12.dp))
            Divider(color = DarkOnSurface.copy(alpha = 0.1f))
            Spacer(modifier = Modifier.height(12.dp))

            // Times
            if (medication.times.isNotEmpty()) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Icon(
                        imageVector = Icons.Default.Notifications,
                        contentDescription = null,
                        tint = DarkPrimary,
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = medication.times.joinToString(" • "),
                        fontSize = AppTextSize.bodyMedium,
                        color = DarkOnSurface.copy(alpha = 0.8f)
                    )
                }
            }

            // Days of Week
            if (!medication.daysOfWeek.isNullOrBlank()) {
                Spacer(modifier = Modifier.height(8.dp))
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Icon(
                        imageVector = Icons.Default.DateRange,
                        contentDescription = null,
                        tint = DarkOnSurface.copy(alpha = 0.6f),
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = formatDaysOfWeek(medication.daysOfWeek),
                        fontSize = AppTextSize.bodySmall,
                        color = DarkOnSurface.copy(alpha = 0.7f)
                    )
                }
            }

            // Frequency
            if (!medication.frequency.isNullOrBlank()) {
                Spacer(modifier = Modifier.height(8.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        imageVector = Icons.Default.Info,
                        contentDescription = null,
                        tint = DarkOnSurface.copy(alpha = 0.6f),
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = medication.frequency,
                        fontSize = AppTextSize.bodySmall,
                        color = DarkOnSurface.copy(alpha = 0.7f)
                    )
                }
            }

            // Description/Notes
            if (!medication.description.isNullOrBlank()) {
                Spacer(modifier = Modifier.height(8.dp))
                Surface(
                    color = DarkOnSurface.copy(alpha = 0.05f),
                    shape = RoundedCornerShape(8.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        text = medication.description,
                        fontSize = AppTextSize.bodySmall,
                        color = DarkOnSurface.copy(alpha = 0.7f),
                        modifier = Modifier.padding(8.dp),
                        fontStyle = androidx.compose.ui.text.font.FontStyle.Italic
                    )
                }
            }

            Spacer(modifier = Modifier.height(12.dp))

            // Action Buttons
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                OutlinedButton(
                    onClick = onToggleActive,
                    modifier = Modifier.weight(1f),
                    colors = ButtonDefaults.outlinedButtonColors(
                        contentColor = if (medication.isActive) DarkOnSurface else AISuccess
                    ),
                    border = androidx.compose.foundation.BorderStroke(
                        1.dp,
                        if (medication.isActive) DarkOnSurface.copy(alpha = 0.3f) else AISuccess.copy(alpha = 0.5f)
                    )
                ) {
                    Icon(
                        imageVector = if (medication.isActive) Icons.Default.Close else Icons.Default.Check,
                        contentDescription = null,
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        text = if (medication.isActive) "Tạm ngưng" else "Kích hoạt",
                        fontSize = 13.sp,
                        fontWeight = FontWeight.SemiBold
                    )
                }

                OutlinedButton(
                    onClick = onDelete,
                    colors = ButtonDefaults.outlinedButtonColors(
                        contentColor = AIError
                    ),
                    border = androidx.compose.foundation.BorderStroke(1.dp, AIError.copy(alpha = 0.5f))
                ) {
                    Icon(
                        imageVector = Icons.Default.Delete,
                        contentDescription = "Xóa",
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text("Xóa", fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
                }
            }
        }
    }
}
