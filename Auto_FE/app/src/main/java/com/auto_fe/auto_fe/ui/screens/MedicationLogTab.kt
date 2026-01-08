package com.auto_fe.auto_fe.ui.screens

import androidx.compose.foundation.background
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.auto_fe.auto_fe.models.MedicationLog
import com.auto_fe.auto_fe.service.be.MedicationLogService
import com.auto_fe.auto_fe.ui.theme.*
import kotlinx.coroutines.launch
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MedicationLogTab(
    accessToken: String,
    currentUserId: Long? = null,  // Add current logged-in user ID
    elderUserId: Long? = null,
    elderUserName: String? = null
) {
    var isLoading by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var medicationLogs by remember { mutableStateOf<List<MedicationLog>>(emptyList()) }
    var statistics by remember { mutableStateOf<Map<String, Any>?>(null) }
    var selectedDays by remember { mutableStateOf(7) }
    
    val coroutineScope = rememberCoroutineScope()
    val medicationLogService = remember { MedicationLogService() }

    // Determine which userId to use
    val targetUserId = elderUserId ?: currentUserId

    // Load medication logs
    LaunchedEffect(selectedDays, targetUserId) {
        isLoading = true
        errorMessage = null
        
        coroutineScope.launch {
            try {
                if (targetUserId == null) {
                    errorMessage = "User ID not available"
                    isLoading = false
                    return@launch
                }
                
                val result = medicationLogService.getElderMedicationHistory(
                    accessToken = accessToken,
                    elderId = targetUserId,
                    days = selectedDays
                )
                
                result.fold(
                    onSuccess = { response ->
                        medicationLogs = response.logs
                        statistics = response.statistics
                    },
                    onFailure = { error ->
                        errorMessage = error.message ?: "Không thể tải lịch sử uống thuốc"
                    }
                )
            } finally {
                isLoading = false
            }
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
    ) {
        // Header and Statistics - Scrollable part
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Header Card
            item {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(containerColor = DarkSurface),
                    shape = RoundedCornerShape(16.dp)
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp)
                    ) {
                        Text(
                            text = if (elderUserId != null) {
                                "📊 Lịch sử uống thuốc - $elderUserName"
                            } else {
                                "📊 Lịch sử uống thuốc của bạn"
                            },
                            style = MaterialTheme.typography.titleLarge,
                            fontWeight = FontWeight.Bold,
                            color = DarkOnSurface
                        )
                        
                        Spacer(modifier = Modifier.height(16.dp))
                        
                        // Time filter
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            listOf(7, 14, 30).forEach { days ->
                                FilterChip(
                                    selected = selectedDays == days,
                                    onClick = { selectedDays = days },
                                    label = { Text("$days ngày") },
                                    colors = FilterChipDefaults.filterChipColors(
                                        selectedContainerColor = DarkPrimary,
                                        selectedLabelColor = Color.White
                                    )
                                )
                            }
                        }
                        
                        // Statistics
                        statistics?.let { stats ->
                            Spacer(modifier = Modifier.height(16.dp))
                            
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                StatCard(
                                    modifier = Modifier.weight(1f),
                                    icon = Icons.Default.Assignment,
                                    label = "Tổng số",
                                    value = "${stats["total"] ?: 0}",
                                    color = DarkPrimary
                                )
                                StatCard(
                                    modifier = Modifier.weight(1f),
                                    icon = Icons.Default.CheckCircle,
                                    label = "Đã uống",
                                    value = "${stats["taken"] ?: 0}",
                                    color = Color(0xFF4CAF50)
                                )
                            }
                            
                            Spacer(modifier = Modifier.height(8.dp))
                            
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                StatCard(
                                    modifier = Modifier.weight(1f),
                                    icon = Icons.Default.Schedule,
                                    label = "Đúng giờ",
                                    value = "${stats["onTime"] ?: 0}",
                                    color = Color(0xFF2196F3)
                                )
                                StatCard(
                                    modifier = Modifier.weight(1f),
                                    icon = Icons.Default.Cancel,
                                    label = "Bỏ lỡ",
                                    value = "${stats["missed"] ?: 0}",
                                    color = Color(0xFFF44336)
                                )
                            }
                            
                            Spacer(modifier = Modifier.height(16.dp))
                            
                            // Adherence rate
                            val adherenceRate = (stats["adherenceRate"] as? Number)?.toDouble() ?: 0.0
                            LinearProgressIndicator(
                                progress = (adherenceRate / 100).toFloat(),
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(8.dp),
                                color = when {
                                    adherenceRate >= 80 -> Color(0xFF4CAF50)
                                    adherenceRate >= 60 -> Color(0xFFFFC107)
                                    else -> Color(0xFFF44336)
                                },
                                trackColor = DarkOnSurface.copy(alpha = 0.2f)
                            )
                            
                            Text(
                                text = "Tỷ lệ tuân thủ: ${String.format("%.1f", adherenceRate)}%",
                                style = MaterialTheme.typography.bodyMedium,
                                color = DarkOnSurface.copy(alpha = 0.7f),
                                modifier = Modifier.padding(top = 4.dp)
                            )
                        }
                    }
                }
            }
            
            // Loading or Error state
            when {
                isLoading -> {
                    item {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(32.dp),
                            contentAlignment = Alignment.Center
                        ) {
                            CircularProgressIndicator(color = DarkPrimary)
                        }
                    }
                }
                errorMessage != null -> {
                    item {
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            colors = CardDefaults.cardColors(containerColor = Color(0xFFF44336).copy(alpha = 0.1f))
                        ) {
                            Text(
                                text = errorMessage ?: "",
                                color = Color(0xFFF44336),
                                modifier = Modifier.padding(16.dp)
                            )
                        }
                    }
                }
                medicationLogs.isEmpty() -> {
                    item {
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            colors = CardDefaults.cardColors(containerColor = DarkSurface)
                        ) {
                            Column(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(32.dp),
                                horizontalAlignment = Alignment.CenterHorizontally
                            ) {
                                Icon(
                                    imageVector = Icons.Default.Assignment,
                                    contentDescription = null,
                                    tint = DarkOnSurface.copy(alpha = 0.3f),
                                    modifier = Modifier.size(64.dp)
                                )
                                Spacer(modifier = Modifier.height(16.dp))
                                Text(
                                    text = "Chưa có lịch sử uống thuốc",
                                    style = MaterialTheme.typography.bodyLarge,
                                    color = DarkOnSurface.copy(alpha = 0.6f)
                                )
                            }
                        }
                    }
                }
                else -> {
                    // Medication logs list
                    items(medicationLogs) { log ->
                        MedicationLogCard(
                            log = log,
                            accessToken = accessToken,
                            onConfirmTaken = { logId ->
                                coroutineScope.launch {
                                    try {
                                        val result = medicationLogService.confirmMedicationTaken(
                                            accessToken = accessToken,
                                            logId = logId,
                                            userId = targetUserId!!
                                        )
                                        result.fold(
                                            onSuccess = {
                                                // Reload data
                                                val refreshResult = medicationLogService.getElderMedicationHistory(
                                                    accessToken = accessToken,
                                                    elderId = targetUserId,
                                                    days = selectedDays
                                                )
                                                refreshResult.fold(
                                                    onSuccess = { response ->
                                                        medicationLogs = response.logs
                                                        statistics = response.statistics
                                                    },
                                                    onFailure = { }
                                                )
                                            },
                                            onFailure = { error ->
                                                errorMessage = error.message
                                            }
                                        )
                                    } catch (e: Exception) {
                                        errorMessage = "Không thể xác nhận: ${e.message}"
                                    }
                                }
                            }
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun StatCard(
    modifier: Modifier = Modifier,
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    label: String,
    value: String,
    color: Color
) {
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(containerColor = color.copy(alpha = 0.1f)),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Icon(
                imageVector = icon,
                contentDescription = null,
                tint = color,
                modifier = Modifier.size(24.dp)
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = value,
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold,
                color = color
            )
            Text(
                text = label,
                style = MaterialTheme.typography.bodySmall,
                color = DarkOnSurface.copy(alpha = 0.7f)
            )
        }
    }
}

@Composable
private fun MedicationLogCard(
    log: MedicationLog,
    accessToken: String,
    onConfirmTaken: (Long) -> Unit
) {
    val statusColor = when (log.status) {
        "TAKEN" -> Color(0xFF4CAF50)
        "MISSED" -> Color(0xFFF44336)
        "PENDING" -> Color(0xFFFFC107)
        else -> DarkOnSurface
    }
    
    val statusText = when (log.status) {
        "TAKEN" -> "Đã uống"
        "MISSED" -> "Bỏ lỡ"
        "PENDING" -> "Chờ uống"
        else -> log.status
    }
    
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = DarkSurface),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.Top
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = log.medicationNames,
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold,
                        color = DarkOnSurface
                    )
                    
                    Spacer(modifier = Modifier.height(4.dp))
                    
                    Text(
                        text = "Giờ nhắc: ${formatMedicationLogDateTime(log.reminderTime)}",
                        style = MaterialTheme.typography.bodyMedium,
                        color = DarkOnSurface.copy(alpha = 0.7f)
                    )
                    
                    if (log.actualTakenTime != null) {
                        Text(
                            text = "Đã uống lúc: ${formatMedicationLogDateTime(log.actualTakenTime)}",
                            style = MaterialTheme.typography.bodySmall,
                            color = DarkOnSurface.copy(alpha = 0.6f)
                        )
                    }
                    
                    if (log.minutesLate != null && log.minutesLate != 0) {
                        val lateText = if (log.minutesLate > 0) {
                            "Trễ ${log.minutesLate} phút"
                        } else {
                            "Sớm ${-log.minutesLate} phút"
                        }
                        Text(
                            text = lateText,
                            style = MaterialTheme.typography.bodySmall,
                            color = if (log.minutesLate > 15) Color(0xFFF44336) else Color(0xFFFFC107)
                        )
                    }
                    
                    if (!log.note.isNullOrBlank()) {
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = "Ghi chú: ${log.note}",
                            style = MaterialTheme.typography.bodySmall,
                            color = DarkOnSurface.copy(alpha = 0.5f)
                        )
                    }
                }
                
                Column(
                    horizontalAlignment = Alignment.End
                ) {
                    Surface(
                        color = statusColor.copy(alpha = 0.2f),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Text(
                            text = statusText,
                            color = statusColor,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
                            style = MaterialTheme.typography.labelMedium
                        )
                    }
                    
                    if (log.medicationCount > 1) {
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = "${log.medicationCount} loại thuốc",
                            style = MaterialTheme.typography.bodySmall,
                            color = DarkOnSurface.copy(alpha = 0.6f)
                        )
                    }
                }
            }
            
            // Confirm button for PENDING status
            if (log.status == "PENDING") {
                Spacer(modifier = Modifier.height(12.dp))
                Button(
                    onClick = { onConfirmTaken(log.id) },
                    modifier = Modifier.fillMaxWidth(),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = DarkPrimary
                    ),
                    shape = RoundedCornerShape(8.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.CheckCircle,
                        contentDescription = null,
                        modifier = Modifier.size(20.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Xác nhận đã uống")
                }
            }
        }
    }
}

private fun formatMedicationLogDateTime(dateTimeStr: String): String {
    return try {
        val dateTime = LocalDateTime.parse(dateTimeStr)
        dateTime.format(DateTimeFormatter.ofPattern("HH:mm - dd/MM/yyyy"))
    } catch (e: Exception) {
        dateTimeStr
    }
}
