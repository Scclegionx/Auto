package com.auto_fe.auto_fe.ui.screens

import android.widget.Toast
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.auto_fe.auto_fe.ui.service.PrescriptionService
import com.auto_fe.auto_fe.ui.theme.*
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PrescriptionDetailScreen(
    prescriptionId: Long,
    accessToken: String,
    onBackClick: () -> Unit
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val prescriptionService = remember { PrescriptionService() }
    
    var prescription by remember { mutableStateOf<PrescriptionService.Prescription?>(null) }
    var isLoading by remember { mutableStateOf(true) }
    var errorMessage by remember { mutableStateOf<String?>(null) }

    // Load prescription detail when screen opens
    LaunchedEffect(prescriptionId) {
        scope.launch {
            isLoading = true
            val result = prescriptionService.getPrescriptionById(prescriptionId, accessToken)
            result.fold(
                onSuccess = { response ->
                    prescription = response.data
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

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Chi tiết đơn thuốc", color = DarkOnSurface) },
                navigationIcon = {
                    IconButton(onClick = onBackClick) {
                        Icon(
                            imageVector = Icons.Default.ArrowBack,
                            contentDescription = "Quay lại",
                            tint = DarkOnSurface
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = DarkSurface.copy(alpha = 0.95f)
                )
            )
        }
    ) { paddingValues ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .background(
                    brush = Brush.verticalGradient(
                        colors = listOf(
                            DarkGradientStart,
                            DarkGradientEnd
                        )
                    )
                )
        ) {
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
                                text = "Đang tải...",
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
                                color = DarkError,
                                textAlign = TextAlign.Center
                            )
                        }
                    }
                }
                prescription != null -> {
                    PrescriptionDetailContent(prescription = prescription!!)
                }
            }
        }
    }
}

@Composable
fun PrescriptionDetailContent(prescription: PrescriptionService.Prescription) {
    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Header Card
        item {
            Card(
                colors = CardDefaults.cardColors(
                    containerColor = DarkSurface.copy(alpha = 0.9f)
                ),
                shape = RoundedCornerShape(20.dp),
                elevation = CardDefaults.cardElevation(defaultElevation = 6.dp)
            ) {
                Column(
                    modifier = Modifier.padding(20.dp)
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.Top
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = prescription.name,
                                fontSize = 24.sp,
                                fontWeight = FontWeight.Bold,
                                color = DarkPrimary
                            )
                            
                            if (!prescription.description.isNullOrBlank()) {
                                Spacer(modifier = Modifier.height(8.dp))
                                Text(
                                    text = prescription.description,
                                    fontSize = 15.sp,
                                    color = DarkOnSurface.copy(alpha = 0.8f),
                                    lineHeight = 22.sp
                                )
                            }
                        }
                        
                        // Status badge
                        Surface(
                            color = if (prescription.isActive) 
                                SuccessColor.copy(alpha = 0.2f) 
                            else 
                                DarkOnSurface.copy(alpha = 0.1f),
                            shape = RoundedCornerShape(16.dp)
                        ) {
                            Text(
                                text = if (prescription.isActive) "✓ Đang dùng" else "⏸ Tạm ngưng",
                                fontSize = 13.sp,
                                color = if (prescription.isActive) SuccessColor else DarkOnSurface.copy(alpha = 0.5f),
                                modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(16.dp))

                    Divider(color = DarkOnSurface.copy(alpha = 0.1f))

                    Spacer(modifier = Modifier.height(16.dp))

                    // Stats
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceAround
                    ) {
                        StatItem(
                            icon = "💊",
                            value = "${prescription.medicationReminders.size}",
                            label = "Loại thuốc"
                        )
                        StatItem(
                            icon = "⏰",
                            value = "${prescription.medicationReminders.count { it.isActive }}",
                            label = "Đang nhắc"
                        )
                        StatItem(
                            icon = "📅",
                            value = getDaysText(prescription.medicationReminders),
                            label = "Lịch uống"
                        )
                    }
                }
            }
        }

        // Medications Header
        item {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "🕐 Lịch nhắc uống thuốc",
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Bold,
                    color = DarkOnSurface
                )
            }
        }

        // Medication List
        if (prescription.medicationReminders.isEmpty()) {
            item {
                Card(
                    colors = CardDefaults.cardColors(
                        containerColor = DarkSurface.copy(alpha = 0.5f)
                    ),
                    shape = RoundedCornerShape(16.dp)
                ) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(32.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            Text(text = "📋", fontSize = 48.sp)
                            Spacer(modifier = Modifier.height(8.dp))
                            Text(
                                text = "Chưa có lịch nhắc nào",
                                color = DarkOnSurface.copy(alpha = 0.6f),
                                fontSize = 14.sp
                            )
                        }
                    }
                }
            }
        } else {
            items(prescription.medicationReminders.sortedBy { it.reminderTime }) { medication ->
                MedicationCard(medication = medication)
            }
        }
    }
}

@Composable
fun StatItem(icon: String, value: String, label: String) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = icon,
            fontSize = 28.sp
        )
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            text = value,
            fontSize = 20.sp,
            fontWeight = FontWeight.Bold,
            color = DarkPrimary
        )
        Text(
            text = label,
            fontSize = 12.sp,
            color = DarkOnSurface.copy(alpha = 0.6f)
        )
    }
}

@Composable
fun MedicationCard(medication: PrescriptionService.MedicationReminder) {
    Card(
        colors = CardDefaults.cardColors(
            containerColor = DarkSurface.copy(alpha = 0.9f)
        ),
        shape = RoundedCornerShape(16.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Time Circle
            Surface(
                shape = CircleShape,
                color = DarkPrimary.copy(alpha = 0.2f),
                modifier = Modifier.size(60.dp)
            ) {
                Box(contentAlignment = Alignment.Center) {
                    Text(
                        text = medication.reminderTime,
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Bold,
                        color = DarkPrimary,
                        textAlign = TextAlign.Center
                    )
                }
            }

            Spacer(modifier = Modifier.width(16.dp))

            // Info
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = medication.name,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    color = DarkOnSurface
                )
                
                Spacer(modifier = Modifier.height(4.dp))
                
                // Type badge
                Surface(
                    color = getTypeBadgeColor(medication.type),
                    shape = RoundedCornerShape(8.dp)
                ) {
                    Text(
                        text = getTypeText(medication.type),
                        fontSize = 11.sp,
                        color = DarkOnSurface,
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp),
                        fontWeight = FontWeight.Medium
                    )
                }

                if (!medication.description.isNullOrBlank()) {
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = medication.description,
                        fontSize = 13.sp,
                        color = DarkOnSurface.copy(alpha = 0.7f)
                    )
                }

                Spacer(modifier = Modifier.height(6.dp))

                // Days of week
                Text(
                    text = formatDaysOfWeek(medication.daysOfWeek),
                    fontSize = 12.sp,
                    color = DarkOnSurface.copy(alpha = 0.6f)
                )
            }

            // Status indicator
            Box(
                modifier = Modifier
                    .size(12.dp)
                    .background(
                        color = if (medication.isActive) SuccessColor else DarkOnSurface.copy(alpha = 0.3f),
                        shape = CircleShape
                    )
            )
        }
    }
}

fun getTypeText(type: String): String {
    return when (type) {
        "BEFORE_MEAL" -> "🍽 Trước ăn"
        "AFTER_MEAL" -> "🍽 Sau ăn"
        "WITH_MEAL" -> "🍽 Trong bữa ăn"
        else -> "💊 $type"
    }
}

fun getTypeBadgeColor(type: String): androidx.compose.ui.graphics.Color {
    return when (type) {
        "BEFORE_MEAL" -> InfoColor.copy(alpha = 0.2f)
        "AFTER_MEAL" -> SuccessColor.copy(alpha = 0.2f)
        "WITH_MEAL" -> WarningColor.copy(alpha = 0.2f)
        else -> DarkOnSurface.copy(alpha = 0.1f)
    }
}

fun formatDaysOfWeek(daysOfWeek: String): String {
    val days = listOf("T2", "T3", "T4", "T5", "T6", "T7", "CN")
    
    if (daysOfWeek == "1111111") {
        return "📅 Hàng ngày"
    }
    
    val activeDays = daysOfWeek.mapIndexed { index, char ->
        if (char == '1' && index < days.size) days[index] else null
    }.filterNotNull()
    
    return if (activeDays.isEmpty()) {
        "📅 Chưa đặt lịch"
    } else {
        "📅 ${activeDays.joinToString(", ")}"
    }
}

fun getDaysText(medications: List<PrescriptionService.MedicationReminder>): String {
    val allDaily = medications.all { it.daysOfWeek == "1111111" }
    return if (allDaily) "Hàng ngày" else "Theo lịch"
}
