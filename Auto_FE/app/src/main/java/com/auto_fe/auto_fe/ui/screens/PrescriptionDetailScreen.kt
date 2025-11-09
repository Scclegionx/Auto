package com.auto_fe.auto_fe.ui.screens

import android.widget.Toast
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.auto_fe.auto_fe.ui.service.PrescriptionService
import com.auto_fe.auto_fe.ui.theme.*
import com.auto_fe.auto_fe.ui.theme.AppTextSize
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PrescriptionDetailScreen(
    prescriptionId: Long,
    accessToken: String,
    onBackClick: () -> Unit,
    onEditClick: (Long) -> Unit = {}
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val prescriptionService = remember { PrescriptionService() }
    
    var prescription by remember { mutableStateOf<PrescriptionService.Prescription?>(null) }
    var isLoading by remember { mutableStateOf(true) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var showDeleteDialog by remember { mutableStateOf(false) }
    var isDeleting by remember { mutableStateOf(false) }

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
                actions = {
                    // Nút sửa đơn thuốc
                    IconButton(onClick = {
                        prescription?.let { onEditClick(it.id) }
                    }) {
                        Icon(
                            imageVector = Icons.Default.Edit,
                            contentDescription = "Sửa đơn thuốc",
                            tint = DarkPrimary
                        )
                    }
                    
                    // Nút xóa đơn thuốc
                    IconButton(
                        onClick = { showDeleteDialog = true },
                        enabled = !isDeleting
                    ) {
                        Icon(
                            imageVector = Icons.Default.Delete,
                            contentDescription = "Xóa đơn thuốc",
                            tint = AIError
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
                            AIBackgroundDeep,
                            AIBackgroundSoft
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
                                color = AIError,
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
        
        // Delete Confirmation Dialog
        if (showDeleteDialog) {
            AlertDialog(
                onDismissRequest = { showDeleteDialog = false },
                title = { Text("Xác nhận xóa", color = DarkOnSurface) },
                text = { 
                    Text(
                        "Bạn có chắc muốn xóa đơn thuốc \"${prescription?.name}\"?\n\nThao tác này không thể hoàn tác và sẽ xóa tất cả thuốc trong đơn.",
                        color = DarkOnSurface
                    ) 
                },
                confirmButton = {
                    Button(
                        onClick = {
                            showDeleteDialog = false
                            isDeleting = true
                            scope.launch {
                                val result = prescriptionService.deletePrescription(
                                    prescriptionId = prescriptionId,
                                    accessToken = accessToken
                                )
                                result.fold(
                                    onSuccess = { message ->
                                        Toast.makeText(context, "✅ $message", Toast.LENGTH_SHORT).show()
                                        onBackClick() // Quay về màn hình trước
                                    },
                                    onFailure = { error ->
                                        Toast.makeText(context, "❌ ${error.message}", Toast.LENGTH_LONG).show()
                                        isDeleting = false
                                    }
                                )
                            }
                        },
                        colors = ButtonDefaults.buttonColors(containerColor = AIError),
                        enabled = !isDeleting
                    ) {
                        if (isDeleting) {
                            CircularProgressIndicator(
                                modifier = Modifier.size(16.dp),
                                color = DarkOnPrimary
                            )
                        } else {
                            Text("Xóa")
                        }
                    }
                },
                dismissButton = {
                    TextButton(
                        onClick = { showDeleteDialog = false },
                        enabled = !isDeleting
                    ) {
                        Text("Hủy", color = DarkOnSurface)
                    }
                },
                containerColor = DarkSurface
            )
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
                                fontSize = AppTextSize.titleMedium,
                                fontWeight = FontWeight.Bold,
                                color = DarkPrimary
                            )
                            
                            if (!prescription.description.isNullOrBlank()) {
                                Spacer(modifier = Modifier.height(8.dp))
                                Text(
                                    text = prescription.description,
                                    fontSize = AppTextSize.bodyMedium,
                                    color = DarkOnSurface.copy(alpha = 0.8f),
                                    lineHeight = 22.sp
                                )
                            }
                        }
                        
                        // Status badge
                        Surface(
                            color = if (prescription.isActive) 
                                AISuccess.copy(alpha = 0.2f) 
                            else 
                                DarkOnSurface.copy(alpha = 0.1f),
                            shape = RoundedCornerShape(12.dp)
                        ) {
                            Text(
                                text = if (prescription.isActive) "✓ Đang dùng" else "⏸ Tạm ngưng",
                                fontSize = 11.sp,
                                color = if (prescription.isActive) AISuccess else DarkOnSurface.copy(alpha = 0.5f),
                                modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
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
                            value = "${prescription.medications?.size ?: prescription.medicationReminders?.size ?: 0}",
                            label = "Loại thuốc"
                        )
                        StatItem(
                            icon = "⏰",
                            value = "${prescription.medications?.count { it.isActive } ?: prescription.medicationReminders?.count { it.isActive } ?: 0}",
                            label = "Đang nhắc"
                        )
                        StatItem(
                            icon = "📅",
                            value = getDaysText(prescription),
                            label = "Lịch uống"
                        )
                    }
                    
                    // Nút xem ảnh (nếu có)
                    if (!prescription.imageUrl.isNullOrBlank()) {
                        Spacer(modifier = Modifier.height(16.dp))
                        
                        var showImageDialog by remember { mutableStateOf(false) }
                        
                        OutlinedButton(
                            onClick = { showImageDialog = true },
                            modifier = Modifier.fillMaxWidth(),
                            colors = ButtonDefaults.outlinedButtonColors(
                                contentColor = DarkPrimary
                            ),
                            border = BorderStroke(1.dp, DarkPrimary.copy(alpha = 0.5f))
                        ) {
                            Text(text = "📷", fontSize = 18.sp)
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = "Xem ảnh đơn thuốc",
                                fontSize = 14.sp,
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

        // Medications Header
        item {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "🕐 Lịch nhắc uống thuốc",
                    fontSize = AppTextSize.titleSmall,
                    fontWeight = FontWeight.Bold,
                    color = DarkOnSurface
                )
            }
        }

        // Medication List
        // ✅ Ưu tiên dùng medications (grouped), fallback về medicationReminders (legacy)
        val medications = prescription.medications ?: emptyList()
        val hasData = medications.isNotEmpty()
        
        if (!hasData) {
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
            items(medications.sortedBy { it.medicationName }) { medication ->
                MedicationGroupCard(medication = medication)
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
            fontSize = AppTextSize.titleSmall,
            fontWeight = FontWeight.Bold,
            color = DarkPrimary
        )
        Text(
            text = label,
            fontSize = AppTextSize.bodySmall,
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
                    fontSize = AppTextSize.bodyMedium,
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
                        color = if (medication.isActive) AISuccess else DarkOnSurface.copy(alpha = 0.3f),
                        shape = CircleShape
                    )
            )
        }
    }
}

// ✅ Modern card for grouped medications
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun MedicationGroupCard(medication: PrescriptionService.Medication) {
    Card(
        colors = CardDefaults.cardColors(
            containerColor = DarkSurface
        ),
        shape = RoundedCornerShape(16.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp),
        modifier = Modifier
            .fillMaxWidth()
            .padding(bottom = 12.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
            // Header: Name + Status
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.weight(1f)
                ) {
                    Text(
                        text = "💊",
                        fontSize = 20.sp
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = medication.medicationName,
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold,
                        color = DarkOnSurface
                    )
                }
                
                // Status indicator
                if (medication.isActive) {
                    Box(
                        modifier = Modifier
                            .size(10.dp)
                            .background(SuccessColor, CircleShape)
                    )
                }
            }

            // Notes
            if (!medication.notes.isNullOrBlank()) {
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = medication.notes,
                    fontSize = 14.sp,
                    color = DarkOnSurface.copy(alpha = 0.7f),
                    lineHeight = 20.sp
                )
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Time chips - Clean & Simple
            FlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                medication.reminderTimes.forEach { time ->
                    Surface(
                        shape = RoundedCornerShape(20.dp),
                        color = DarkPrimary.copy(alpha = 0.12f)
                    ) {
                        Row(
                            modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            Text(
                                text = "🕐",
                                fontSize = 16.sp
                            )
                            Text(
                                text = time,
                                fontSize = 15.sp,
                                fontWeight = FontWeight.SemiBold,
                                color = DarkPrimary
                            )
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(12.dp))

            // Days of week - Compact
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                Text(
                    text = "📅",
                    fontSize = 14.sp
                )
                Text(
                    text = formatDaysOfWeek(medication.daysOfWeek),
                    fontSize = 13.sp,
                    color = DarkOnSurface.copy(alpha = 0.6f)
                )
            }
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
        "BEFORE_MEAL" -> AIInfo.copy(alpha = 0.2f)
        "AFTER_MEAL" -> AISuccess.copy(alpha = 0.2f)
        "WITH_MEAL" -> AIWarning.copy(alpha = 0.2f)
        else -> DarkOnSurface.copy(alpha = 0.1f)
    }
}

fun formatDaysOfWeek(daysOfWeek: String): String {
    val days = listOf("T2", "T3", "T4", "T5", "T6", "T7", "CN")
    
    if (daysOfWeek == "1111111") {
        return "Hàng ngày"
    }
    
    val activeDays = daysOfWeek.mapIndexed { index, char ->
        if (char == '1' && index < days.size) days[index] else null
    }.filterNotNull()
    
    return if (activeDays.isEmpty()) {
        "Chưa đặt lịch"
    } else {
        activeDays.joinToString(", ")
    }
}

fun getDaysText(prescription: PrescriptionService.Prescription): String {
    val allDaily = if (prescription.medications != null) {
        prescription.medications.all { it.daysOfWeek == "1111111" }
    } else {
        prescription.medicationReminders?.all { it.daysOfWeek == "1111111" } ?: false
    }
    return if (allDaily) "Hàng ngày" else "Theo lịch"
}

@Composable
fun ZoomableImageDialog(
    imageUrl: String?,
    onDismiss: () -> Unit
) {
    if (imageUrl.isNullOrBlank()) {
        onDismiss()
        return
    }
    
    var scale by remember { mutableStateOf(1f) }
    var offsetX by remember { mutableStateOf(0f) }
    var offsetY by remember { mutableStateOf(0f) }
    
    androidx.compose.ui.window.Dialog(
        onDismissRequest = onDismiss,
        properties = androidx.compose.ui.window.DialogProperties(
            usePlatformDefaultWidth = false
        )
    ) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(androidx.compose.ui.graphics.Color.Black)
        ) {
            // Image với zoom
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .pointerInput(Unit) {
                        detectTransformGestures { _, pan, zoom, _ ->
                            scale = (scale * zoom).coerceIn(1f, 5f)
                            if (scale > 1f) {
                                offsetX += pan.x
                                offsetY += pan.y
                            } else {
                                offsetX = 0f
                                offsetY = 0f
                            }
                        }
                    },
                contentAlignment = Alignment.Center
            ) {
                coil.compose.AsyncImage(
                    model = imageUrl,
                    contentDescription = "Ảnh đơn thuốc",
                    modifier = Modifier
                        .fillMaxSize()
                        .graphicsLayer(
                            scaleX = scale,
                            scaleY = scale,
                            translationX = offsetX,
                            translationY = offsetY
                        ),
                    contentScale = androidx.compose.ui.layout.ContentScale.Fit
                )
            }
            
            // Nút đóng
            IconButton(
                onClick = onDismiss,
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(16.dp)
                    .background(
                        DarkSurface.copy(alpha = 0.7f),
                        shape = CircleShape
                    )
            ) {
                Icon(
                    imageVector = Icons.Default.ArrowBack,
                    contentDescription = "Đóng",
                    tint = DarkOnSurface
                )
            }
            
            // Indicator zoom level
            if (scale > 1f) {
                Surface(
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .padding(16.dp),
                    color = DarkSurface.copy(alpha = 0.7f),
                    shape = RoundedCornerShape(20.dp)
                ) {
                    Text(
                        text = "Zoom: ${String.format("%.1f", scale)}x",
                        modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
                        color = DarkOnSurface,
                        fontSize = 12.sp
                    )
                }
            }
        }
    }
}
