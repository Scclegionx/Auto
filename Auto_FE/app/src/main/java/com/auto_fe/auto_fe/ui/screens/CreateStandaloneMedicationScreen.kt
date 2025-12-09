package com.auto_fe.auto_fe.ui.screens

import android.widget.Toast
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.auto_fe.auto_fe.ui.service.StandaloneMedicationService
import com.auto_fe.auto_fe.ui.theme.*
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CreateStandaloneMedicationScreen(
    accessToken: String,
    onDismiss: () -> Unit,
    onSuccess: () -> Unit,
    elderUserId: Long? = null,  // Nếu có = Supervisor tạo cho Elder
    elderUserName: String? = null  // Tên Elder để hiển thị
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val medicationService = remember { StandaloneMedicationService() }
    
    val isSupervisorMode = elderUserId != null

    var medicationName by remember { mutableStateOf("") }
    var description by remember { mutableStateOf("") }
    var selectedTimes by remember { mutableStateOf<List<String>>(emptyList()) }
    var daysOfWeek by remember { mutableStateOf("1111111") } // Default: everyday
    var isLoading by remember { mutableStateOf(false) }

    // Time picker dialog state
    var showTimePicker by remember { mutableStateOf(false) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { 
                    Text(
                        text = if (isSupervisorMode) {
                            "Thêm Thuốc cho $elderUserName"
                        } else {
                            "Thêm Thuốc Ngoài Đơn"
                        },
                        fontWeight = FontWeight.Bold
                    ) 
                },
                navigationIcon = {
                    IconButton(onClick = onDismiss) {
                        Icon(Icons.Default.Close, "Đóng")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = DarkBackground,
                    titleContentColor = DarkOnSurface,
                    navigationIconContentColor = DarkOnSurface
                )
            )
        },
        containerColor = DarkBackground
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Medication Name
            OutlinedTextField(
                value = medicationName,
                onValueChange = { medicationName = it },
                label = { Text("Tên thuốc") },
                placeholder = { Text("VD: Paracetamol 500mg") },
                modifier = Modifier.fillMaxWidth(),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = DarkPrimary,
                    unfocusedBorderColor = DarkOnSurface.copy(alpha = 0.3f),
                    focusedLabelColor = DarkPrimary,
                    unfocusedLabelColor = DarkOnSurface.copy(alpha = 0.6f),
                    focusedTextColor = DarkOnSurface,
                    unfocusedTextColor = DarkOnSurface
                ),
                singleLine = true
            )

            // Description
            OutlinedTextField(
                value = description,
                onValueChange = { description = it },
                label = { Text("Ghi chú") },
                placeholder = { Text("Mô tả cách dùng, liều lượng...") },
                modifier = Modifier.fillMaxWidth(),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = DarkPrimary,
                    unfocusedBorderColor = DarkOnSurface.copy(alpha = 0.3f),
                    focusedLabelColor = DarkPrimary,
                    unfocusedLabelColor = DarkOnSurface.copy(alpha = 0.6f),
                    focusedTextColor = DarkOnSurface,
                    unfocusedTextColor = DarkOnSurface
                ),
                minLines = 3,
                maxLines = 5
            )

            // Reminder Times Section
            Card(
                colors = CardDefaults.cardColors(containerColor = DarkSurface.copy(alpha = 0.5f)),
                shape = RoundedCornerShape(12.dp)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            "Giờ nhắc nhở",
                            fontSize = 16.sp,
                            fontWeight = FontWeight.SemiBold,
                            color = DarkOnSurface
                        )
                        IconButton(onClick = { showTimePicker = true }) {
                            Icon(
                                Icons.Default.Add,
                                "Thêm giờ",
                                tint = DarkPrimary
                            )
                        }
                    }

                    if (selectedTimes.isEmpty()) {
                        Text(
                            "Chưa có giờ nhắc nhở",
                            fontSize = 14.sp,
                            color = DarkOnSurface.copy(alpha = 0.5f),
                            modifier = Modifier.padding(vertical = 8.dp)
                        )
                    } else {
                        Column(
                            verticalArrangement = Arrangement.spacedBy(8.dp),
                            modifier = Modifier.padding(top = 8.dp)
                        ) {
                            selectedTimes.sorted().forEach { time ->
                                Surface(
                                    color = DarkPrimary.copy(alpha = 0.2f),
                                    shape = RoundedCornerShape(8.dp)
                                ) {
                                    Row(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .padding(horizontal = 12.dp, vertical = 8.dp),
                                        horizontalArrangement = Arrangement.SpaceBetween,
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        Text(
                                            time,
                                            fontSize = 16.sp,
                                            fontWeight = FontWeight.Medium,
                                            color = DarkOnSurface
                                        )
                                        IconButton(
                                            onClick = {
                                                selectedTimes = selectedTimes - time
                                            },
                                            modifier = Modifier.size(32.dp)
                                        ) {
                                            Icon(
                                                Icons.Default.Close,
                                                "Xóa",
                                                tint = AIError,
                                                modifier = Modifier.size(20.dp)
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Days of Week Section
            Card(
                colors = CardDefaults.cardColors(containerColor = DarkSurface.copy(alpha = 0.5f)),
                shape = RoundedCornerShape(12.dp)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        "Ngày uống thuốc",
                        fontSize = 16.sp,
                        fontWeight = FontWeight.SemiBold,
                        color = DarkOnSurface,
                        modifier = Modifier.padding(bottom = 12.dp)
                    )

                    DaysOfWeekSelector(
                        selectedDays = daysOfWeek,
                        onDaysChanged = { daysOfWeek = it }
                    )
                }
            }

            Spacer(modifier = Modifier.weight(1f))

            // Create Button
            Button(
                onClick = {
                    when {
                        medicationName.isBlank() -> {
                            Toast.makeText(context, "Vui lòng nhập tên thuốc", Toast.LENGTH_SHORT).show()
                        }
                        selectedTimes.isEmpty() -> {
                            Toast.makeText(context, "Vui lòng chọn ít nhất 1 giờ nhắc nhở", Toast.LENGTH_SHORT).show()
                        }
                        !daysOfWeek.contains('1') -> {
                            Toast.makeText(context, "Vui lòng chọn ít nhất 1 ngày", Toast.LENGTH_SHORT).show()
                        }
                        else -> {
                            scope.launch {
                                isLoading = true
                                val request = StandaloneMedicationService.MedicationRequest(
                                    name = medicationName.trim(),
                                    description = description.trim().ifBlank { null },
                                    type = "OVER_THE_COUNTER",
                                    reminderTimes = selectedTimes.sorted(),
                                    daysOfWeek = daysOfWeek,
                                    isActive = true,
                                    elderUserId = elderUserId  // Pass elderUserId to service
                                )

                                val result = medicationService.create(accessToken, request)
                                result.fold(
                                    onSuccess = { response ->
                                        isLoading = false
                                        Toast.makeText(context, "${response.message}", Toast.LENGTH_SHORT).show()
                                        onSuccess()
                                    },
                                    onFailure = { error ->
                                        isLoading = false
                                        Toast.makeText(context, "${error.message}", Toast.LENGTH_LONG).show()
                                    }
                                )
                            }
                        }
                    }
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp),
                enabled = !isLoading,
                colors = ButtonDefaults.buttonColors(
                    containerColor = DarkPrimary,
                    contentColor = DarkOnPrimary
                ),
                shape = RoundedCornerShape(12.dp)
            ) {
                if (isLoading) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(24.dp),
                        color = DarkOnPrimary
                    )
                } else {
                    Text(
                        "Thêm Thuốc",
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            }
        }
    }

    // Time Picker Dialog
    if (showTimePicker) {
        TimePickerDialog(
            onDismiss = { showTimePicker = false },
            onTimeSelected = { hour, minute ->
                val timeString = String.format("%02d:%02d", hour, minute)
                if (!selectedTimes.contains(timeString)) {
                    selectedTimes = selectedTimes + timeString
                }
                showTimePicker = false
            }
        )
    }
}

@Composable
private fun DaysOfWeekSelector(
    selectedDays: String,
    onDaysChanged: (String) -> Unit
) {
    // T2-CN: index 0=T2, 1=T3, 2=T4, 3=T5, 4=T6, 5=T7, 6=CN
    val dayLabels = listOf("T2", "T3", "T4", "T5", "T6", "T7", "CN")
    
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        dayLabels.forEachIndexed { index, label ->
            val isSelected = selectedDays.getOrNull(index) == '1'
            
            Surface(
                onClick = {
                    val newDays = selectedDays.toCharArray()
                    newDays[index] = if (isSelected) '0' else '1'
                    onDaysChanged(String(newDays))
                },
                shape = RoundedCornerShape(8.dp),
                color = if (isSelected) DarkPrimary else DarkOnSurface.copy(alpha = 0.1f),
                modifier = Modifier.size(44.dp)
            ) {
                Box(
                    contentAlignment = Alignment.Center,
                    modifier = Modifier.fillMaxSize()
                ) {
                    Text(
                        label,
                        fontSize = 13.sp,
                        fontWeight = FontWeight.SemiBold,
                        color = if (isSelected) DarkOnPrimary else DarkOnSurface.copy(alpha = 0.6f)
                    )
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun TimePickerDialog(
    onDismiss: () -> Unit,
    onTimeSelected: (Int, Int) -> Unit
) {
    val timePickerState = rememberTimePickerState()

    AlertDialog(
        onDismissRequest = onDismiss,
        confirmButton = {
            TextButton(
                onClick = {
                    onTimeSelected(timePickerState.hour, timePickerState.minute)
                }
            ) {
                Text("OK", color = DarkPrimary, fontWeight = FontWeight.Bold)
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Hủy", color = DarkOnSurface.copy(alpha = 0.7f))
            }
        },
        title = {
            Text(
                "Chọn giờ nhắc nhở",
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold,
                color = DarkOnSurface
            )
        },
        text = {
            TimePicker(
                state = timePickerState,
                colors = TimePickerDefaults.colors(
                    clockDialColor = DarkBackground,
                    selectorColor = DarkPrimary,
                    timeSelectorSelectedContainerColor = DarkPrimary,
                    timeSelectorUnselectedContainerColor = DarkOnSurface.copy(alpha = 0.1f),
                    timeSelectorSelectedContentColor = DarkOnPrimary,
                    timeSelectorUnselectedContentColor = DarkOnSurface
                )
            )
        },
        containerColor = DarkSurface,
        tonalElevation = 8.dp
    )
}
