package com.auto_fe.auto_fe.ui.screens

import android.widget.Toast
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.auto_fe.auto_fe.ui.service.PrescriptionService
import com.auto_fe.auto_fe.ui.theme.*
import kotlinx.coroutines.launch

data class MedicationReminderForm(
    var name: String = "",
    var description: String = "",
    var type: String = "PRESCRIPTION", // PRESCRIPTION, OVER_THE_COUNTER
    var mealTiming: String = "AFTER_MEAL", // BEFORE_MEAL, AFTER_MEAL, WITH_MEAL - sẽ thêm vào description
    var reminderTimes: MutableList<String> = mutableListOf("08:00"),
    var daysOfWeek: String = "1111111" // Mon-Sun: 1111111 = everyday
)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CreatePrescriptionScreen(
    accessToken: String,
    onBackClick: () -> Unit,
    onSuccess: () -> Unit
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val prescriptionService = remember { PrescriptionService() }

    // Form states
    var name by remember { mutableStateOf("") }
    var description by remember { mutableStateOf("") }
    var imageUrl by remember { mutableStateOf("https://via.placeholder.com/150") }
    var medications by remember { mutableStateOf(listOf(MedicationReminderForm())) }
    var isLoading by remember { mutableStateOf(false) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Tạo đơn thuốc mới", color = DarkOnSurface) },
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
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .verticalScroll(rememberScrollState())
                    .padding(16.dp)
            ) {
                // Thông tin đơn thuốc
                Card(
                    colors = CardDefaults.cardColors(containerColor = DarkSurface.copy(alpha = 0.9f)),
                    shape = RoundedCornerShape(16.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text(
                            text = "📋 Thông tin đơn thuốc",
                            fontSize = 18.sp,
                            fontWeight = FontWeight.Bold,
                            color = DarkPrimary
                        )

                        Spacer(modifier = Modifier.height(16.dp))

                        OutlinedTextField(
                            value = name,
                            onValueChange = { name = it },
                            label = { Text("Tên đơn thuốc", color = DarkOnSurface.copy(alpha = 0.7f)) },
                            placeholder = { Text("VD: Đơn thuốc tháng 10/2025") },
                            singleLine = true,
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = DarkPrimary,
                                unfocusedBorderColor = DarkOnSurface.copy(alpha = 0.3f),
                                focusedTextColor = DarkOnSurface,
                                unfocusedTextColor = DarkOnSurface
                            ),
                            modifier = Modifier.fillMaxWidth()
                        )

                        Spacer(modifier = Modifier.height(12.dp))

                        OutlinedTextField(
                            value = description,
                            onValueChange = { description = it },
                            label = { Text("Mô tả", color = DarkOnSurface.copy(alpha = 0.7f)) },
                            placeholder = { Text("Ghi chú về đơn thuốc") },
                            minLines = 3,
                            maxLines = 5,
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = DarkPrimary,
                                unfocusedBorderColor = DarkOnSurface.copy(alpha = 0.3f),
                                focusedTextColor = DarkOnSurface,
                                unfocusedTextColor = DarkOnSurface
                            ),
                            modifier = Modifier.fillMaxWidth()
                        )
                    }
                }

                Spacer(modifier = Modifier.height(16.dp))

                // Danh sách thuốc
                Card(
                    colors = CardDefaults.cardColors(containerColor = DarkSurface.copy(alpha = 0.9f)),
                    shape = RoundedCornerShape(16.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = "💊 Danh sách thuốc (${medications.size})",
                                fontSize = 18.sp,
                                fontWeight = FontWeight.Bold,
                                color = DarkPrimary
                            )

                            IconButton(
                                onClick = {
                                    medications = medications + MedicationReminderForm()
                                }
                            ) {
                                Icon(
                                    imageVector = Icons.Default.Add,
                                    contentDescription = "Thêm thuốc",
                                    tint = DarkPrimary
                                )
                            }
                        }

                        Spacer(modifier = Modifier.height(16.dp))

                        medications.forEachIndexed { index, medication ->
                            MedicationReminderCard(
                                medication = medication,
                                index = index,
                                onUpdate = { updated ->
                                    medications = medications.toMutableList().apply {
                                        set(index, updated)
                                    }
                                },
                                onDelete = {
                                    if (medications.size > 1) {
                                        medications = medications.toMutableList().apply {
                                            removeAt(index)
                                        }
                                    } else {
                                        Toast.makeText(context, "Phải có ít nhất 1 loại thuốc", Toast.LENGTH_SHORT).show()
                                    }
                                }
                            )

                            if (index < medications.size - 1) {
                                Spacer(modifier = Modifier.height(12.dp))
                                Divider(color = DarkOnSurface.copy(alpha = 0.1f))
                                Spacer(modifier = Modifier.height(12.dp))
                            }
                        }
                    }
                }

                Spacer(modifier = Modifier.height(24.dp))

                // Buttons
                Button(
                    onClick = {
                        if (name.isBlank()) {
                            Toast.makeText(context, "❌ Vui lòng nhập tên đơn thuốc", Toast.LENGTH_SHORT).show()
                            return@Button
                        }
                        if (description.isBlank()) {
                            Toast.makeText(context, "❌ Vui lòng nhập mô tả", Toast.LENGTH_SHORT).show()
                            return@Button
                        }
                        if (medications.any { it.name.isBlank() }) {
                            Toast.makeText(context, "❌ Vui lòng nhập tên thuốc", Toast.LENGTH_SHORT).show()
                            return@Button
                        }

                        isLoading = true
                        scope.launch {
                            val result = prescriptionService.createPrescription(
                                name = name,
                                description = description,
                                imageUrl = imageUrl,
                                medications = medications,
                                accessToken = accessToken
                            )
                            result.fold(
                                onSuccess = { response ->
                                    Toast.makeText(context, "✅ ${response.message}", Toast.LENGTH_SHORT).show()
                                    onSuccess()
                                },
                                onFailure = { error ->
                                    Toast.makeText(context, "❌ ${error.message}", Toast.LENGTH_LONG).show()
                                    isLoading = false
                                }
                            )
                        }
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(56.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = DarkPrimary),
                    enabled = !isLoading
                ) {
                    if (isLoading) {
                        CircularProgressIndicator(
                            color = DarkOnPrimary,
                            modifier = Modifier.size(24.dp)
                        )
                    } else {
                        Text("✅ Tạo đơn thuốc", fontSize = 16.sp, fontWeight = FontWeight.Bold)
                    }
                }

                Spacer(modifier = Modifier.height(16.dp))
            }
        }
    }
}

@Composable
fun MedicationReminderCard(
    medication: MedicationReminderForm,
    index: Int,
    onUpdate: (MedicationReminderForm) -> Unit,
    onDelete: () -> Unit
) {
    Column {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "Thuốc ${index + 1}",
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold,
                color = DarkOnSurface
            )

            IconButton(onClick = onDelete) {
                Icon(
                    imageVector = Icons.Default.Delete,
                    contentDescription = "Xóa",
                    tint = DarkError
                )
            }
        }

        Spacer(modifier = Modifier.height(8.dp))

        OutlinedTextField(
            value = medication.name,
            onValueChange = { onUpdate(medication.copy(name = it)) },
            label = { Text("Tên thuốc", color = DarkOnSurface.copy(alpha = 0.7f)) },
            placeholder = { Text("VD: Paracetamol 500mg") },
            singleLine = true,
            colors = OutlinedTextFieldDefaults.colors(
                focusedBorderColor = DarkPrimary,
                unfocusedBorderColor = DarkOnSurface.copy(alpha = 0.3f),
                focusedTextColor = DarkOnSurface,
                unfocusedTextColor = DarkOnSurface
            ),
            modifier = Modifier.fillMaxWidth()
        )

        Spacer(modifier = Modifier.height(12.dp))

        OutlinedTextField(
            value = medication.description,
            onValueChange = { onUpdate(medication.copy(description = it)) },
            label = { Text("Ghi chú", color = DarkOnSurface.copy(alpha = 0.7f)) },
            placeholder = { Text("VD: 1 viên trước ăn sáng") },
            minLines = 2,
            colors = OutlinedTextFieldDefaults.colors(
                focusedBorderColor = DarkPrimary,
                unfocusedBorderColor = DarkOnSurface.copy(alpha = 0.3f),
                focusedTextColor = DarkOnSurface,
                unfocusedTextColor = DarkOnSurface
            ),
            modifier = Modifier.fillMaxWidth()
        )

        Spacer(modifier = Modifier.height(12.dp))

        // Medication Type selection (PRESCRIPTION or OVER_THE_COUNTER)
        Text(
            text = "Loại thuốc:",
            fontSize = 14.sp,
            color = DarkOnSurface.copy(alpha = 0.7f)
        )
        Spacer(modifier = Modifier.height(8.dp))
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            listOf(
                "PRESCRIPTION" to "💊 Thuốc kê đơn",
                "OVER_THE_COUNTER" to "🏪 Thuốc không kê đơn"
            ).forEach { (type, label) ->
                FilterChip(
                    selected = medication.type == type,
                    onClick = { onUpdate(medication.copy(type = type)) },
                    label = { Text(label, fontSize = 12.sp) },
                    colors = FilterChipDefaults.filterChipColors(
                        selectedContainerColor = DarkPrimary,
                        selectedLabelColor = DarkOnPrimary
                    )
                )
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        // Meal Timing selection
        Text(
            text = "Thời điểm uống:",
            fontSize = 14.sp,
            color = DarkOnSurface.copy(alpha = 0.7f)
        )
        Spacer(modifier = Modifier.height(8.dp))
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            listOf(
                "BEFORE_MEAL" to "🍽 Trước ăn",
                "AFTER_MEAL" to "🍴 Sau ăn",
                "WITH_MEAL" to "🥄 Trong bữa"
            ).forEach { (mealTiming, label) ->
                FilterChip(
                    selected = medication.mealTiming == mealTiming,
                    onClick = { onUpdate(medication.copy(mealTiming = mealTiming)) },
                    label = { Text(label, fontSize = 12.sp) },
                    colors = FilterChipDefaults.filterChipColors(
                        selectedContainerColor = DarkPrimary,
                        selectedLabelColor = DarkOnPrimary
                    )
                )
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        // Reminder times
        Text(
            text = "⏰ Giờ nhắc nhở:",
            fontSize = 14.sp,
            color = DarkOnSurface.copy(alpha = 0.7f)
        )
        Spacer(modifier = Modifier.height(8.dp))

        medication.reminderTimes.forEachIndexed { timeIndex, time ->
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                OutlinedTextField(
                    value = time,
                    onValueChange = { newTime ->
                        val times = medication.reminderTimes.toMutableList()
                        times[timeIndex] = newTime
                        onUpdate(medication.copy(reminderTimes = times))
                    },
                    label = { Text("Giờ ${timeIndex + 1}", fontSize = 12.sp) },
                    placeholder = { Text("HH:mm") },
                    singleLine = true,
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = DarkPrimary,
                        unfocusedBorderColor = DarkOnSurface.copy(alpha = 0.3f),
                        focusedTextColor = DarkOnSurface,
                        unfocusedTextColor = DarkOnSurface
                    ),
                    modifier = Modifier.weight(1f)
                )

                if (medication.reminderTimes.size > 1) {
                    IconButton(
                        onClick = {
                            val times = medication.reminderTimes.toMutableList()
                            times.removeAt(timeIndex)
                            onUpdate(medication.copy(reminderTimes = times))
                        }
                    ) {
                        Text("❌", fontSize = 16.sp)
                    }
                }
            }
            Spacer(modifier = Modifier.height(4.dp))
        }

        TextButton(
            onClick = {
                val times = medication.reminderTimes.toMutableList()
                times.add("12:00")
                onUpdate(medication.copy(reminderTimes = times))
            }
        ) {
            Text("➕ Thêm giờ nhắc", color = DarkPrimary, fontSize = 13.sp)
        }

        Spacer(modifier = Modifier.height(12.dp))

        // Days of week
        Text(
            text = "📅 Ngày trong tuần:",
            fontSize = 14.sp,
            color = DarkOnSurface.copy(alpha = 0.7f)
        )
        Spacer(modifier = Modifier.height(8.dp))

        DaysOfWeekSelector(
            selectedDays = medication.daysOfWeek,
            onDaysChange = { onUpdate(medication.copy(daysOfWeek = it)) }
        )
    }
}

@Composable
fun DaysOfWeekSelector(
    selectedDays: String,
    onDaysChange: (String) -> Unit
) {
    val days = listOf("T2", "T3", "T4", "T5", "T6", "T7", "CN")

    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        days.forEachIndexed { index, day ->
            val isSelected = selectedDays.getOrNull(index) == '1'
            FilterChip(
                selected = isSelected,
                onClick = {
                    val newDays = selectedDays.toCharArray()
                    newDays[index] = if (isSelected) '0' else '1'
                    onDaysChange(String(newDays))
                },
                label = { Text(day, fontSize = 11.sp) },
                colors = FilterChipDefaults.filterChipColors(
                    selectedContainerColor = DarkPrimary,
                    selectedLabelColor = DarkOnPrimary
                ),
                modifier = Modifier.weight(1f)
            )
        }
    }

    Spacer(modifier = Modifier.height(8.dp))

    // Quick select
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        TextButton(onClick = { onDaysChange("1111111") }) {
            Text("Hàng ngày", fontSize = 11.sp, color = DarkPrimary)
        }
        TextButton(onClick = { onDaysChange("1111100") }) {
            Text("T2-T6", fontSize = 11.sp, color = DarkPrimary)
        }
        TextButton(onClick = { onDaysChange("0000011") }) {
            Text("Cuối tuần", fontSize = 11.sp, color = DarkPrimary)
        }
    }
}
