package com.auto_fe.auto_fe.ui.screens

import android.net.Uri
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.*
import androidx.compose.foundation.Image
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
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import coil.compose.rememberAsyncImagePainter
import com.auto_fe.auto_fe.service.be.PrescriptionService
import com.auto_fe.auto_fe.service.be.OcrService
import com.auto_fe.auto_fe.ui.theme.*
import com.auto_fe.auto_fe.ui.theme.AppTextSize
import kotlinx.coroutines.launch
import java.io.File
import java.io.FileOutputStream

data class MedicationReminderForm(
    var name: String = "",
    var description: String = "",
    var type: String = "PRESCRIPTION", // Mặc định là thuốc kê đơn (theo đơn thuốc)
    var reminderTimes: MutableList<String> = mutableListOf("08:00"),
    var daysOfWeek: String = "1111111" // Mon-Sun: 1111111 = everyday
)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CreatePrescriptionScreen(
    accessToken: String,
    onBackClick: () -> Unit,
    onSuccess: () -> Unit,
    editPrescriptionId: Long? = null,  // Null = tạo mới, có giá trị = chỉnh sửa
    elderUserId: Long? = null,  // Nếu có = Supervisor tạo cho Elder
    elderUserName: String? = null  // Tên Elder để hiển thị
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val prescriptionService = remember { PrescriptionService() }
    val ocrService = remember { OcrService() }
    
    val isEditMode = editPrescriptionId != null
    val isSupervisorMode = elderUserId != null

    // Form states
    var name by remember { mutableStateOf("") }
    var description by remember { mutableStateOf("") }
    var imageUrl by remember { mutableStateOf("") }
    var medications by remember { mutableStateOf(listOf(MedicationReminderForm())) }
    var isLoading by remember { mutableStateOf(false) }
    var isLoadingData by remember { mutableStateOf(isEditMode) }
    var isOcrProcessing by remember { mutableStateOf(false) }
    var selectedImageUri by remember { mutableStateOf<Uri?>(null) }
    var selectedImageFile by remember { mutableStateOf<File?>(null) }  // Lưu File để upload sau
    var ocrJob by remember { mutableStateOf<kotlinx.coroutines.Job?>(null) }
    
    // Dialog states
    var showEditDialog by remember { mutableStateOf(false) }
    var editingIndex by remember { mutableStateOf<Int?>(null) }
    var editingMedication by remember { mutableStateOf<MedicationReminderForm?>(null) }
    
    // Error states cho validation
    var nameError by remember { mutableStateOf<String?>(null) }
    var descriptionError by remember { mutableStateOf<String?>(null) }
    var medicationErrors by remember { mutableStateOf<Map<Int, String>>(emptyMap()) }

    // Image picker for OCR (AI tự động điền)
    val ocrImagePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            selectedImageUri = it
            ocrJob = scope.launch {
                try {
                    // Convert URI to File
                    val inputStream = context.contentResolver.openInputStream(it)
                    val file = File(context.cacheDir, "prescription_${System.currentTimeMillis()}.jpg")
                    val outputStream = FileOutputStream(file)
                    inputStream?.copyTo(outputStream)
                    inputStream?.close()
                    outputStream.close()

                    // Lưu File để upload sau
                    selectedImageFile = file

                    // Call OCR API to extract data ONLY
                    isOcrProcessing = true
                    val result = ocrService.extractPrescriptionFromImage(file, accessToken)
                    
                    result.fold(
                        onSuccess = { ocrResult ->
                            // Auto-fill form với kết quả OCR
                            name = ocrResult.name
                            description = ocrResult.description
                            // KHÔNG dùng imageUrl từ OCR nữa (vì chưa upload)
                            
                            medications = ocrResult.medications.map { med ->
                                MedicationReminderForm(
                                    name = med.name,
                                    description = med.description,
                                    type = med.type,
                                    reminderTimes = med.reminderTimes.toMutableList(),
                                    daysOfWeek = med.daysOfWeek
                                )
                            }
                            
                            Toast.makeText(
                                context,
                                "OCR thành công! Đã tự động điền thông tin",
                                Toast.LENGTH_LONG
                            ).show()
                        },
                        onFailure = { error ->
                            Toast.makeText(
                                context,
                                "OCR thất bại: ${error.message}",
                                Toast.LENGTH_LONG
                            ).show()
                        }
                    )
                    
                    isOcrProcessing = false
                } catch (e: Exception) {
                    Toast.makeText(context, "Lỗi: ${e.message}", Toast.LENGTH_LONG).show()
                    isOcrProcessing = false
                }
            }
        }
    }

    // Image picker thông thường (chỉ đính kèm ảnh, không OCR)
    val simpleImagePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            selectedImageUri = it
            // Convert URI to File
            try {
                val inputStream = context.contentResolver.openInputStream(it)
                val file = File(context.cacheDir, "prescription_${System.currentTimeMillis()}.jpg")
                val outputStream = FileOutputStream(file)
                inputStream?.copyTo(outputStream)
                inputStream?.close()
                outputStream.close()

                selectedImageFile = file
                Toast.makeText(context, "Đã chọn ảnh", Toast.LENGTH_SHORT).show()
            } catch (e: Exception) {
                Toast.makeText(context, "Lỗi: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        }
    }

    // Load prescription data khi ở chế độ edit
    LaunchedEffect(editPrescriptionId) {
        if (editPrescriptionId != null) {
            scope.launch {
                isLoadingData = true
                val result = prescriptionService.getPrescriptionById(editPrescriptionId, accessToken)
                result.fold(
                    onSuccess = { response ->
                        response.data?.let { prescription ->
                            name = prescription.name
                            description = prescription.description ?: ""
                            imageUrl = prescription.imageUrl ?: ""
                            
                            // Convert medications to form
                            medications = prescription.medications?.map { med ->
                                MedicationReminderForm(
                                    name = med.medicationName,
                                    description = med.notes ?: "",
                                    type = med.type,
                                    reminderTimes = med.reminderTimes.toMutableList(),
                                    daysOfWeek = med.daysOfWeek
                                )
                            } ?: listOf(MedicationReminderForm())
                        }
                        isLoadingData = false
                    },
                    onFailure = { error ->
                        Toast.makeText(context, "${error.message}", Toast.LENGTH_LONG).show()
                        isLoadingData = false
                        onBackClick()
                    }
                )
            }
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { 
                    Text(
                        text = when {
                            isEditMode && isSupervisorMode -> "Sửa đơn thuốc cho ${elderUserName}"
                            isEditMode -> "Sửa đơn thuốc"
                            isSupervisorMode -> "Tạo đơn thuốc cho ${elderUserName}"
                            else -> "Tạo đơn thuốc mới"
                        },
                        color = DarkOnSurface
                    )
                },
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
                            AIBackgroundDeep,
                            AIBackgroundSoft
                        )
                    )
                )
        ) {
            if (isLoadingData || isOcrProcessing) {
                // Loading state khi đang tải dữ liệu edit hoặc OCR
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        modifier = Modifier.padding(24.dp)
                    ) {
                        // Hiển thị ảnh đã chọn
                        if (isOcrProcessing && selectedImageUri != null) {
                            Card(
                                colors = CardDefaults.cardColors(
                                    containerColor = DarkSurface.copy(alpha = 0.9f)
                                ),
                                shape = RoundedCornerShape(16.dp),
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(250.dp)
                            ) {
                                Image(
                                    painter = rememberAsyncImagePainter(selectedImageUri),
                                    contentDescription = "Selected image",
                                    modifier = Modifier
                                        .fillMaxSize()
                                        .clip(RoundedCornerShape(16.dp)),
                                    contentScale = ContentScale.Fit
                                )
                            }
                            Spacer(modifier = Modifier.height(24.dp))
                        }
                        
                        // Rotating loading indicator
                        val infiniteTransition = rememberInfiniteTransition(label = "rotation")
                        val rotation by infiniteTransition.animateFloat(
                            initialValue = 0f,
                            targetValue = 360f,
                            animationSpec = infiniteRepeatable(
                                animation = tween(1000, easing = LinearEasing),
                                repeatMode = RepeatMode.Restart
                            ),
                            label = "rotation"
                        )
                        
                        Box(
                            modifier = Modifier.size(64.dp),
                            contentAlignment = Alignment.Center
                        ) {
                            CircularProgressIndicator(
                                color = DarkPrimary,
                                modifier = Modifier
                                    .size(64.dp)
                                    .rotate(rotation),
                                strokeWidth = 4.dp
                            )
                        }
                        
                        Spacer(modifier = Modifier.height(16.dp))
                        
                        Text(
                            text = if (isOcrProcessing) "Đang quét đơn thuốc..." else "Đang tải dữ liệu...",
                            color = DarkOnSurface.copy(alpha = 0.7f),
                            fontSize = 18.sp,
                            fontWeight = FontWeight.Bold
                        )
                        
                        if (isOcrProcessing) {
                            Spacer(modifier = Modifier.height(8.dp))
                            Text(
                                text = "AI đang phân tích ảnh của bạn",
                                color = DarkOnSurface.copy(alpha = 0.6f),
                                fontSize = 14.sp
                            )
                            Spacer(modifier = Modifier.height(4.dp))
                            Text(
                                text = "Vui lòng chờ 10-30 giây",
                                color = DarkOnSurface.copy(alpha = 0.5f),
                                fontSize = 12.sp
                            )
                            
                            Spacer(modifier = Modifier.height(24.dp))
                            
                            // Nút huỷ
                            OutlinedButton(
                                onClick = {
                                    ocrJob?.cancel()
                                    isOcrProcessing = false
                                    selectedImageUri = null
                                    Toast.makeText(context, "Đã huỷ OCR", Toast.LENGTH_SHORT).show()
                                },
                                colors = ButtonDefaults.outlinedButtonColors(
                                    contentColor = DarkOnSurface.copy(alpha = 0.7f)
                                ),
                                modifier = Modifier.fillMaxWidth(0.6f)
                            ) {
                                Text(
                                    text = "Huỷ",
                                    fontSize = 16.sp,
                                    fontWeight = FontWeight.Bold
                                )
                            }
                        }
                    }
                }
            } else {
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
                                fontSize = AppTextSize.titleSmall,
                                fontWeight = FontWeight.Bold,
                                color = DarkPrimary
                            )

                        Spacer(modifier = Modifier.height(16.dp))

                        OutlinedTextField(
                            value = name,
                            onValueChange = { 
                                name = it
                                nameError = null // Clear error khi user gõ
                            },
                            label = { Text("Tên đơn thuốc", color = DarkOnSurface.copy(alpha = 0.7f)) },
                            placeholder = { Text("VD: Đơn thuốc tháng 10/2025") },
                            singleLine = true,
                            isError = nameError != null,
                            supportingText = {
                                nameError?.let {
                                    Text(text = it, color = MaterialTheme.colorScheme.error)
                                }
                            },
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = DarkPrimary,
                                unfocusedBorderColor = DarkOnSurface.copy(alpha = 0.3f),
                                focusedTextColor = DarkOnSurface,
                                unfocusedTextColor = DarkOnSurface,
                                errorBorderColor = MaterialTheme.colorScheme.error
                            ),
                            modifier = Modifier.fillMaxWidth()
                        )

                        Spacer(modifier = Modifier.height(12.dp))

                        OutlinedTextField(
                            value = description,
                            onValueChange = { 
                                description = it
                                descriptionError = null // Clear error khi user gõ
                            },
                            label = { Text("Mô tả", color = DarkOnSurface.copy(alpha = 0.7f)) },
                            placeholder = { Text("Ghi chú về đơn thuốc") },
                            minLines = 3,
                            maxLines = 5,
                            isError = descriptionError != null,
                            supportingText = {
                                descriptionError?.let {
                                    Text(text = it, color = MaterialTheme.colorScheme.error)
                                }
                            },
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = DarkPrimary,
                                unfocusedBorderColor = DarkOnSurface.copy(alpha = 0.3f),
                                focusedTextColor = DarkOnSurface,
                                unfocusedTextColor = DarkOnSurface,
                                errorBorderColor = MaterialTheme.colorScheme.error
                            ),
                            modifier = Modifier.fillMaxWidth()
                        )
                    }
                }

                Spacer(modifier = Modifier.height(16.dp))

                // Card upload ảnh (chỉ hiển thị khi tạo mới)
                if (!isEditMode) {
                    Card(
                        colors = CardDefaults.cardColors(containerColor = DarkSurface.copy(alpha = 0.9f)),
                        shape = RoundedCornerShape(16.dp),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Column(modifier = Modifier.padding(16.dp)) {
                            Text(
                                text = "�️ Đính kèm ảnh (Tùy chọn)",
                                fontSize = AppTextSize.titleSmall,
                                fontWeight = FontWeight.Bold,
                                color = DarkPrimary
                            )
                            
                            Spacer(modifier = Modifier.height(12.dp))
                            
                            // 2 nút: OCR và Chọn ảnh thông thường
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                // Nút 1: OCR (AI tự động điền)
                                Button(
                                    onClick = { ocrImagePickerLauncher.launch("image/*") },
                                    modifier = Modifier
                                        .weight(1f)
                                        .height(70.dp),
                                    colors = ButtonDefaults.buttonColors(containerColor = DarkPrimary),
                                    enabled = !isOcrProcessing
                                ) {
                                    Column(
                                        horizontalAlignment = Alignment.CenterHorizontally,
                                        verticalArrangement = Arrangement.Center
                                    ) {
                                        Text(
                                            text = "OCR",
                                            fontSize = 17.sp,
                                            fontWeight = FontWeight.Bold
                                        )
                                        Spacer(modifier = Modifier.height(4.dp))
                                        Text(
                                            text = "AI điền tự động",
                                            fontSize = 13.sp,
                                            color = DarkOnPrimary.copy(alpha = 0.9f)
                                        )
                                    }
                                }
                                
                                // Nút 2: Chọn ảnh thông thường
                                Button(
                                    onClick = { simpleImagePickerLauncher.launch("image/*") },
                                    modifier = Modifier
                                        .weight(1f)
                                        .height(70.dp),
                                    colors = ButtonDefaults.buttonColors(
                                        containerColor = DarkPrimary.copy(alpha = 0.7f)
                                    ),
                                    enabled = !isOcrProcessing
                                ) {
                                    Column(
                                        horizontalAlignment = Alignment.CenterHorizontally,
                                        verticalArrangement = Arrangement.Center
                                    ) {
                                        Text(
                                            text = "Chọn ảnh",
                                            fontSize = 17.sp,
                                            fontWeight = FontWeight.Bold
                                        )
                                        Spacer(modifier = Modifier.height(4.dp))
                                        Text(
                                            text = "Điền thủ công",
                                            fontSize = 13.sp,
                                            color = DarkOnPrimary.copy(alpha = 0.9f)
                                        )
                                    }
                                }
                            }
                            
                            // Hiển thị preview ảnh đã chọn
                            if (selectedImageUri != null && !isOcrProcessing) {
                                Spacer(modifier = Modifier.height(12.dp))
                                
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Text(
                                        text = "Đã chọn ảnh",
                                        fontSize = 13.sp,
                                        color = DarkPrimary,
                                        fontWeight = FontWeight.Bold
                                    )
                                    TextButton(
                                        onClick = {
                                            selectedImageUri = null
                                            selectedImageFile = null
                                        }
                                    ) {
                                        Text("Xóa", color = AIError, fontSize = 13.sp)
                                    }
                                }
                                
                                Card(
                                    colors = CardDefaults.cardColors(
                                        containerColor = DarkOnSurface.copy(alpha = 0.05f)
                                    ),
                                    shape = RoundedCornerShape(8.dp)
                                ) {
                                    Image(
                                        painter = rememberAsyncImagePainter(selectedImageUri),
                                        contentDescription = "Preview",
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .height(150.dp)
                                            .clip(RoundedCornerShape(8.dp)),
                                        contentScale = ContentScale.Fit
                                    )
                                }
                            }
                        }
                    }
                    
                    Spacer(modifier = Modifier.height(16.dp))
                }

                // Danh sách thuốc - DẠNG BẢNG
                Card(
                    colors = CardDefaults.cardColors(containerColor = DarkSurface.copy(alpha = 0.9f)),
                    shape = RoundedCornerShape(16.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        // Header với nút thêm
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = "Danh sách thuốc (${medications.size})",
                                fontSize = AppTextSize.titleSmall,
                                fontWeight = FontWeight.Bold,
                                color = DarkPrimary
                            )

                            Button(
                                onClick = {
                                    val newMedication = MedicationReminderForm()
                                    medications = medications + newMedication
                                    // Tự động mở dialog edit cho thuốc mới
                                    editingIndex = medications.size - 1  // Fix: index cuối cùng là size - 1
                                    editingMedication = newMedication.copy(
                                        reminderTimes = newMedication.reminderTimes.toMutableList()
                                    )
                                    showEditDialog = true
                                },
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = DarkPrimary
                                ),
                                contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp)
                            ) {
                                Icon(
                                    imageVector = Icons.Default.Add,
                                    contentDescription = "Thêm",
                                    modifier = Modifier.size(16.dp)
                                )
                                Spacer(modifier = Modifier.width(4.dp))
                                Text("Thêm", fontSize = 16.sp)
                            }
                        }

                        Spacer(modifier = Modifier.height(16.dp))

                        // Table Header
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .background(
                                    color = DarkPrimary.copy(alpha = 0.2f),
                                    shape = RoundedCornerShape(8.dp)
                                )
                                .padding(vertical = 8.dp, horizontal = 12.dp),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = "STT",
                                fontSize = 14.sp,
                                fontWeight = FontWeight.Bold,
                                color = DarkOnSurface,
                                modifier = Modifier.width(35.dp)
                            )
                            Text(
                                text = "Tên thuốc",
                                fontSize = 14.sp,
                                fontWeight = FontWeight.Bold,
                                color = DarkOnSurface,
                                modifier = Modifier.weight(1f)
                            )
                            Text(
                                text = "Giờ uống",
                                fontSize = 14.sp,
                                fontWeight = FontWeight.Bold,
                                color = DarkOnSurface,
                                modifier = Modifier.width(80.dp),
                                textAlign = TextAlign.Center
                            )
                            Spacer(modifier = Modifier.width(48.dp)) // Space for action button
                        }

                        Spacer(modifier = Modifier.height(8.dp))

                        // Table Rows
                        medications.forEachIndexed { index, medication ->
                            MedicationTableRow(
                                medication = medication,
                                index = index,
                                onEdit = { 
                                    editingIndex = index
                                    editingMedication = medication.copy(
                                        reminderTimes = medication.reminderTimes.toMutableList()
                                    )
                                    showEditDialog = true
                                    // Clear error khi user mở dialog edit
                                    medicationErrors = medicationErrors - index
                                },
                                onDelete = {
                                    if (medications.size > 1) {
                                        medications = medications.toMutableList().apply {
                                            removeAt(index)
                                        }
                                        // Clear error cho medication bị xóa
                                        medicationErrors = medicationErrors.filterKeys { it != index }
                                    } else {
                                        Toast.makeText(context, "Phải có ít nhất 1 loại thuốc", Toast.LENGTH_SHORT).show()
                                    }
                                },
                                errorMessage = medicationErrors[index]
                            )

                            if (index < medications.size - 1) {
                                Divider(
                                    color = DarkOnSurface.copy(alpha = 0.1f),
                                    modifier = Modifier.padding(vertical = 4.dp)
                                )
                            }
                        }
                    }
                }

                Spacer(modifier = Modifier.height(24.dp))

                // Buttons
                Button(
                    onClick = {
                        // Clear old errors
                        nameError = null
                        descriptionError = null
                        medicationErrors = emptyMap()
                        
                        // Validate chi tiết
                        var hasError = false
                        
                        // Validate name
                        if (name.trim().isEmpty()) {
                            nameError = "Tên đơn thuốc không được để trống"
                            hasError = true
                        } else if (name.trim().length < 3) {
                            nameError = "Tên đơn thuốc phải từ 3 ký tự trở lên"
                            hasError = true
                        } else if (name.length > 200) {
                            nameError = "Tên không được quá 200 ký tự"
                            hasError = true
                        }
                        
                        // Validate description (optional, chỉ validate max length)
                        if (description.length > 1000) {
                            descriptionError = "Mô tả không được quá 1000 ký tự"
                            hasError = true
                        }
                        
                        // Validate medications
                        val medErrors = mutableMapOf<Int, String>()
                        medications.forEachIndexed { index, med ->
                            if (med.name.trim().isEmpty()) {
                                medErrors[index] = "Tên thuốc không được để trống"
                                hasError = true
                            }
                            if (med.reminderTimes.isEmpty()) {
                                medErrors[index] = medErrors[index]?.let { "$it; Phải có ít nhất 1 giờ" } ?: "Phải có ít nhất 1 giờ nhắc"
                                hasError = true
                            }
                        }
                        medicationErrors = medErrors
                        
                        if (hasError) {
                            Toast.makeText(context, "Vui lòng kiểm tra lại thông tin", Toast.LENGTH_LONG).show()
                            return@Button
                        }

                        isLoading = true
                        scope.launch {
                            val result = if (isEditMode && editPrescriptionId != null) {
                                // Cập nhật đơn thuốc
                                prescriptionService.updatePrescription(
                                    prescriptionId = editPrescriptionId,
                                    name = name,
                                    description = description,
                                    imageUrl = imageUrl,
                                    medications = medications,
                                    accessToken = accessToken,
                                    elderUserId = elderUserId
                                )
                            } else {
                                // Tạo mới đơn thuốc
                                if (selectedImageFile != null) {
                                    // Có ảnh → Dùng API /create-with-image (Lazy Upload)
                                    prescriptionService.createPrescriptionWithImage(
                                        name = name,
                                        description = description,
                                        imageFile = selectedImageFile!!,
                                        medications = medications,
                                        accessToken = accessToken
                                        , elderUserId = elderUserId
                                    )
                                } else {
                                    // Không có ảnh → Dùng API /create thông thường
                                    prescriptionService.createPrescription(
                                        name = name,
                                        description = description,
                                        imageUrl = imageUrl,
                                        medications = medications,
                                        accessToken = accessToken
                                        , elderUserId = elderUserId
                                    )
                                }
                            }
                            
                            result.fold(
                                onSuccess = { response ->
                                    Toast.makeText(context, "${response.message}", Toast.LENGTH_SHORT).show()
                                    onSuccess()
                                },
                                onFailure = { error ->
                                    Toast.makeText(context, "${error.message}", Toast.LENGTH_LONG).show()
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
                        Text(
                            if (isEditMode) "Cập nhật đơn thuốc" else "Tạo đơn thuốc",
                            fontSize = AppTextSize.labelLarge,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }

                Spacer(modifier = Modifier.height(16.dp))
                }
            }
        }
    }

    // Show edit dialog when needed
    if (showEditDialog && editingMedication != null && editingIndex != null) {
        MedicationEditDialog(
            medication = editingMedication!!,
            onDismiss = { 
                showEditDialog = false
                editingMedication = null
                editingIndex = null
            },
            onConfirm = { updatedMedication ->
                medications = medications.toMutableList().apply {
                    set(editingIndex!!, updatedMedication)
                }
                showEditDialog = false
                editingMedication = null
                editingIndex = null
            }
        )
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
                fontSize = AppTextSize.bodyMedium,
                fontWeight = FontWeight.Bold,
                color = DarkOnSurface
            )

            IconButton(onClick = onDelete) {
                Icon(
                    imageVector = Icons.Default.Delete,
                    contentDescription = "Xóa",
                    tint = AIError
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

        // Reminder times
        Text(
            text = "⏰ Giờ nhắc nhở:",
            fontSize = AppTextSize.bodyMedium,
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

// COMPONENT MỚI: Table Row hiển thị tóm tắt thuốc
@Composable
fun MedicationTableRow(
    medication: MedicationReminderForm,
    index: Int,
    onEdit: () -> Unit,
    onDelete: () -> Unit,
    errorMessage: String? = null
) {
    Column(
        modifier = Modifier.fillMaxWidth()
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .clickable(onClick = onEdit)
                .padding(vertical = 12.dp, horizontal = 12.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            // STT
            Text(
                text = "${index + 1}",
                fontSize = 14.sp,
                fontWeight = FontWeight.Bold,
                color = if (errorMessage != null) MaterialTheme.colorScheme.error else DarkOnSurface,
                modifier = Modifier.width(35.dp)
            )

            // Tên thuốc + Ghi chú
            Column(
                modifier = Modifier.weight(1f)
            ) {
                Text(
                    text = medication.name.ifBlank { "Chưa đặt tên" },
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Medium,
                    color = if (errorMessage != null) {
                        MaterialTheme.colorScheme.error
                    } else if (medication.name.isBlank()) {
                        DarkOnSurface.copy(alpha = 0.4f)
                    } else {
                        DarkOnSurface
                    },
                    maxLines = 1
                )
                if (medication.description.isNotBlank()) {
                    Text(
                        text = medication.description,
                        fontSize = 12.sp,
                        color = DarkOnSurface.copy(alpha = 0.6f),
                        maxLines = 1
                    )
                }
            }

            // Giờ uống - Hiển thị theo chiều dọc
            Column(
                modifier = Modifier.width(80.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                medication.reminderTimes.take(3).forEach { time ->
                    Text(
                        text = time,
                        fontSize = 12.sp,
                        color = DarkPrimary,
                        textAlign = TextAlign.Center
                    )
                }
                if (medication.reminderTimes.size > 3) {
                    Text(
                        text = "+${medication.reminderTimes.size - 3}",
                        fontSize = 11.sp,
                        color = DarkPrimary.copy(alpha = 0.7f),
                        textAlign = TextAlign.Center
                    )
                }
            }

            // Nút xóa
            IconButton(
                onClick = onDelete,
                modifier = Modifier.size(36.dp)
            ) {
                Icon(
                    imageVector = Icons.Default.Delete,
                    contentDescription = "Xóa",
                    tint = AIError,
                    modifier = Modifier.size(20.dp)
                )
            }
        }
        
        // Error message
        errorMessage?.let {
            Text(
                text = it,
                fontSize = 12.sp,
                color = MaterialTheme.colorScheme.error,
                modifier = Modifier.padding(horizontal = 12.dp, vertical = 4.dp)
            )
        }
    }
}

@Composable
private fun DaysOfWeekSelector(
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
                label = { 
                    Text(
                        text = day, 
                        fontSize = 11.sp,
                        modifier = Modifier.fillMaxWidth(),
                        textAlign = TextAlign.Center
                    ) 
                },
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

@Composable
fun MedicationEditDialog(
    medication: MedicationReminderForm,
    onDismiss: () -> Unit,
    onConfirm: (MedicationReminderForm) -> Unit
) {
    var editedMedication by remember { mutableStateOf(medication) }
    var showTimePickerDialog by remember { mutableStateOf(false) }

    Dialog(onDismissRequest = onDismiss) {
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(max = 600.dp),
            colors = CardDefaults.cardColors(containerColor = DarkSurface)
        ) {
            Column(
                modifier = Modifier
                    .verticalScroll(rememberScrollState())
                    .padding(16.dp)
            ) {
                // Title
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Chỉnh sửa thuốc",
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold,
                        color = DarkOnSurface
                    )
                    IconButton(onClick = onDismiss) {
                        Icon(
                            imageVector = Icons.Default.Close,
                            contentDescription = "Đóng",
                            tint = DarkOnSurface
                        )
                    }
                }

                Spacer(modifier = Modifier.height(16.dp))

                // Name field
                OutlinedTextField(
                    value = editedMedication.name,
                    onValueChange = { editedMedication = editedMedication.copy(name = it) },
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

                // Description field
                OutlinedTextField(
                    value = editedMedication.description,
                    onValueChange = { editedMedication = editedMedication.copy(description = it) },
                    label = { Text("Ghi chú", color = DarkOnSurface.copy(alpha = 0.7f)) },
                    placeholder = { Text("VD: Uống 1 viên sau ăn") },
                    singleLine = true,
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = DarkPrimary,
                        unfocusedBorderColor = DarkOnSurface.copy(alpha = 0.3f),
                        focusedTextColor = DarkOnSurface,
                        unfocusedTextColor = DarkOnSurface
                    ),
                    modifier = Modifier.fillMaxWidth()
                )

                Spacer(modifier = Modifier.height(16.dp))

                // Days of week selector
                Text(
                    text = "Ngày trong tuần",
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold,
                    color = DarkOnSurface
                )
                Spacer(modifier = Modifier.height(8.dp))
                DaysOfWeekSelector(
                    selectedDays = editedMedication.daysOfWeek,
                    onDaysChange = { editedMedication = editedMedication.copy(daysOfWeek = it) }
                )

                Spacer(modifier = Modifier.height(20.dp))

                // Reminder times section
                Text(
                    text = "Giờ nhắc nhở",
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold,
                    color = DarkOnSurface
                )

                Spacer(modifier = Modifier.height(8.dp))

                // List of reminder times
                    editedMedication.reminderTimes.forEachIndexed { index, time ->
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 4.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = time,
                            fontSize = 16.sp,
                            color = DarkOnSurface
                        )
                        IconButton(
                            onClick = {
                                editedMedication = editedMedication.copy(
                                    reminderTimes = editedMedication.reminderTimes.toMutableList().apply {
                                        removeAt(index)
                                    }
                                )
                            }
                        ) {
                            Icon(
                                imageVector = Icons.Default.Delete,
                                contentDescription = "Xóa giờ",
                                tint = AIError
                            )
                        }
                    }
                }

                Spacer(modifier = Modifier.height(8.dp))

                // Add time button
                Button(
                    onClick = { showTimePickerDialog = true },
                    modifier = Modifier.fillMaxWidth(),
                    colors = ButtonDefaults.buttonColors(containerColor = DarkPrimary.copy(alpha = 0.1f))
                ) {
                    Icon(
                        imageVector = Icons.Default.Add,
                        contentDescription = "Thêm giờ",
                        tint = DarkPrimary
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Thêm giờ nhắc", color = DarkPrimary)
                }

                Spacer(modifier = Modifier.height(24.dp))

                // Action buttons
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    OutlinedButton(
                        onClick = onDismiss,
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.outlinedButtonColors(
                            contentColor = DarkOnSurface
                        )
                    ) {
                        Text("Hủy")
                    }

                    Button(
                        onClick = { onConfirm(editedMedication) },
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(containerColor = DarkPrimary),
                        enabled = editedMedication.name.isNotBlank() && editedMedication.reminderTimes.isNotEmpty()
                    ) {
                        Text("Lưu")
                    }
                }
            }
        }
    }

    // Time Picker Dialog
    if (showTimePickerDialog) {
        TimePickerDialog(
            onDismiss = { showTimePickerDialog = false },
            onTimeSelected = { time ->
                if (!editedMedication.reminderTimes.contains(time)) {
                    editedMedication = editedMedication.copy(
                        reminderTimes = (editedMedication.reminderTimes + time).sorted().toMutableList()
                    )
                }
                showTimePickerDialog = false
            }
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TimePickerDialog(
    onDismiss: () -> Unit,
    onTimeSelected: (String) -> Unit
) {
    var selectedHour by remember { mutableIntStateOf(8) }
    var selectedMinute by remember { mutableIntStateOf(0) }

    Dialog(onDismissRequest = onDismiss) {
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            colors = CardDefaults.cardColors(containerColor = DarkSurface)
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = "Chọn giờ nhắc nhở",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    color = DarkOnSurface
                )

                Spacer(modifier = Modifier.height(16.dp))

                // Time display
                Text(
                    text = String.format("%02d:%02d", selectedHour, selectedMinute),
                    fontSize = 48.sp,
                    fontWeight = FontWeight.Bold,
                    color = DarkPrimary
                )

                Spacer(modifier = Modifier.height(16.dp))

                // Hour and Minute pickers
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceEvenly
                ) {
                    // Hour picker
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text("Giờ", fontSize = 12.sp, color = DarkOnSurface.copy(alpha = 0.6f))
                        Spacer(modifier = Modifier.height(8.dp))
                        
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            IconButton(onClick = { 
                                selectedHour = if (selectedHour > 0) selectedHour - 1 else 23 
                            }) {
                                Text("▲", fontSize = 20.sp, color = DarkPrimary)
                            }
                        }
                        
                        Text(
                            text = String.format("%02d", selectedHour),
                            fontSize = 32.sp,
                            fontWeight = FontWeight.Bold,
                            color = DarkOnSurface
                        )
                        
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            IconButton(onClick = { 
                                selectedHour = if (selectedHour < 23) selectedHour + 1 else 0 
                            }) {
                                Text("▼", fontSize = 20.sp, color = DarkPrimary)
                            }
                        }
                    }

                    Text(":", fontSize = 32.sp, fontWeight = FontWeight.Bold, color = DarkOnSurface)

                    // Minute picker
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text("Phút", fontSize = 12.sp, color = DarkOnSurface.copy(alpha = 0.6f))
                        Spacer(modifier = Modifier.height(8.dp))
                        
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            IconButton(onClick = { 
                                selectedMinute = if (selectedMinute > 0) selectedMinute - 5 else 55 
                            }) {
                                Text("▲", fontSize = 20.sp, color = DarkPrimary)
                            }
                        }
                        
                        Text(
                            text = String.format("%02d", selectedMinute),
                            fontSize = 32.sp,
                            fontWeight = FontWeight.Bold,
                            color = DarkOnSurface
                        )
                        
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            IconButton(onClick = { 
                                selectedMinute = if (selectedMinute < 55) selectedMinute + 5 else 0 
                            }) {
                                Text("▼", fontSize = 20.sp, color = DarkPrimary)
                            }
                        }
                    }
                }

                Spacer(modifier = Modifier.height(24.dp))

                // Action buttons
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    OutlinedButton(
                        onClick = onDismiss,
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.outlinedButtonColors(
                            contentColor = DarkOnSurface
                        )
                    ) {
                        Text("Hủy")
                    }

                    Button(
                        onClick = { 
                            val timeString = String.format("%02d:%02d", selectedHour, selectedMinute)
                            onTimeSelected(timeString)
                        },
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(containerColor = DarkPrimary)
                    ) {
                        Text("Xác nhận")
                    }
                }
            }
        }
    }
}
