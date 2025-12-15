package com.auto_fe.auto_fe.ui.screens

import android.net.Uri
import android.util.Log
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import coil.compose.AsyncImage
import coil.compose.rememberAsyncImagePainter
import com.auto_fe.auto_fe.ui.service.MedicalDocumentData
import com.auto_fe.auto_fe.ui.service.MedicalDocumentFileData
import com.auto_fe.auto_fe.ui.service.MedicalDocumentService
import com.auto_fe.auto_fe.ui.theme.*
import kotlinx.coroutines.launch
import java.io.File
import java.io.FileOutputStream

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MedicalDocumentsScreen(
    accessToken: String,
    onBackClick: () -> Unit = {}
) {
    val context = LocalContext.current
    val service = remember { MedicalDocumentService() }
    val coroutineScope = rememberCoroutineScope()
    
    var documents by remember { mutableStateOf<List<MedicalDocumentData>>(emptyList()) }
    var isLoading by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var showCreateDialog by remember { mutableStateOf(false) }
    var selectedDocument by remember { mutableStateOf<MedicalDocumentData?>(null) }
    
    // Load documents
    fun loadDocuments() {
        coroutineScope.launch {
            isLoading = true
            errorMessage = null
            
            service.getDocuments(accessToken).fold(
                onSuccess = { docs ->
                    documents = docs
                    isLoading = false
                },
                onFailure = { error ->
                    errorMessage = error.message
                    isLoading = false
                }
            )
        }
    }
    
    LaunchedEffect(Unit) {
        loadDocuments()
    }
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Hồ sơ bệnh án") },
                navigationIcon = {
                    IconButton(onClick = onBackClick) {
                        Icon(Icons.Default.ArrowBack, "Quay lại")
                    }
                },
                actions = {
                    IconButton(onClick = { showCreateDialog = true }) {
                        Icon(Icons.Default.Add, "Thêm tài liệu", tint = DarkPrimary)
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = DarkSurface,
                    titleContentColor = DarkOnSurface
                )
            )
        },
        containerColor = DarkBackground
    ) { paddingValues ->
        when {
            isLoading -> {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(paddingValues),
                    contentAlignment = Alignment.Center
                ) {
                    CircularProgressIndicator(color = DarkPrimary)
                }
            }
            
            errorMessage != null -> {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(paddingValues),
                    contentAlignment = Alignment.Center
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Icon(
                            Icons.Default.Warning,
                            contentDescription = null,
                            modifier = Modifier.size(64.dp),
                            tint = Color.Red
                        )
                        Spacer(Modifier.height(16.dp))
                        Text(errorMessage ?: "Lỗi không xác định", color = DarkOnSurface)
                        Spacer(Modifier.height(16.dp))
                        Button(onClick = { loadDocuments() }) {
                            Text("Thử lại")
                        }
                    }
                }
            }
            
            documents.isEmpty() -> {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(paddingValues),
                    contentAlignment = Alignment.Center
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Icon(
                            Icons.Default.Description,
                            contentDescription = null,
                            modifier = Modifier.size(64.dp),
                            tint = Color.Gray
                        )
                        Spacer(Modifier.height(16.dp))
                        Text("Chưa có tài liệu y tế", color = Color.Gray)
                        Spacer(Modifier.height(16.dp))
                        Button(onClick = { showCreateDialog = true }) {
                            Icon(Icons.Default.Add, contentDescription = null)
                            Spacer(Modifier.width(8.dp))
                            Text("Thêm tài liệu")
                        }
                    }
                }
            }
            
            else -> {
                LazyColumn(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(paddingValues)
                        .padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    items(documents) { document ->
                        DocumentCard(
                            document = document,
                            onClick = { selectedDocument = document },
                            onDelete = {
                                coroutineScope.launch {
                                    service.deleteDocument(accessToken, document.id).fold(
                                        onSuccess = {
                                            Toast.makeText(context, "Đã xóa tài liệu", Toast.LENGTH_SHORT).show()
                                            loadDocuments()
                                        },
                                        onFailure = { error ->
                                            Toast.makeText(context, error.message, Toast.LENGTH_SHORT).show()
                                        }
                                    )
                                }
                            }
                        )
                    }
                }
            }
        }
    }
    
    // Create document dialog
    if (showCreateDialog) {
        CreateDocumentDialog(
            onDismiss = { showCreateDialog = false },
            onConfirm = { name, description ->
                coroutineScope.launch {
                    service.createDocument(accessToken, name, description).fold(
                        onSuccess = {
                            Toast.makeText(context, "Tạo tài liệu thành công", Toast.LENGTH_SHORT).show()
                            showCreateDialog = false
                            loadDocuments()
                        },
                        onFailure = { error ->
                            Toast.makeText(context, error.message, Toast.LENGTH_SHORT).show()
                        }
                    )
                }
            }
        )
    }
    
    // Document detail dialog
    selectedDocument?.let { document ->
        DocumentDetailDialog(
            document = document,
            accessToken = accessToken,
            service = service,
            onDismiss = { 
                selectedDocument = null
                loadDocuments()
            }
        )
    }
}

@Composable
fun DocumentCard(
    document: MedicalDocumentData,
    onClick: () -> Unit,
    onDelete: () -> Unit
) {
    var showDeleteDialog by remember { mutableStateOf(false) }
    
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onClick() },
        colors = CardDefaults.cardColors(
            containerColor = DarkSurface
        ),
        shape = RoundedCornerShape(20.dp),
        elevation = CardDefaults.cardElevation(
            defaultElevation = 3.dp,
            pressedElevation = 6.dp
        )
    ) {
        Box {
            Column(
                modifier = Modifier.padding(20.dp)
            ) {
                // Header with icon and delete button
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.Top
                ) {
                    Row(
                        modifier = Modifier.weight(1f),
                        verticalAlignment = Alignment.Top
                    ) {
                        // Gradient icon box
                        Box(
                            modifier = Modifier
                                .size(72.dp)
                                .clip(RoundedCornerShape(16.dp))
                                .background(
                                    androidx.compose.ui.graphics.Brush.verticalGradient(
                                        colors = listOf(
                                            DarkPrimary.copy(alpha = 0.2f),
                                            DarkPrimary.copy(alpha = 0.05f)
                                        )
                                    )
                                )
                                .border(
                                    width = 1.5.dp,
                                    color = DarkPrimary.copy(alpha = 0.3f),
                                    shape = RoundedCornerShape(16.dp)
                                ),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(
                                Icons.Default.Description,
                                contentDescription = null,
                                tint = DarkPrimary,
                                modifier = Modifier.size(36.dp)
                            )
                        }
                        
                        Spacer(Modifier.width(16.dp))
                        
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = document.name,
                                fontWeight = FontWeight.Bold,
                                color = DarkOnSurface,
                                fontSize = AppTextSize.bodyLarge,
                                maxLines = 2,
                                overflow = TextOverflow.Ellipsis,
                                lineHeight = AppTextSize.bodyLarge * 1.3
                            )
                            
                            Spacer(Modifier.height(8.dp))
                            
                            // File count badge
                            Surface(
                                shape = RoundedCornerShape(8.dp),
                                color = DarkPrimary.copy(alpha = 0.15f)
                            ) {
                                Row(
                                    modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Icon(
                                        Icons.Default.AttachFile,
                                        contentDescription = null,
                                        tint = DarkPrimary,
                                        modifier = Modifier.size(16.dp)
                                    )
                                    Spacer(Modifier.width(4.dp))
                                    Text(
                                        text = "${document.fileCount} file",
                                        fontSize = AppTextSize.bodySmall,
                                        color = DarkPrimary,
                                        fontWeight = FontWeight.SemiBold
                                    )
                                }
                            }
                        }
                    }
                    
                    // Delete button
                    Surface(
                        shape = CircleShape,
                        color = Color.Red.copy(alpha = 0.1f),
                        modifier = Modifier.size(36.dp)
                    ) {
                        IconButton(
                            onClick = { showDeleteDialog = true },
                            modifier = Modifier.size(36.dp)
                        ) {
                            Icon(
                                Icons.Default.Delete,
                                "Xóa",
                                tint = Color.Red,
                                modifier = Modifier.size(18.dp)
                            )
                        }
                    }
                }
                
                // Description
                if (document.description.isNotBlank()) {
                    Spacer(Modifier.height(16.dp))
                    Surface(
                        shape = RoundedCornerShape(12.dp),
                        color = DarkBackground.copy(alpha = 0.5f)
                    ) {
                        Row(
                            modifier = Modifier.padding(12.dp),
                            verticalAlignment = Alignment.Top
                        ) {
                            Icon(
                                Icons.Default.Info,
                                contentDescription = null,
                                tint = DarkOnSurface.copy(alpha = 0.6f),
                                modifier = Modifier.size(16.dp)
                            )
                            Spacer(Modifier.width(8.dp))
                            Text(
                                text = document.description,
                                fontSize = AppTextSize.bodySmall,
                                color = DarkOnSurface.copy(alpha = 0.8f),
                                maxLines = 3,
                                overflow = TextOverflow.Ellipsis,
                                lineHeight = AppTextSize.bodySmall * 1.5,
                                modifier = Modifier.weight(1f)
                            )
                        }
                    }
                }
                
                Spacer(Modifier.height(16.dp))
                
                // Footer with date and action
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    // Date badge
                    Surface(
                        shape = RoundedCornerShape(8.dp),
                        color = DarkBackground.copy(alpha = 0.3f)
                    ) {
                        Row(
                            modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Icon(
                                Icons.Default.CalendarToday,
                                contentDescription = null,
                                tint = Color.Gray,
                                modifier = Modifier.size(14.dp)
                            )
                            Spacer(Modifier.width(6.dp))
                            Text(
                                text = document.createdAt.take(10),
                                fontSize = AppTextSize.bodySmall,
                                color = Color.Gray,
                                fontWeight = FontWeight.Medium
                            )
                        }
                    }
                    
                    // View detail button
                    Surface(
                        shape = RoundedCornerShape(8.dp),
                        color = DarkPrimary.copy(alpha = 0.1f)
                    ) {
                        Row(
                            modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = "Xem chi tiết",
                                fontSize = AppTextSize.bodySmall,
                                color = DarkPrimary,
                                fontWeight = FontWeight.Bold
                            )
                            Spacer(Modifier.width(4.dp))
                            Icon(
                                Icons.Default.ArrowForward,
                                contentDescription = null,
                                tint = DarkPrimary,
                                modifier = Modifier.size(16.dp)
                            )
                        }
                    }
                }
            }
        }
    }
    
    if (showDeleteDialog) {
        AlertDialog(
            onDismissRequest = { showDeleteDialog = false },
            title = { Text("Xác nhận xóa") },
            text = { Text("Bạn có chắc muốn xóa tài liệu này và tất cả file đính kèm?") },
            confirmButton = {
                TextButton(
                    onClick = {
                        onDelete()
                        showDeleteDialog = false
                    }
                ) {
                    Text("Xóa", color = Color.Red)
                }
            },
            dismissButton = {
                TextButton(onClick = { showDeleteDialog = false }) {
                    Text("Hủy")
                }
            }
        )
    }
}

@Composable
fun CreateDocumentDialog(
    onDismiss: () -> Unit,
    onConfirm: (String, String) -> Unit
) {
    var name by remember { mutableStateOf("") }
    var description by remember { mutableStateOf("") }
    
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Tạo tài liệu mới") },
        text = {
            Column {
                OutlinedTextField(
                    value = name,
                    onValueChange = { name = it },
                    label = { Text("Tên tài liệu") },
                    placeholder = { Text("VD: Kết quả xét nghiệm máu") },
                    modifier = Modifier.fillMaxWidth()
                )
                Spacer(Modifier.height(8.dp))
                OutlinedTextField(
                    value = description,
                    onValueChange = { description = it },
                    label = { Text("Mô tả") },
                    placeholder = { Text("VD: Xét nghiệm ngày 15/12/2025") },
                    modifier = Modifier.fillMaxWidth(),
                    minLines = 3
                )
            }
        },
        confirmButton = {
            TextButton(
                onClick = { onConfirm(name, description) },
                enabled = name.isNotBlank()
            ) {
                Text("Tạo")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Hủy")
            }
        }
    )
}

@Composable
fun DocumentDetailDialog(
    document: MedicalDocumentData,
    accessToken: String,
    service: MedicalDocumentService,
    onDismiss: () -> Unit
) {
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()
    var files by remember { mutableStateOf(document.files) }
    var isUploading by remember { mutableStateOf(false) }
    var showFileTypeDialog by remember { mutableStateOf(false) }
    var selectedImageUrl by remember { mutableStateOf<String?>(null) }
    
    val imagePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        if (uri == null) {
            Toast.makeText(context, "Không có file được chọn", Toast.LENGTH_SHORT).show()
            return@rememberLauncherForActivityResult
        }
        
        coroutineScope.launch {
            try {
                isUploading = true
                
                val fileType = context.contentResolver.getType(uri) ?: "image/jpeg"
                val fileName = getFileName(context, uri) ?: "image.jpg"
                val file = uriToFile(context, uri, fileName)
                
                // Check file size (max 10MB)
                if (file.length() > 10 * 1024 * 1024) {
                    Toast.makeText(context, "File quá lớn (tối đa 10MB)", Toast.LENGTH_SHORT).show()
                    isUploading = false
                    file.delete()
                    return@launch
                }
                
                android.util.Log.d("MedicalDocumentsScreen", "Image picker - fileType: $fileType, fileName: $fileName")
                
                service.uploadFile(accessToken, document.id, file, fileType).fold(
                    onSuccess = { fileData ->
                        files = files + fileData
                        Toast.makeText(context, "Upload thành công: $fileName", Toast.LENGTH_SHORT).show()
                    },
                    onFailure = { error ->
                        Toast.makeText(context, "Lỗi upload: ${error.message}", Toast.LENGTH_LONG).show()
                    }
                )
                
                isUploading = false
                file.delete()
            } catch (e: Exception) {
                Toast.makeText(context, "Lỗi: ${e.message}", Toast.LENGTH_LONG).show()
                isUploading = false
            }
        }
    }
    
    val pdfPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        if (uri == null) {
            Toast.makeText(context, "Không có file được chọn", Toast.LENGTH_SHORT).show()
            return@rememberLauncherForActivityResult
        }
        
        coroutineScope.launch {
            try {
                isUploading = true
                
                val fileType = context.contentResolver.getType(uri) ?: "application/pdf"
                val fileName = getFileName(context, uri) ?: "document.pdf"
                val file = uriToFile(context, uri, fileName)
                
                Log.d("MedicalDocuments", "PDF picker - fileType: $fileType, fileName: $fileName")
                
                // Check file size (max 10MB)
                if (file.length() > 10 * 1024 * 1024) {
                    Toast.makeText(context, "File quá lớn (tối đa 10MB)", Toast.LENGTH_SHORT).show()
                    isUploading = false
                    file.delete()
                    return@launch
                }
                
                service.uploadFile(accessToken, document.id, file, fileType).fold(
                    onSuccess = { fileData ->
                        files = files + fileData
                        Toast.makeText(context, "Upload thành công: $fileName", Toast.LENGTH_SHORT).show()
                    },
                    onFailure = { error ->
                        Toast.makeText(context, "Lỗi upload: ${error.message}", Toast.LENGTH_LONG).show()
                    }
                )
                
                isUploading = false
                file.delete()
            } catch (e: Exception) {
                Toast.makeText(context, "Lỗi: ${e.message}", Toast.LENGTH_LONG).show()
                isUploading = false
            }
        }
    }
    
    Dialog(onDismissRequest = onDismiss) {
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .fillMaxHeight(0.8f),
            colors = CardDefaults.cardColors(containerColor = DarkSurface)
        ) {
            Column(
                modifier = Modifier.fillMaxSize()
            ) {
                // Header
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = document.name,
                        fontWeight = FontWeight.Bold,
                        color = DarkOnSurface,
                        modifier = Modifier.weight(1f)
                    )
                    IconButton(onClick = onDismiss) {
                        Icon(Icons.Default.Close, "Đóng", tint = DarkOnSurface)
                    }
                }
                
                Divider(color = DarkOnSurface.copy(alpha = 0.1f))
                
                // Files list
                LazyColumn(
                    modifier = Modifier
                        .weight(1f)
                        .padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    item {
                        Text(
                            text = document.description,
                            color = DarkOnSurface.copy(alpha = 0.7f),
                            fontSize = AppTextSize.bodySmall
                        )
                        Spacer(Modifier.height(16.dp))
                        Text(
                            text = "Tệp đính kèm (${files.size})",
                            fontWeight = FontWeight.Bold,
                            color = DarkOnSurface
                        )
                        Spacer(Modifier.height(8.dp))
                    }
                    
                    if (files.isEmpty()) {
                        item {
                            Box(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(32.dp),
                                contentAlignment = Alignment.Center
                            ) {
                                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                    Icon(
                                        Icons.Default.InsertDriveFile,
                                        contentDescription = null,
                                        modifier = Modifier.size(48.dp),
                                        tint = Color.Gray.copy(alpha = 0.5f)
                                    )
                                    Spacer(Modifier.height(8.dp))
                                    Text(
                                        text = "Chưa có file nào",
                                        color = Color.Gray,
                                        fontSize = AppTextSize.bodySmall
                                    )
                                }
                            }
                        }
                    } else {
                        // Separate images and PDFs
                        val images = files.filter { it.fileType.startsWith("image/") }
                        val pdfs = files.filter { it.fileType == "application/pdf" }
                        
                        // Images grid
                        if (images.isNotEmpty()) {
                            item {
                                Text(
                                    text = "Ảnh (${images.size})",
                                    fontWeight = FontWeight.SemiBold,
                                    color = DarkPrimary,
                                    fontSize = AppTextSize.bodyMedium,
                                    modifier = Modifier.padding(vertical = 8.dp)
                                )
                            }
                            item {
                                LazyVerticalGrid(
                                    columns = GridCells.Fixed(3),
                                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                                    verticalArrangement = Arrangement.spacedBy(8.dp),
                                    modifier = Modifier.height(((images.size / 3 + 1) * 130).dp)
                                ) {
                                    items(images) { file ->
                                        ImageThumbnail(
                                            file = file,
                                            onClick = { selectedImageUrl = file.fileUrl },
                                            onDelete = {
                                                coroutineScope.launch {
                                                    service.deleteFile(accessToken, document.id, file.id).fold(
                                                        onSuccess = {
                                                            files = files.filter { it.id != file.id }
                                                            Toast.makeText(context, "Đã xóa ảnh", Toast.LENGTH_SHORT).show()
                                                        },
                                                        onFailure = { error ->
                                                            Toast.makeText(context, error.message, Toast.LENGTH_SHORT).show()
                                                        }
                                                    )
                                                }
                                            }
                                        )
                                    }
                                }
                            }
                        }
                        
                        // PDFs list
                        if (pdfs.isNotEmpty()) {
                            item {
                                Spacer(Modifier.height(16.dp))
                                Text(
                                    text = "Tài liệu PDF (${pdfs.size})",
                                    fontWeight = FontWeight.SemiBold,
                                    color = Color(0xFFE53935),
                                    fontSize = AppTextSize.bodyMedium,
                                    modifier = Modifier.padding(vertical = 8.dp)
                                )
                            }
                            items(pdfs) { file ->
                                PdfFileItem(
                                    file = file,
                                    onDownload = {
                                        downloadFile(context, file.fileUrl, file.fileName)
                                    },
                                    onDelete = {
                                        coroutineScope.launch {
                                            service.deleteFile(accessToken, document.id, file.id).fold(
                                                onSuccess = {
                                                    files = files.filter { it.id != file.id }
                                                    Toast.makeText(context, "Đã xóa PDF", Toast.LENGTH_SHORT).show()
                                                },
                                                onFailure = { error ->
                                                    Toast.makeText(context, error.message, Toast.LENGTH_SHORT).show()
                                                }
                                            )
                                        }
                                    }
                                )
                            }
                        }
                    }
                }
                
                Divider(
                    color = DarkOnSurface.copy(alpha = 0.1f),
                    thickness = 1.dp
                )
                
                // Upload buttons section
                if (isUploading) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(24.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Column(
                            horizontalAlignment = Alignment.CenterHorizontally,
                            verticalArrangement = Arrangement.spacedBy(12.dp)
                        ) {
                            CircularProgressIndicator(
                                modifier = Modifier.size(32.dp),
                                color = DarkPrimary,
                                strokeWidth = 3.dp
                            )
                            Text(
                                text = "Đang upload...",
                                color = DarkOnSurface,
                                fontWeight = FontWeight.Medium
                            )
                        }
                    }
                } else {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        // Upload Image button
                        Button(
                            onClick = { imagePickerLauncher.launch("image/*") },
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(56.dp),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = DarkPrimary
                            ),
                            shape = RoundedCornerShape(12.dp),
                            elevation = ButtonDefaults.buttonElevation(
                                defaultElevation = 2.dp,
                                pressedElevation = 4.dp
                            )
                        ) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.Center
                            ) {
                                Icon(
                                    Icons.Default.Image,
                                    contentDescription = null,
                                    modifier = Modifier.size(24.dp)
                                )
                                Spacer(Modifier.width(12.dp))
                                Column {
                                    Text(
                                        text = "Thêm ảnh",
                                        fontWeight = FontWeight.Bold,
                                        fontSize = AppTextSize.bodyLarge
                                    )
                                    Text(
                                        text = "JPG, PNG, WebP",
                                        fontSize = AppTextSize.bodySmall,
                                        color = Color.White.copy(alpha = 0.8f)
                                    )
                                }
                            }
                        }
                        
                        // Upload PDF button
                        Button(
                            onClick = { pdfPickerLauncher.launch("application/pdf") },
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(56.dp),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = Color(0xFFE53935)
                            ),
                            shape = RoundedCornerShape(12.dp),
                            elevation = ButtonDefaults.buttonElevation(
                                defaultElevation = 2.dp,
                                pressedElevation = 4.dp
                            )
                        ) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.Center
                            ) {
                                Icon(
                                    Icons.Default.PictureAsPdf,
                                    contentDescription = null,
                                    modifier = Modifier.size(24.dp)
                                )
                                Spacer(Modifier.width(12.dp))
                                Column {
                                    Text(
                                        text = "Thêm PDF",
                                        fontWeight = FontWeight.Bold,
                                        fontSize = AppTextSize.bodyLarge
                                    )
                                    Text(
                                        text = "Tài liệu y tế",
                                        fontSize = AppTextSize.bodySmall,
                                        color = Color.White.copy(alpha = 0.8f)
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Image viewer dialog
    selectedImageUrl?.let { imageUrl ->
        MedicalImageViewerDialog(
            imageUrl = imageUrl,
            onDismiss = { selectedImageUrl = null }
        )
    }
}

@Composable
fun ImageThumbnail(
    file: MedicalDocumentFileData,
    onClick: () -> Unit,
    onDelete: () -> Unit
) {
    var showDeleteDialog by remember { mutableStateOf(false) }
    
    Box(
        modifier = Modifier
            .aspectRatio(1f)
            .clip(RoundedCornerShape(8.dp))
            .background(DarkBackground)
            .clickable { onClick() }
    ) {
        AsyncImage(
            model = file.fileUrl,
            contentDescription = file.fileName,
            modifier = Modifier.fillMaxSize(),
            contentScale = ContentScale.Crop
        )
        
        // Delete button overlay
        Box(
            modifier = Modifier
                .align(Alignment.TopEnd)
                .padding(4.dp)
                .size(24.dp)
                .clip(CircleShape)
                .background(Color.Black.copy(alpha = 0.6f))
                .clickable { showDeleteDialog = true },
            contentAlignment = Alignment.Center
        ) {
            Icon(
                Icons.Default.Close,
                contentDescription = "Xóa",
                tint = Color.White,
                modifier = Modifier.size(16.dp)
            )
        }
    }
    
    if (showDeleteDialog) {
        AlertDialog(
            onDismissRequest = { showDeleteDialog = false },
            title = { Text("Xác nhận xóa") },
            text = { Text("Bạn có chắc muốn xóa ảnh này?") },
            confirmButton = {
                TextButton(
                    onClick = {
                        onDelete()
                        showDeleteDialog = false
                    }
                ) {
                    Text("Xóa", color = Color.Red)
                }
            },
            dismissButton = {
                TextButton(onClick = { showDeleteDialog = false }) {
                    Text("Hủy")
                }
            }
        )
    }
}

@Composable
fun PdfFileItem(
    file: MedicalDocumentFileData,
    onDownload: () -> Unit,
    onDelete: () -> Unit
) {
    var showDeleteDialog by remember { mutableStateOf(false) }
    
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = DarkSurface),
        shape = RoundedCornerShape(16.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Left content - name and info
            Column(modifier = Modifier.weight(1f)) {
                // File name with PDF icon inline
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Icon(
                        Icons.Default.PictureAsPdf,
                        contentDescription = null,
                        tint = Color(0xFFE53935),
                        modifier = Modifier.size(20.dp)
                    )
                    Spacer(Modifier.width(8.dp))
                    Text(
                        text = file.fileName,
                        color = DarkOnSurface,
                        fontSize = AppTextSize.bodyMedium,
                        fontWeight = FontWeight.SemiBold,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                        modifier = Modifier.weight(1f)
                    )
                }
                
                Spacer(Modifier.height(8.dp))
                
                // Size badge and note (no date)
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Surface(
                        shape = RoundedCornerShape(6.dp),
                        color = DarkPrimary.copy(alpha = 0.1f)
                    ) {
                        Text(
                            text = formatFileSize(file.fileSize),
                            color = DarkPrimary,
                            fontSize = AppTextSize.bodySmall,
                            fontWeight = FontWeight.Medium,
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                        )
                    }
                    
                    // Note inline (if exists)
                    file.note?.let {
                        Text(
                            text = "• $it",
                            color = DarkOnSurface.copy(alpha = 0.6f),
                            fontSize = AppTextSize.bodySmall,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                            fontStyle = androidx.compose.ui.text.font.FontStyle.Italic,
                            modifier = Modifier.weight(1f)
                        )
                    }
                }
            }
            
            Spacer(Modifier.width(12.dp))
            
            // Action buttons column
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                // Download button with background
                Surface(
                    shape = CircleShape,
                    color = DarkPrimary.copy(alpha = 0.1f),
                    modifier = Modifier.size(40.dp)
                ) {
                    IconButton(onClick = onDownload) {
                        Icon(
                            Icons.Default.Download,
                            contentDescription = "Tải về",
                            tint = DarkPrimary,
                            modifier = Modifier.size(20.dp)
                        )
                    }
                }
                
                // Delete button with background
                Surface(
                    shape = CircleShape,
                    color = Color.Red.copy(alpha = 0.1f),
                    modifier = Modifier.size(40.dp)
                ) {
                    IconButton(onClick = { showDeleteDialog = true }) {
                        Icon(
                            Icons.Default.Delete,
                            contentDescription = "Xóa",
                            tint = Color.Red,
                            modifier = Modifier.size(20.dp)
                        )
                    }
                }
            }
        }
    }
    
    if (showDeleteDialog) {
        AlertDialog(
            onDismissRequest = { showDeleteDialog = false },
            title = { Text("Xác nhận xóa") },
            text = { Text("Bạn có chắc muốn xóa file PDF này?") },
            confirmButton = {
                TextButton(
                    onClick = {
                        onDelete()
                        showDeleteDialog = false
                    }
                ) {
                    Text("Xóa", color = Color.Red)
                }
            },
            dismissButton = {
                TextButton(onClick = { showDeleteDialog = false }) {
                    Text("Hủy")
                }
            }
        )
    }
}

@Composable
fun MedicalImageViewerDialog(
    imageUrl: String,
    onDismiss: () -> Unit
) {
    var scale by remember { mutableStateOf(1f) }
    var offsetX by remember { mutableStateOf(0f) }
    var offsetY by remember { mutableStateOf(0f) }
    
    Dialog(
        onDismissRequest = onDismiss,
        properties = DialogProperties(usePlatformDefaultWidth = false)
    ) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(Color.Black)
        ) {
            AsyncImage(
                model = imageUrl,
                contentDescription = "Full image",
                modifier = Modifier
                    .fillMaxSize()
                    .pointerInput(Unit) {
                        detectTransformGestures { _, pan, zoom, _ ->
                            scale = (scale * zoom).coerceIn(1f, 5f)
                            
                            if (scale > 1f) {
                                offsetX += pan.x
                                offsetY += pan.y
                                
                                // Limit pan based on scale
                                val maxOffsetX = size.width * (scale - 1) / 2
                                val maxOffsetY = size.height * (scale - 1) / 2
                                offsetX = offsetX.coerceIn(-maxOffsetX, maxOffsetX)
                                offsetY = offsetY.coerceIn(-maxOffsetY, maxOffsetY)
                            } else {
                                // Reset offset when scale is 1
                                offsetX = 0f
                                offsetY = 0f
                            }
                        }
                    }
                    .graphicsLayer(
                        scaleX = scale,
                        scaleY = scale,
                        translationX = offsetX,
                        translationY = offsetY
                    ),
                contentScale = ContentScale.Fit
            )
            
            // Top bar with close button and zoom indicator
            Row(
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(16.dp)
                    .background(Color.Black.copy(alpha = 0.5f), RoundedCornerShape(8.dp))
                    .padding(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                if (scale > 1f) {
                    Text(
                        text = "${(scale * 100).toInt()}%",
                        color = Color.White,
                        fontSize = AppTextSize.bodySmall,
                        modifier = Modifier.padding(end = 8.dp)
                    )
                }
                
                IconButton(
                    onClick = {
                        scale = 1f
                        offsetX = 0f
                        offsetY = 0f
                        onDismiss()
                    }
                ) {
                    Icon(
                        Icons.Default.Close,
                        contentDescription = "Đóng",
                        tint = Color.White,
                        modifier = Modifier.size(28.dp)
                    )
                }
            }
            
            // Reset zoom button (visible when zoomed)
            if (scale > 1f) {
                FloatingActionButton(
                    onClick = {
                        scale = 1f
                        offsetX = 0f
                        offsetY = 0f
                    },
                    modifier = Modifier
                        .align(Alignment.BottomEnd)
                        .padding(16.dp),
                    containerColor = Color.White.copy(alpha = 0.8f)
                ) {
                    Icon(
                        Icons.Default.ZoomOut,
                        contentDescription = "Reset zoom",
                        tint = Color.Black
                    )
                }
            }
        }
    }
}

// Helper functions
private fun formatFileSize(sizeInBytes: Int): String {
    val kb = sizeInBytes / 1024.0
    return if (kb < 1024) {
        String.format("%.1f KB", kb)
    } else {
        val mb = kb / 1024.0
        String.format("%.2f MB", mb)
    }
}

private fun downloadFile(context: android.content.Context, url: String, fileName: String) {
    try {
        val downloadManager = context.getSystemService(android.content.Context.DOWNLOAD_SERVICE) as android.app.DownloadManager
        val request = android.app.DownloadManager.Request(Uri.parse(url))
            .setTitle(fileName)
            .setDescription("Đang tải về tài liệu y tế...")
            .setNotificationVisibility(android.app.DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
            .setDestinationInExternalPublicDir(android.os.Environment.DIRECTORY_DOWNLOADS, "MedicalDocs/$fileName")
            .setAllowedOverMetered(true)
            .setAllowedOverRoaming(true)
            .setMimeType("application/pdf")
        
        val downloadId = downloadManager.enqueue(request)
        Toast.makeText(context, "Đang tải về: $fileName", Toast.LENGTH_SHORT).show()
        
        Log.d("MedicalDocuments", "Download started - ID: $downloadId, File: $fileName")
    } catch (e: Exception) {
        Log.e("MedicalDocuments", "Download error: ${e.message}")
        Toast.makeText(context, "Lỗi tải file: ${e.message}", Toast.LENGTH_LONG).show()
    }
}

private fun getFileName(context: android.content.Context, uri: Uri): String? {
    var fileName: String? = null
    
    if (uri.scheme == "content") {
        val cursor = context.contentResolver.query(uri, null, null, null, null)
        cursor?.use {
            if (it.moveToFirst()) {
                val nameIndex = it.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                if (nameIndex >= 0) {
                    fileName = it.getString(nameIndex)
                }
            }
        }
    }
    
    if (fileName == null) {
        fileName = uri.path?.substringAfterLast('/') ?: "file_${System.currentTimeMillis()}"
    }
    
    return fileName
}

private fun uriToFile(context: android.content.Context, uri: Uri, fileName: String? = null): File {
    val contentResolver = context.contentResolver
    val name = fileName ?: "temp_${System.currentTimeMillis()}"
    val tempFile = File(context.cacheDir, name)
    
    contentResolver.openInputStream(uri)?.use { input ->
        FileOutputStream(tempFile).use { output ->
            input.copyTo(output)
        }
    }
    
    return tempFile
}
