package com.auto_fe.auto_fe.ui.screens

import android.widget.Toast
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import com.auto_fe.auto_fe.service.be.EmergencyContactService
import com.auto_fe.auto_fe.ui.theme.*
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun EmergencyContactScreen(
    accessToken: String,
    onBackClick: () -> Unit = {}
) {
    val context = LocalContext.current
    val emergencyContactService = remember { EmergencyContactService() }
    val coroutineScope = rememberCoroutineScope()
    
    var contacts by remember { mutableStateOf<List<EmergencyContactService.EmergencyContact>>(emptyList()) }
    var isLoading by remember { mutableStateOf(true) }
    var showAddDialog by remember { mutableStateOf(false) }
    var selectedContact by remember { mutableStateOf<EmergencyContactService.EmergencyContact?>(null) }
    var showEditDialog by remember { mutableStateOf(false) }
    var showDeleteDialog by remember { mutableStateOf(false) }
    
    // Load contacts
    fun loadContacts() {
        coroutineScope.launch {
            isLoading = true
            val result = emergencyContactService.getAll(accessToken)
            isLoading = false
            
            result.onSuccess { response ->
                contacts = response.data ?: emptyList()
            }.onFailure { error ->
                Toast.makeText(context, "${error.message}", Toast.LENGTH_LONG).show()
            }
        }
    }
    
    // Load contacts khi màn hình mở
    LaunchedEffect(Unit) {
        loadContacts()
    }
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = { 
                    Text(
                        "Liên hệ khẩn cấp",
                        color = DarkOnSurface,
                        fontWeight = FontWeight.Bold
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
                    containerColor = DarkSurface
                )
            )
        },
        floatingActionButton = {
            FloatingActionButton(
                onClick = { showAddDialog = true },
                containerColor = DarkPrimary,
                contentColor = DarkOnPrimary
            ) {
                Icon(Icons.Default.Add, contentDescription = "Thêm liên hệ")
            }
        },
        containerColor = DarkBackground
    ) { paddingValues ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
        ) {
            if (isLoading) {
                CircularProgressIndicator(
                    modifier = Modifier.align(Alignment.Center),
                    color = DarkPrimary
                )
            } else if (contacts.isEmpty()) {
                // Empty state
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(48.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center
                ) {
                    Surface(
                        shape = RoundedCornerShape(24.dp),
                        color = DarkPrimary.copy(alpha = 0.1f),
                        modifier = Modifier.size(120.dp)
                    ) {
                        Box(contentAlignment = Alignment.Center) {
                            Icon(
                                imageVector = Icons.Default.Person,
                                contentDescription = null,
                                modifier = Modifier.size(64.dp),
                                tint = DarkPrimary.copy(alpha = 0.5f)
                            )
                        }
                    }
                    Spacer(modifier = Modifier.height(24.dp))
                    Text(
                        "Chưa có liên hệ khẩn cấp",
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold,
                        color = DarkOnSurface.copy(alpha = 0.7f)
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        "Thêm thông tin liên hệ để sử dụng trong trường hợp khẩn cấp",
                        style = MaterialTheme.typography.bodyMedium,
                        color = DarkOnSurface.copy(alpha = 0.5f),
                        textAlign = androidx.compose.ui.text.style.TextAlign.Center
                    )
                    Spacer(modifier = Modifier.height(24.dp))
                    Button(
                        onClick = { showAddDialog = true },
                        colors = ButtonDefaults.buttonColors(
                            containerColor = DarkPrimary,
                            contentColor = DarkOnPrimary
                        ),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Add,
                            contentDescription = null,
                            modifier = Modifier.size(20.dp)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Thêm liên hệ", fontWeight = FontWeight.Medium)
                    }
                }
            } else {
                LazyColumn(
                    modifier = Modifier.fillMaxSize(),
                    contentPadding = PaddingValues(16.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    items(contacts) { contact ->
                        EmergencyContactCard(
                            contact = contact,
                            onEditClick = {
                                selectedContact = contact
                                showEditDialog = true
                            },
                            onDeleteClick = {
                                selectedContact = contact
                                showDeleteDialog = true
                            }
                        )
                    }
                }
            }
        }
    }
    
    // Add Dialog
    if (showAddDialog) {
        EmergencyContactDialog(
            contact = null,
            onDismiss = { showAddDialog = false },
            onConfirm = { request ->
                coroutineScope.launch {
                    val result = emergencyContactService.create(accessToken, request)
                    result.onSuccess {
                        Toast.makeText(context, "Thêm liên hệ thành công", Toast.LENGTH_SHORT).show()
                        showAddDialog = false
                        loadContacts()
                    }.onFailure { error ->
                        Toast.makeText(context, "${error.message}", Toast.LENGTH_LONG).show()
                    }
                }
            }
        )
    }
    
    // Edit Dialog
    if (showEditDialog && selectedContact != null) {
        EmergencyContactDialog(
            contact = selectedContact,
            onDismiss = { 
                showEditDialog = false
                selectedContact = null
            },
            onConfirm = { request ->
                coroutineScope.launch {
                    val result = emergencyContactService.update(accessToken, selectedContact!!.id, request)
                    result.onSuccess {
                        Toast.makeText(context, "Cập nhật thành công", Toast.LENGTH_SHORT).show()
                        showEditDialog = false
                        selectedContact = null
                        loadContacts()
                    }.onFailure { error ->
                        Toast.makeText(context, "${error.message}", Toast.LENGTH_LONG).show()
                    }
                }
            }
        )
    }
    
    // Delete Dialog
    if (showDeleteDialog && selectedContact != null) {
        AlertDialog(
            onDismissRequest = { 
                showDeleteDialog = false
                selectedContact = null
            },
            title = { Text("Xác nhận xóa", color = DarkOnSurface) },
            text = { 
                Text(
                    "Bạn có chắc muốn xóa liên hệ \"${selectedContact!!.name}\"?",
                    color = DarkOnSurface.copy(alpha = 0.8f)
                )
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        coroutineScope.launch {
                            val result = emergencyContactService.delete(accessToken, selectedContact!!.id)
                            result.onSuccess {
                                Toast.makeText(context, "Xóa thành công", Toast.LENGTH_SHORT).show()
                                showDeleteDialog = false
                                selectedContact = null
                                loadContacts()
                            }.onFailure { error ->
                                Toast.makeText(context, "${error.message}", Toast.LENGTH_LONG).show()
                            }
                        }
                    }
                ) {
                    Text("Xóa", color = AIError)
                }
            },
            dismissButton = {
                TextButton(
                    onClick = { 
                        showDeleteDialog = false
                        selectedContact = null
                    }
                ) {
                    Text("Hủy", color = DarkOnSurface)
                }
            },
            containerColor = DarkSurface
        )
    }
}

@Composable
fun EmergencyContactCard(
    contact: EmergencyContactService.EmergencyContact,
    onEditClick: () -> Unit,
    onDeleteClick: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onEditClick() },
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(
            containerColor = DarkSurface
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(20.dp)
        ) {
            // Header
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.Top
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = contact.name,
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold,
                        color = DarkOnSurface
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    Surface(
                        shape = RoundedCornerShape(8.dp),
                        color = DarkPrimary.copy(alpha = 0.15f)
                    ) {
                        Text(
                            text = contact.relationship,
                            style = MaterialTheme.typography.bodyMedium,
                            fontWeight = FontWeight.Medium,
                            color = DarkPrimary,
                            modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp)
                        )
                    }
                }
                
                IconButton(
                    onClick = onDeleteClick,
                    colors = IconButtonDefaults.iconButtonColors(
                        contentColor = AIError
                    )
                ) {
                    Icon(
                        imageVector = Icons.Default.Delete,
                        contentDescription = "Xóa",
                        modifier = Modifier.size(22.dp)
                    )
                }
            }
            
            Spacer(modifier = Modifier.height(16.dp))
            Divider(color = DarkOnSurface.copy(alpha = 0.1f))
            Spacer(modifier = Modifier.height(16.dp))
            
            // Phone
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Surface(
                    shape = RoundedCornerShape(10.dp),
                    color = DarkPrimary.copy(alpha = 0.1f),
                    modifier = Modifier.size(40.dp)
                ) {
                    Box(contentAlignment = Alignment.Center) {
                        Icon(
                            imageVector = Icons.Default.Phone,
                            contentDescription = null,
                            tint = DarkPrimary,
                            modifier = Modifier.size(20.dp)
                        )
                    }
                }
                Column {
                    Text(
                        text = "Số điện thoại",
                        style = MaterialTheme.typography.bodySmall,
                        color = DarkOnSurface.copy(alpha = 0.5f)
                    )
                    Text(
                        text = contact.phoneNumber,
                        style = MaterialTheme.typography.bodyLarge,
                        fontWeight = FontWeight.Medium,
                        color = DarkOnSurface
                    )
                }
            }
            
            // Address
            if (!contact.address.isNullOrBlank()) {
                Spacer(modifier = Modifier.height(12.dp))
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Surface(
                        shape = RoundedCornerShape(10.dp),
                        color = DarkOnSurface.copy(alpha = 0.05f),
                        modifier = Modifier.size(40.dp)
                    ) {
                        Box(contentAlignment = Alignment.Center) {
                            Icon(
                                imageVector = Icons.Default.LocationOn,
                                contentDescription = null,
                                tint = DarkOnSurface.copy(alpha = 0.6f),
                                modifier = Modifier.size(20.dp)
                            )
                        }
                    }
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = "Địa chỉ",
                            style = MaterialTheme.typography.bodySmall,
                            color = DarkOnSurface.copy(alpha = 0.5f)
                        )
                        Text(
                            text = contact.address,
                            style = MaterialTheme.typography.bodyMedium,
                            color = DarkOnSurface.copy(alpha = 0.8f)
                        )
                    }
                }
            }
            
            // Note
            if (!contact.note.isNullOrBlank()) {
                Spacer(modifier = Modifier.height(12.dp))
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Surface(
                        shape = RoundedCornerShape(10.dp),
                        color = DarkOnSurface.copy(alpha = 0.05f),
                        modifier = Modifier.size(40.dp)
                    ) {
                        Box(contentAlignment = Alignment.Center) {
                            Icon(
                                imageVector = Icons.Default.Info,
                                contentDescription = null,
                                tint = DarkOnSurface.copy(alpha = 0.6f),
                                modifier = Modifier.size(20.dp)
                            )
                        }
                    }
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = "Ghi chú",
                            style = MaterialTheme.typography.bodySmall,
                            color = DarkOnSurface.copy(alpha = 0.5f)
                        )
                        Text(
                            text = contact.note,
                            style = MaterialTheme.typography.bodyMedium,
                            color = DarkOnSurface.copy(alpha = 0.7f),
                            fontStyle = androidx.compose.ui.text.font.FontStyle.Italic
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun EmergencyContactDialog(
    contact: EmergencyContactService.EmergencyContact?,
    onDismiss: () -> Unit,
    onConfirm: (EmergencyContactService.EmergencyContactRequest) -> Unit
) {
    var name by remember { mutableStateOf(contact?.name ?: "") }
    var phoneNumber by remember { mutableStateOf(contact?.phoneNumber ?: "") }
    var address by remember { mutableStateOf(contact?.address ?: "") }
    var relationship by remember { mutableStateOf(contact?.relationship ?: "") }
    var note by remember { mutableStateOf(contact?.note ?: "") }
    
    var nameError by remember { mutableStateOf(false) }
    var phoneError by remember { mutableStateOf(false) }
    var relationshipError by remember { mutableStateOf(false) }
    
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { 
            Text(
                if (contact == null) "Thêm liên hệ khẩn cấp" else "Sửa liên hệ khẩn cấp",
                color = DarkOnSurface
            )
        },
        text = {
            Column(
                modifier = Modifier.fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                OutlinedTextField(
                    value = name,
                    onValueChange = { 
                        name = it
                        nameError = it.isBlank()
                    },
                    label = { Text("Tên *") },
                    isError = nameError,
                    modifier = Modifier.fillMaxWidth(),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedTextColor = DarkOnSurface,
                        unfocusedTextColor = DarkOnSurface,
                        focusedBorderColor = DarkPrimary,
                        unfocusedBorderColor = DarkOnSurface.copy(alpha = 0.3f),
                        focusedLabelColor = DarkPrimary,
                        unfocusedLabelColor = DarkOnSurface.copy(alpha = 0.6f),
                        cursorColor = DarkPrimary
                    )
                )
                
                OutlinedTextField(
                    value = phoneNumber,
                    onValueChange = { 
                        // Only allow digits, +, and limit length
                        if (it.isEmpty() || it.matches(Regex("^[+0-9]*$"))) {
                            phoneNumber = it
                            phoneError = it.isBlank()
                        }
                    },
                    label = { Text("Số điện thoại *") },
                    placeholder = { Text("0123456789 hoặc +84123456789") },
                    isError = phoneError,
                    modifier = Modifier.fillMaxWidth(),
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Phone),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedTextColor = DarkOnSurface,
                        unfocusedTextColor = DarkOnSurface,
                        focusedBorderColor = DarkPrimary,
                        unfocusedBorderColor = DarkOnSurface.copy(alpha = 0.3f),
                        focusedLabelColor = DarkPrimary,
                        unfocusedLabelColor = DarkOnSurface.copy(alpha = 0.6f),
                        cursorColor = DarkPrimary
                    )
                )
                
                OutlinedTextField(
                    value = relationship,
                    onValueChange = { 
                        relationship = it
                        relationshipError = it.isBlank()
                    },
                    label = { Text("Mối quan hệ *") },
                    isError = relationshipError,
                    placeholder = { Text("Ví dụ: Bố, Mẹ, Bạn bè...") },
                    modifier = Modifier.fillMaxWidth(),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedTextColor = DarkOnSurface,
                        unfocusedTextColor = DarkOnSurface,
                        focusedBorderColor = DarkPrimary,
                        unfocusedBorderColor = DarkOnSurface.copy(alpha = 0.3f),
                        focusedLabelColor = DarkPrimary,
                        unfocusedLabelColor = DarkOnSurface.copy(alpha = 0.6f),
                        cursorColor = DarkPrimary
                    )
                )
                
                OutlinedTextField(
                    value = address,
                    onValueChange = { address = it },
                    label = { Text("Địa chỉ") },
                    modifier = Modifier.fillMaxWidth(),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedTextColor = DarkOnSurface,
                        unfocusedTextColor = DarkOnSurface,
                        focusedBorderColor = DarkPrimary,
                        unfocusedBorderColor = DarkOnSurface.copy(alpha = 0.3f),
                        focusedLabelColor = DarkPrimary,
                        unfocusedLabelColor = DarkOnSurface.copy(alpha = 0.6f),
                        cursorColor = DarkPrimary
                    )
                )
                
                OutlinedTextField(
                    value = note,
                    onValueChange = { note = it },
                    label = { Text("Ghi chú") },
                    modifier = Modifier.fillMaxWidth(),
                    minLines = 2,
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedTextColor = DarkOnSurface,
                        unfocusedTextColor = DarkOnSurface,
                        focusedBorderColor = DarkPrimary,
                        unfocusedBorderColor = DarkOnSurface.copy(alpha = 0.3f),
                        focusedLabelColor = DarkPrimary,
                        unfocusedLabelColor = DarkOnSurface.copy(alpha = 0.6f),
                        cursorColor = DarkPrimary
                    )
                )
            }
        },
        confirmButton = {
            TextButton(
                onClick = {
                    val hasError = name.isBlank() || phoneNumber.isBlank() || relationship.isBlank()
                    nameError = name.isBlank()
                    phoneError = phoneNumber.isBlank()
                    relationshipError = relationship.isBlank()
                    
                    if (!hasError) {
                        onConfirm(
                            EmergencyContactService.EmergencyContactRequest(
                                name = name,
                                phoneNumber = phoneNumber,
                                address = address.ifBlank { null },
                                relationship = relationship,
                                note = note.ifBlank { null }
                            )
                        )
                    }
                }
            ) {
                Text("Lưu", color = DarkPrimary)
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Hủy", color = DarkOnSurface)
            }
        },
        containerColor = DarkSurface
    )
}
