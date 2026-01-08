package com.auto_fe.auto_fe.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
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
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import com.auto_fe.auto_fe.service.be.RelationshipService
import com.auto_fe.auto_fe.service.be.UserService
import com.auto_fe.auto_fe.ui.theme.*
import kotlinx.coroutines.launch

/**
 * Màn hình tìm kiếm user để gửi yêu cầu kết nối
 * (Khác với SearchUserScreen dùng cho chat)
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SearchUserForConnectionScreen(
    accessToken: String,
    onBack: () -> Unit
) {
    val scope = rememberCoroutineScope()
    val userService = remember { UserService() }
    val relationshipService = remember { RelationshipService() }
    
    var searchQuery by remember { mutableStateOf("") }
    var searchResults by remember { mutableStateOf<List<UserService.ProfileData>>(emptyList()) }
    var isSearching by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    
    // Dialog gửi yêu cầu
    var showSendRequestDialog by remember { mutableStateOf(false) }
    var selectedUser by remember { mutableStateOf<UserService.ProfileData?>(null) }
    var requestMessage by remember { mutableStateOf("") }
    var isSendingRequest by remember { mutableStateOf(false) }
    
    // Snackbar state
    val snackbarHostState = remember { SnackbarHostState() }
    var snackbarMessage by remember { mutableStateOf<String?>(null) }
    var isSuccessMessage by remember { mutableStateOf(false) }
    
    // Show snackbar when message changes
    LaunchedEffect(snackbarMessage) {
        snackbarMessage?.let { message ->
            snackbarHostState.showSnackbar(
                message = message,
                duration = SnackbarDuration.Short
            )
            snackbarMessage = null
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Tìm người để kết nối", color = DarkOnPrimary) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Default.ArrowBack, "Quay lại", tint = DarkOnPrimary)
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = DarkPrimary
                )
            )
        },
        snackbarHost = {
            SnackbarHost(hostState = snackbarHostState) { data ->
                Snackbar(
                    snackbarData = data,
                    containerColor = if (isSuccessMessage) DarkPrimary else DarkError,
                    contentColor = DarkOnPrimary
                )
            }
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(
                    brush = Brush.verticalGradient(
                        colors = listOf(AIBackgroundDeep, AIBackgroundSoft)
                    )
                )
                .padding(paddingValues)
                .padding(16.dp)
        ) {
            // Search Bar
            OutlinedTextField(
                value = searchQuery,
                onValueChange = { searchQuery = it },
                modifier = Modifier.fillMaxWidth(),
                placeholder = { Text("Nhập email hoặc tên...", color = DarkOnSurface.copy(0.6f)) },
                leadingIcon = {
                    Icon(Icons.Default.Search, null, tint = DarkPrimary)
                },
                trailingIcon = {
                    if (searchQuery.isNotEmpty()) {
                        IconButton(onClick = { searchQuery = "" }) {
                            Icon(Icons.Default.Close, "Xóa", tint = DarkOnSurface)
                        }
                    }
                },
                singleLine = true,
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = DarkPrimary,
                    unfocusedBorderColor = DarkOnSurface.copy(0.3f),
                    focusedTextColor = DarkOnSurface,
                    unfocusedTextColor = DarkOnSurface
                ),
                shape = RoundedCornerShape(12.dp)
            )

            Spacer(modifier = Modifier.height(8.dp))

            // Search Button
            Button(
                onClick = {
                    if (searchQuery.isNotBlank()) {
                        scope.launch {
                            isSearching = true
                            errorMessage = null
                            userService.searchUsers(accessToken, searchQuery)
                                .onSuccess { results ->
                                    searchResults = results
                                    isSearching = false
                                }
                                .onFailure { error ->
                                    errorMessage = error.message
                                    isSearching = false
                                }
                        }
                    }
                },
                modifier = Modifier.fillMaxWidth(),
                enabled = searchQuery.isNotBlank() && !isSearching,
                colors = ButtonDefaults.buttonColors(containerColor = DarkPrimary)
            ) {
                if (isSearching) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(20.dp),
                        color = DarkOnPrimary
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                }
                Text("Tìm kiếm", color = DarkOnPrimary)
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Error message
            errorMessage?.let { error ->
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(containerColor = DarkError.copy(0.1f)),
                    shape = RoundedCornerShape(8.dp)
                ) {
                    Text(
                        text = error,
                        color = DarkError,
                        modifier = Modifier.padding(12.dp)
                    )
                }
                Spacer(modifier = Modifier.height(8.dp))
            }

            // Search Results
            if (searchResults.isNotEmpty()) {
                Text(
                    "Kết quả tìm kiếm (${searchResults.size})",
                    fontWeight = FontWeight.Bold,
                    color = DarkOnSurface,
                    modifier = Modifier.padding(bottom = 8.dp)
                )

                LazyColumn(
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    items(searchResults) { user ->
                        UserSearchResultCard(
                            user = user,
                            onClick = {
                                selectedUser = user
                                showSendRequestDialog = true
                            }
                        )
                    }
                }
            } else if (!isSearching && searchQuery.isNotEmpty()) {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        "Không tìm thấy người dùng",
                        color = DarkOnSurface.copy(0.6f)
                    )
                }
            }
        }
    }

    // Dialog gửi yêu cầu kết nối
    if (showSendRequestDialog && selectedUser != null) {
        AlertDialog(
            onDismissRequest = { 
                if (!isSendingRequest) {
                    showSendRequestDialog = false
                    requestMessage = ""
                }
            },
            title = { Text("Gửi yêu cầu kết nối", color = DarkOnSurface) },
            text = {
                Column {
                    Text(
                        "Gửi yêu cầu kết nối đến ${selectedUser?.fullName ?: "người dùng này"}?",
                        color = DarkOnSurface
                    )
                    Spacer(modifier = Modifier.height(12.dp))
                    OutlinedTextField(
                        value = requestMessage,
                        onValueChange = { requestMessage = it },
                        placeholder = { Text("Lời nhắn (tùy chọn)...") },
                        modifier = Modifier.fillMaxWidth(),
                        maxLines = 3,
                        enabled = !isSendingRequest
                    )
                }
            },
            confirmButton = {
                Button(
                    onClick = {
                        scope.launch {
                            isSendingRequest = true
                            selectedUser?.let { user ->
                                user.id?.let { userId ->
                                    relationshipService.sendRequest(
                                        accessToken = accessToken,
                                        targetUserId = userId,
                                        message = requestMessage.ifBlank { null }
                                    ).onSuccess {
                                        showSendRequestDialog = false
                                        requestMessage = ""
                                        isSendingRequest = false
                                        isSuccessMessage = true
                                        snackbarMessage = "Đã gửi yêu cầu kết nối đến ${user.fullName}"
                                    }.onFailure { error ->
                                        showSendRequestDialog = false
                                        requestMessage = ""
                                        isSendingRequest = false
                                        isSuccessMessage = false
                                        snackbarMessage = getVietnameseErrorMessage(error.message)
                                    }
                                } ?: run {
                                    isSendingRequest = false
                                    isSuccessMessage = false
                                    snackbarMessage = "Không thể gửi yêu cầu: Thông tin người dùng không hợp lệ"
                                }
                            }
                        }
                    },
                    enabled = !isSendingRequest,
                    colors = ButtonDefaults.buttonColors(containerColor = DarkPrimary)
                ) {
                    if (isSendingRequest) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(16.dp),
                            color = DarkOnPrimary
                        )
                    } else {
                        Text("Gửi", color = DarkOnPrimary)
                    }
                }
            },
            dismissButton = {
                TextButton(
                    onClick = { 
                        showSendRequestDialog = false
                        requestMessage = ""
                    },
                    enabled = !isSendingRequest
                ) {
                    Text("Hủy", color = DarkOnSurface)
                }
            },
            containerColor = DarkSurface
        )
    }
}

@Composable
private fun UserSearchResultCard(
    user: UserService.ProfileData,
    onClick: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        colors = CardDefaults.cardColors(containerColor = DarkSurface),
        shape = RoundedCornerShape(12.dp),
        elevation = CardDefaults.cardElevation(2.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Avatar
            Box(
                modifier = Modifier
                    .size(48.dp)
                    .clip(CircleShape)
                    .background(DarkPrimary.copy(0.2f)),
                contentAlignment = Alignment.Center
            ) {
                if (!user.avatar.isNullOrBlank()) {
                    AsyncImage(
                        model = user.avatar,
                        contentDescription = "Avatar",
                        modifier = Modifier
                            .fillMaxSize()
                            .clip(CircleShape)
                    )
                } else {
                    Icon(
                        Icons.Default.Person,
                        contentDescription = null,
                        tint = DarkPrimary,
                        modifier = Modifier.size(28.dp)
                    )
                }
            }

            Spacer(modifier = Modifier.width(12.dp))

            // User Info
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = user.fullName ?: "No Name",
                    fontWeight = FontWeight.Bold,
                    color = DarkOnSurface,
                    fontSize = AppTextSize.bodyMedium
                )
                Text(
                    text = user.email ?: "",
                    color = DarkOnSurface.copy(0.7f),
                    fontSize = AppTextSize.bodySmall
                )
                // Role badge
                Text(
                    text = when (user.role) {
                        "ELDER" -> "🧓 Elder"
                        "SUPERVISOR" -> "👨‍⚕️ Supervisor"
                        else -> "👤 User"
                    },
                    color = DarkPrimary,
                    fontSize = AppTextSize.bodySmall,
                    fontWeight = FontWeight.Medium
                )
            }

            // Arrow
            Icon(
                Icons.Default.Send,
                contentDescription = "Gửi yêu cầu",
                tint = DarkPrimary
            )
        }
    }
}

private fun getVietnameseErrorMessage(errorMessage: String?): String {
    return "Gửi yêu cầu thất bại, vui lòng thử lại"
}
