package com.auto_fe.auto_fe.ui.screens

import android.net.Uri
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.AttachFile
import androidx.compose.material.icons.filled.Cancel
import androidx.compose.material.icons.filled.Image
import androidx.compose.material.icons.filled.InsertDriveFile
import androidx.compose.material.icons.filled.Send
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import coil.compose.AsyncImage
import com.auto_fe.auto_fe.models.ChatMessage
import com.auto_fe.auto_fe.ui.service.ChatService
import com.auto_fe.auto_fe.ui.service.WebSocketManager
import com.auto_fe.auto_fe.ui.theme.*
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatDetailScreen(
    accessToken: String,
    currentUserId: Long,
    userEmail: String,            // Email của user hiện tại (để subscribe WebSocket)
    chatId: Long? = null,        // Chat ID (luôn có sau khi create)
    receiverId: Long? = null,    // Deprecated - không dùng nữa
    chatName: String? = null,
    onBackClick: () -> Unit
) {
    val scope = rememberCoroutineScope()
    val context = LocalContext.current
    val chatService = remember { ChatService() }
    val wsManager = remember { WebSocketManager() }
    val listState = rememberLazyListState()

    var messages by remember { mutableStateOf<List<ChatMessage>>(emptyList()) }
    var messageText by remember { mutableStateOf("") }
    var isLoading by remember { mutableStateOf(false) }
    var isLoadingMore by remember { mutableStateOf(false) }
    var currentPage by remember { mutableStateOf(0) }
    var hasMorePages by remember { mutableStateOf(true) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var wsConnected by remember { mutableStateOf(false) }
    
    // Image viewer states
    var showImageViewer by remember { mutableStateOf(false) }
    var selectedImageUrl by remember { mutableStateOf<String?>(null) }
    
    // File upload states
    var selectedFileUri by remember { mutableStateOf<Uri?>(null) }
    var selectedFileName by remember { mutableStateOf<String?>(null) }
    var selectedFileType by remember { mutableStateOf<String?>(null) } // "IMAGE" or "FILE"
    var isUploading by remember { mutableStateOf(false) }
    var showAttachmentMenu by remember { mutableStateOf(false) }
    
    // File picker launchers
    val imagePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            selectedFileUri = it
            selectedFileName = chatService.getFileName(context, it)
            selectedFileType = "IMAGE"
            Log.d("ChatDetailScreen", "Image selected: $selectedFileName")
        }
    }
    
    val filePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            selectedFileUri = it
            selectedFileName = chatService.getFileName(context, it)
            selectedFileType = "FILE"
            Log.d("ChatDetailScreen", "File selected: $selectedFileName")
        }
    }

    // Connect WebSocket when screen opens
    LaunchedEffect(Unit) {
        Log.d("ChatDetailScreen", "=== WebSocket Connection Start ===")
        Log.d("ChatDetailScreen", "User email: $userEmail")
        Log.d("ChatDetailScreen", "Chat ID: $chatId")
        Log.d("ChatDetailScreen", "Access token length: ${accessToken.length}")
        
        wsManager.connect(
            accessToken = accessToken,
            onConnected = {
                wsConnected = true
                Log.d("ChatDetailScreen", "WebSocket connected successfully")
                
                // Subscribe to chat topic to receive real-time messages
                chatId?.let { id ->
                    val topicSubscription = wsManager.subscribeToTopic(
                        topic = "/topic/chat-$id",
                        onMessageReceived = { newMessage ->
                            Log.d("ChatDetailScreen", "📨 New message received: ${newMessage.content}")
                            
                            // Add message to UI
                            messages = messages + newMessage
                            
                            // Auto scroll to bottom (index 0 in reversed layout)
                            scope.launch {
                                if (messages.isNotEmpty()) {
                                    listState.animateScrollToItem(0)
                                }
                            }
                        }
                    )
                    
                    if (topicSubscription != null) {
                        Log.d("ChatDetailScreen", "Subscribed to /topic/chat-$id")
                    } else {
                        Log.e("ChatDetailScreen", "Failed to subscribe to topic")
                    }
                }
            },
            onError = { error ->
                Log.e("ChatDetailScreen", "WebSocket connection error: $error")
                wsConnected = false
            }
        )
    }

    // Disconnect WebSocket when screen closes
    DisposableEffect(Unit) {
        onDispose {
            Log.d("ChatDetailScreen", "Disconnecting WebSocket")
            wsManager.disconnect()
        }
    }

    // Load messages if chatId exists
    LaunchedEffect(chatId) {
        chatId?.let { id ->
            scope.launch {
                isLoading = true
                errorMessage = null
                currentPage = 0
                val result = chatService.getMessages(id, 0, 50, accessToken)
                result.fold(
                    onSuccess = { pageResponse ->
                        messages = pageResponse.content.reversed() // Reverse to show newest at bottom
                        hasMorePages = currentPage < pageResponse.totalPages - 1
                        isLoading = false
                        // No need to scroll - reverseLayout will show newest (index 0) at bottom automatically
                        
                        // Mark all messages as read
                        scope.launch {
                            chatService.markAsRead(id, accessToken)
                            Log.d("ChatDetailScreen", "Marked all messages as read")
                        }
                    },
                    onFailure = { error ->
                        errorMessage = "Không thể tải tin nhắn: ${error.message}"
                        isLoading = false
                    }
                )
            }
        }
    }

    // Load more messages function
    val loadMoreMessages: () -> Unit = {
        if (!isLoadingMore && hasMorePages && chatId != null) {
            scope.launch {
                isLoadingMore = true
                val nextPage = currentPage + 1
                val result = chatService.getMessages(chatId, nextPage, 50, accessToken)
                result.fold(
                    onSuccess = { pageResponse ->
                        // Prepend older messages to the beginning
                        messages = pageResponse.content.reversed() + messages
                        hasMorePages = nextPage < pageResponse.totalPages - 1
                        currentPage = nextPage
                        isLoadingMore = false
                        
                        Log.d("ChatDetailScreen", "Loaded page $nextPage, total messages: ${messages.size}")
                    },
                    onFailure = { error ->
                        Log.e("ChatDetailScreen", "Failed to load more messages", error)
                        isLoadingMore = false
                    }
                )
            }
        }
    }

    // Detect when user scrolls to top (bottom in reversed layout) to load more
    LaunchedEffect(listState) {
        snapshotFlow { 
            listState.layoutInfo.visibleItemsInfo.lastOrNull()?.index ?: 0
        }
            .collect { lastVisibleIndex ->
                val totalItems = listState.layoutInfo.totalItemsCount
                // Load more when scrolled near end (which is top due to reverseLayout)
                if (totalItems > 0 && lastVisibleIndex >= totalItems - 3 && !isLoadingMore && hasMorePages) {
                    loadMoreMessages()
                }
            }
    }

    // Send message function
    val sendMessage: () -> Unit = {
        Log.d("ChatDetailScreen", "[SEND-TRIGGER] messageText='$messageText', selectedFileUri=$selectedFileUri, isUploading=$isUploading")
        
        if ((messageText.isNotBlank() || selectedFileUri != null) && chatId != null && !isUploading) {
            scope.launch {
                try {
                    val content = messageText.trim()
                    messageText = "" // Clear input immediately
                    
                    var attachmentUrl: String? = null
                    var attachmentName: String? = null
                    var attachmentType: String? = null
                    var attachmentSize: Long? = null
                    var messageType = "TEXT"
                    
                    // Upload file if selected
                    selectedFileUri?.let { uri ->
                        Log.d("ChatDetailScreen", "[SEND-UPLOAD] Starting upload for URI=$uri, type=$selectedFileType")
                        isUploading = true
                        
                        val uploadResult = when (selectedFileType) {
                            "IMAGE" -> chatService.uploadImage(uri, context, accessToken)
                            "FILE" -> chatService.uploadFile(uri, context, accessToken)
                            else -> null
                        }
                        
                        uploadResult?.fold(
                            onSuccess = { uploadData ->
                                attachmentUrl = uploadData["url"] as? String
                                attachmentName = (uploadData["originalFilename"] as? String) ?: selectedFileName
                                attachmentType = uploadData["format"] as? String
                                attachmentSize = (uploadData["bytes"] as? Number)?.toLong()
                                messageType = selectedFileType ?: "TEXT"
                                
                                Log.d("ChatDetailScreen", "[SEND-UPLOAD] Upload success, URL=$attachmentUrl")
                                
                                // Clear selection
                                selectedFileUri = null
                                selectedFileName = null
                                selectedFileType = null
                            },
                            onFailure = { error ->
                                errorMessage = "Không thể tải lên file: ${error.message}"
                                Log.e("ChatDetailScreen", "[SEND-UPLOAD] Upload failed: ${error.message}")
                                isUploading = false
                                return@launch
                            }
                        )
                        
                        isUploading = false
                    }
                    
                    Log.d("ChatDetailScreen", "[SEND-MESSAGE] Calling sendMessage API...")
                    // Send message via REST API (more reliable)
                    val result = chatService.sendMessage(
                        chatId = chatId,
                        receiverId = null,
                        content = content.ifBlank { attachmentName ?: "File" },
                        accessToken = accessToken,
                        messageType = messageType,
                        attachmentUrl = attachmentUrl,
                        attachmentName = attachmentName,
                        attachmentType = attachmentType,
                        attachmentSize = attachmentSize
                    )
                    
                    result.fold(
                        onSuccess = { sentMessage ->
                            // Add message to list if not already added by WebSocket
                            if (!messages.any { it.id == sentMessage.id }) {
                                messages = messages + sentMessage
                                // Auto scroll to bottom (index 0 in reversed layout)
                                scope.launch {
                                    listState.animateScrollToItem(0)
                                }
                            }
                        },
                        onFailure = { error ->
                            errorMessage = "Không thể gửi tin nhắn: ${error.message}"
                            Log.e("ChatDetailScreen", "Send message failed", error)
                            messageText = content // Restore message on error
                        }
                    )
                } catch (e: Exception) {
                    errorMessage = "Đã xảy ra lỗi: ${e.message}"
                    Log.e("ChatDetailScreen", "Error in sendMessage", e)
                    isUploading = false
                }
            }
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        // Chat name
                        Text(
                            text = chatName ?: "Chat",
                            fontWeight = FontWeight.Bold,
                            color = DarkOnSurface
                        )
                        // Status indicator dot at the end
                        Box(
                            modifier = Modifier
                                .size(8.dp)
                                .background(
                                    color = if (wsConnected) Color(0xFF4CAF50) else Color.Gray.copy(alpha = 0.3f),
                                    shape = RoundedCornerShape(50)
                                )
                        )
                    }
                },
                navigationIcon = {
                    IconButton(onClick = onBackClick) {
                        Icon(
                            Icons.Default.ArrowBack,
                            contentDescription = "Back",
                            tint = DarkOnSurface
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = DarkSurface.copy(alpha = 0.95f)
                )
            )
        },
        containerColor = AIBackgroundDeep
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .imePadding() // Auto adjust when keyboard appears
        ) {
            // Messages List
            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
            ) {
                when {
                    isLoading -> {
                        CircularProgressIndicator(
                            modifier = Modifier.align(Alignment.Center),
                            color = DarkPrimary
                        )
                    }
                    errorMessage != null -> {
                        Text(
                            text = errorMessage ?: "",
                            color = AIError,
                            modifier = Modifier
                                .align(Alignment.Center)
                                .padding(16.dp)
                        )
                    }
                    messages.isEmpty() -> {
                        Text(
                            text = "Chưa có tin nhắn.\nGửi tin nhắn đầu tiên!",
                            color = DarkOnSurface.copy(alpha = 0.6f),
                            modifier = Modifier
                                .align(Alignment.Center)
                                .padding(16.dp),
                            textAlign = androidx.compose.ui.text.style.TextAlign.Center
                        )
                    }
                    else -> {
                        LazyColumn(
                            state = listState,
                            modifier = Modifier.fillMaxSize(),
                            contentPadding = PaddingValues(16.dp),
                            verticalArrangement = Arrangement.spacedBy(8.dp),
                            reverseLayout = true // Reverse layout so newest at bottom
                        ) {
                            // Messages (reversed order because of reverseLayout)
                            items(messages.reversed()) { message ->
                                MessageBubble(
                                    message = message,
                                    isCurrentUser = message.senderId == currentUserId,
                                    onImageClick = { imageUrl ->
                                        selectedImageUrl = imageUrl
                                        showImageViewer = true
                                    }
                                )
                            }
                            
                            // Loading more indicator at bottom (will appear at top due to reverseLayout)
                            if (isLoadingMore) {
                                item {
                                    Box(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .padding(vertical = 8.dp),
                                        contentAlignment = Alignment.Center
                                    ) {
                                        CircularProgressIndicator(
                                            modifier = Modifier.size(24.dp),
                                            color = DarkPrimary,
                                            strokeWidth = 2.dp
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Input Area
            Card(
                modifier = Modifier
                    .fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = DarkSurface
                ),
                shape = RoundedCornerShape(0.dp),
                elevation = CardDefaults.cardElevation(defaultElevation = 8.dp)
            ) {
                Column(modifier = Modifier.fillMaxWidth()) {
                    // File preview if file is selected
                    selectedFileUri?.let { uri ->
                        Card(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(8.dp),
                            colors = CardDefaults.cardColors(
                                containerColor = AIBackgroundSoft
                            )
                        ) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(12.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Icon(
                                    imageVector = if (selectedFileType == "IMAGE") Icons.Default.Image else Icons.Default.InsertDriveFile,
                                    contentDescription = null,
                                    tint = DarkPrimary,
                                    modifier = Modifier.size(32.dp)
                                )
                                Spacer(modifier = Modifier.width(12.dp))
                                Column(modifier = Modifier.weight(1f)) {
                                    Text(
                                        text = selectedFileName ?: "File đã chọn",
                                        color = DarkOnSurface,
                                        style = MaterialTheme.typography.bodyMedium
                                    )
                                    Text(
                                        text = if (selectedFileType == "IMAGE") "Hình ảnh" else "Tệp đính kèm",
                                        color = DarkOnSurface.copy(alpha = 0.6f),
                                        style = MaterialTheme.typography.bodySmall
                                    )
                                }
                                IconButton(
                                    onClick = {
                                        selectedFileUri = null
                                        selectedFileName = null
                                        selectedFileType = null
                                    }
                                ) {
                                    Icon(
                                        Icons.Default.Cancel,
                                        contentDescription = "Hủy",
                                        tint = DarkOnSurface.copy(alpha = 0.6f)
                                    )
                                }
                            }
                        }
                    }
                    
                    // Upload progress indicator
                    if (isUploading) {
                        LinearProgressIndicator(
                            modifier = Modifier.fillMaxWidth(),
                            color = DarkPrimary
                        )
                    }
                    
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(8.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        // Attachment button with dropdown menu
                        Box {
                            IconButton(
                                onClick = { showAttachmentMenu = true }
                            ) {
                                Icon(
                                    Icons.Default.AttachFile,
                                    contentDescription = "Đính kèm",
                                    tint = DarkPrimary
                                )
                            }
                            
                            DropdownMenu(
                                expanded = showAttachmentMenu,
                                onDismissRequest = { showAttachmentMenu = false }
                            ) {
                                DropdownMenuItem(
                                    text = { 
                                        Row(verticalAlignment = Alignment.CenterVertically) {
                                            Icon(Icons.Default.Image, contentDescription = null, modifier = Modifier.size(20.dp))
                                            Spacer(modifier = Modifier.width(8.dp))
                                            Text("Chọn hình ảnh")
                                        }
                                    },
                                    onClick = {
                                        showAttachmentMenu = false
                                        imagePickerLauncher.launch("image/*")
                                    }
                                )
                                DropdownMenuItem(
                                    text = { 
                                        Row(verticalAlignment = Alignment.CenterVertically) {
                                            Icon(Icons.Default.InsertDriveFile, contentDescription = null, modifier = Modifier.size(20.dp))
                                            Spacer(modifier = Modifier.width(8.dp))
                                            Text("Chọn tệp")
                                        }
                                    },
                                    onClick = {
                                        showAttachmentMenu = false
                                        filePickerLauncher.launch("*/*")
                                    }
                                )
                            }
                        }
                        
                        OutlinedTextField(
                            value = messageText,
                            onValueChange = { messageText = it },
                            modifier = Modifier.weight(1f),
                            placeholder = { Text("Nhập tin nhắn...") },
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedContainerColor = AIBackgroundSoft,
                                unfocusedContainerColor = AIBackgroundSoft,
                                focusedTextColor = DarkOnSurface,
                                unfocusedTextColor = DarkOnSurface,
                                focusedBorderColor = DarkPrimary,
                                unfocusedBorderColor = DarkOnSurface.copy(alpha = 0.3f),
                                cursorColor = DarkPrimary
                            ),
                        shape = RoundedCornerShape(24.dp),
                        maxLines = 5
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    IconButton(
                        onClick = sendMessage,
                        enabled = (messageText.isNotBlank() || selectedFileUri != null) && !isUploading
                    ) {
                        Icon(
                            Icons.Default.Send,
                            contentDescription = "Send",
                            tint = if ((messageText.isNotBlank() || selectedFileUri != null) && !isUploading) 
                                DarkPrimary 
                            else 
                                DarkOnSurface.copy(alpha = 0.3f)
                        )
                    }
                }
                }
            }
        }
    }
    
    // Image Viewer Dialog
    if (showImageViewer && selectedImageUrl != null) {
        ImageViewerDialog(
            imageUrl = selectedImageUrl!!,
            onDismiss = { 
                showImageViewer = false
                selectedImageUrl = null
            }
        )
    }
}

@Composable
private fun ImageViewerDialog(
    imageUrl: String,
    onDismiss: () -> Unit
) {
    var scale by remember { mutableStateOf(1f) }
    var offsetX by remember { mutableStateOf(0f) }
    var offsetY by remember { mutableStateOf(0f) }
    
    Dialog(
        onDismissRequest = onDismiss,
        properties = DialogProperties(
            usePlatformDefaultWidth = false,
            decorFitsSystemWindows = false
        )
    ) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(Color.Black)
                .pointerInput(scale) {
                    // Only allow tap to dismiss when not zoomed
                    if (scale == 1f) {
                        detectTapGestures { onDismiss() }
                    }
                }
        ) {
            // Fullscreen image with zoom and pan
            AsyncImage(
                model = imageUrl,
                contentDescription = "Full image",
                contentScale = ContentScale.Fit,
                modifier = Modifier
                    .fillMaxSize()
                    .align(Alignment.Center)
                    .pointerInput(Unit) {
                        detectTransformGestures { _, pan, zoom, _ ->
                            scale = (scale * zoom).coerceIn(0.5f, 5f)
                            
                            // Only allow pan if zoomed in
                            if (scale > 1f) {
                                offsetX += pan.x
                                offsetY += pan.y
                            } else {
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
                    )
            )
            
            // Top bar with controls
            Row(
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(16.dp)
                    .background(
                        Color.Black.copy(alpha = 0.6f),
                        shape = RoundedCornerShape(24.dp)
                    )
                    .padding(horizontal = 8.dp, vertical = 4.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                // Zoom percentage
                Text(
                    text = "${(scale * 100).toInt()}%",
                    color = Color.White,
                    fontSize = AppTextSize.bodySmall,
                    modifier = Modifier.padding(horizontal = 8.dp)
                )
                
                // Reset zoom button
                if (scale != 1f || offsetX != 0f || offsetY != 0f) {
                    IconButton(
                        onClick = {
                            scale = 1f
                            offsetX = 0f
                            offsetY = 0f
                        },
                        modifier = Modifier.size(32.dp)
                    ) {
                        Text(
                            text = "⟲",
                            color = Color.White,
                            fontSize = AppTextSize.bodyLarge
                        )
                    }
                }
                
                // Close button
                IconButton(
                    onClick = onDismiss,
                    modifier = Modifier.size(32.dp)
                ) {
                    Icon(
                        Icons.Default.Cancel,
                        contentDescription = "Close",
                        tint = Color.White,
                        modifier = Modifier.size(24.dp)
                    )
                }
            }
            
            // Hint text at bottom
            if (scale == 1f) {
                Text(
                    text = "Pinch để zoom • Tap để đóng",
                    color = Color.White.copy(alpha = 0.7f),
                    fontSize = AppTextSize.bodySmall,
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .padding(16.dp)
                        .background(
                            Color.Black.copy(alpha = 0.5f),
                            shape = RoundedCornerShape(16.dp)
                        )
                        .padding(horizontal = 16.dp, vertical = 8.dp)
                )
            }
        }
    }
}

@Composable
private fun MessageBubble(
    message: ChatMessage,
    isCurrentUser: Boolean,
    onImageClick: (String) -> Unit = {}
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (isCurrentUser) Arrangement.End else Arrangement.Start
    ) {
        Card(
            modifier = Modifier
                .widthIn(max = 280.dp),
            colors = CardDefaults.cardColors(
                containerColor = if (isCurrentUser) DarkPrimary else DarkSurface
            ),
            shape = RoundedCornerShape(
                topStart = 16.dp,
                topEnd = 16.dp,
                bottomStart = if (isCurrentUser) 16.dp else 4.dp,
                bottomEnd = if (isCurrentUser) 4.dp else 16.dp
            ),
            elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
        ) {
            Column(
                modifier = Modifier.padding(12.dp)
            ) {
                // Display attachment if present
                when (message.messageType) {
                    "IMAGE" -> {
                        message.attachmentUrl?.let { imageUrl ->
                            AsyncImage(
                                model = imageUrl,
                                contentDescription = message.attachmentName ?: "Image",
                                contentScale = ContentScale.Crop,
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .heightIn(max = 200.dp)
                                    .clip(RoundedCornerShape(12.dp))
                                    .clickable {
                                        onImageClick(imageUrl)
                                    }
                                    .padding(bottom = 8.dp)
                            )
                        }
                        if (message.content.isNotBlank() && message.content != (message.attachmentName ?: "")) {
                            Text(
                                text = message.content,
                                color = if (isCurrentUser) DarkOnPrimary else DarkOnSurface,
                                fontSize = AppTextSize.bodyMedium
                            )
                        }
                    }
                    "FILE" -> {
                        message.attachmentUrl?.let { fileUrl ->
                            Card(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(bottom = 8.dp)
                                    .clickable {
                                        // TODO: Handle file download/open
                                        Log.d("ChatDetailScreen", "Open file: $fileUrl")
                                    },
                                colors = CardDefaults.cardColors(
                                    containerColor = if (isCurrentUser) 
                                        DarkOnPrimary.copy(alpha = 0.1f) 
                                    else 
                                        DarkPrimary.copy(alpha = 0.1f)
                                )
                            ) {
                                Row(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(8.dp),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Icon(
                                        Icons.Default.InsertDriveFile,
                                        contentDescription = null,
                                        tint = if (isCurrentUser) DarkOnPrimary else DarkOnSurface,
                                        modifier = Modifier.size(32.dp)
                                    )
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Column(modifier = Modifier.weight(1f)) {
                                        Text(
                                            text = message.attachmentName ?: "File",
                                            color = if (isCurrentUser) DarkOnPrimary else DarkOnSurface,
                                            fontSize = AppTextSize.bodySmall,
                                            maxLines = 1
                                        )
                                        message.attachmentSize?.let { size ->
                                            Text(
                                                text = formatFileSize(size),
                                                color = if (isCurrentUser) 
                                                    DarkOnPrimary.copy(alpha = 0.7f) 
                                                else 
                                                    DarkOnSurface.copy(alpha = 0.6f),
                                                fontSize = AppTextSize.bodySmall
                                            )
                                        }
                                    }
                                }
                            }
                        }
                        if (message.content.isNotBlank() && message.content != (message.attachmentName ?: "")) {
                            Text(
                                text = message.content,
                                color = if (isCurrentUser) DarkOnPrimary else DarkOnSurface,
                                fontSize = AppTextSize.bodyMedium
                            )
                        }
                    }
                    else -> {
                        // TEXT message
                        Text(
                            text = message.content,
                            color = if (isCurrentUser) DarkOnPrimary else DarkOnSurface,
                            fontSize = AppTextSize.bodyMedium
                        )
                    }
                }
                
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = formatTime(message.createdAt),
                    color = if (isCurrentUser) 
                        DarkOnPrimary.copy(alpha = 0.7f) 
                    else 
                        DarkOnSurface.copy(alpha = 0.6f),
                    fontSize = AppTextSize.bodySmall
                )
            }
        }
    }
}

private fun formatFileSize(bytes: Long): String {
    return when {
        bytes < 1024 -> "$bytes B"
        bytes < 1024 * 1024 -> String.format("%.1f KB", bytes / 1024.0)
        bytes < 1024 * 1024 * 1024 -> String.format("%.1f MB", bytes / (1024.0 * 1024))
        else -> String.format("%.1f GB", bytes / (1024.0 * 1024 * 1024))
    }
}

private fun formatTime(timestamp: String?): String {
    if (timestamp == null) return ""
    return try {
        val instant = java.time.Instant.parse(timestamp)
        val localTime = java.time.LocalDateTime.ofInstant(
            instant,
            java.time.ZoneId.systemDefault()
        )
        val formatter = java.time.format.DateTimeFormatter.ofPattern("HH:mm")
        localTime.format(formatter)
    } catch (e: Exception) {
        Log.e("ChatDetailScreen", "Error formatting time: $timestamp", e)
        ""
    }
}
